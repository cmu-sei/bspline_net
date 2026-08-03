#!/usr/bin/env bash
# ============================================================================
# Koopman trajectory-model sweep launcher
# ----------------------------------------------------------------------------
# Main sweep:  lifted_dim x seed (rollout 251) + rollout variants + 1 physics-on
# Ablation:    batch_size x interval, one-axis-at-a-time off a (2500, 95%) baseline
# Ablation:    widths_ic (bottleneck depth + uniform width), one-axis-at-a-time
#              off the (2500, 95%, widths_ic=128,64,128) baseline
#
# Design decisions baked in:
#   * FIXED epoch budget (NO EarlyStopping); pick best-smoothed ckpt post-hoc.
#   * min_epochs > alpha_warmup_end so no run stops mid-warmup, AND so the
#     transient post-alpha spike has time to settle (the spikes are transient).
#   * NUMBA threads capped via env BEFORE python starts (in every subprocess),
#     so runs x workers x threads <= physical cores (avoids TBB oversubscription).
#   * Round-robin over visible GPUs, RUNS_PER_GPU concurrent per device,
#     throttled with `wait -n` so we never exceed the concurrency cap.
#   * set -uo pipefail (NOT -e: one failed run must not kill the whole sweep).
# ============================================================================
set -uo pipefail

# ----------------------------- knobs ----------------------------------------
# Launcher-level only. All MODEL/TRAINING constants (epochs, min_epochs, alpha
# warmup, regen_every, num_workers, persistent_workers, monitored metric,
# early-stopping-off, devices=1) live in the config -- see the git tag for the
# exact version run. This script passes ONLY what varies per run.
GPUS=(0 1)                 # visible CUDA device ids
RUNS_PER_GPU=1             # 2 runs per GPU -> total concurrent = len(GPUS)*2 = 4 (testing GPU capacity)
MAX_CONCURRENT=$(( ${#GPUS[@]} * RUNS_PER_GPU ))

# Thread budget (parallelism, not model config): keep
#   concurrent_runs * num_workers * NUMBA_THREADS <= physical cores.
export OMP_NUM_THREADS=1
NUMBA_THREADS=20          
# Shared anchor for the rollout/physics/ablation arms
ANCHOR_LIFTED=128
ANCHOR_ROLLOUT=251
ANCHOR_WIDTHS_IC="128,64,128"   # matches configs/model/traj/default.yaml widths_ic

# Sweep axes
SEEDS=(42 43 44)
LIFTED_DIMS=(64 128 256)
ROLLOUTS_EXTRA=(101 501)   # extra rollouts (251 covered by the dim x seed arm)

# widths_ic ablation axes (see discussion: bottleneck depth vs. uniform width
# are two different questions, kept as separate lists so they can be read
# independently even though both share the ANCHOR_WIDTHS_IC baseline point)
WIDTHS_IC_BOTTLENECK=("128,32,128")                # narrow-middle variant only
WIDTHS_IC_UNIFORM=("64,64,64" "256,256,256")        # off-anchor uniform-width variants only
WIDTHS_IC_SHARED="128,128,128"                      # flat control point, shared by both arms

LOGDIR="sweep_logs"          # flat root; NO datetime. stdout logfiles + manifest + per-run outputs
mkdir -p "outputs/$LOGDIR/traj"
MANIFEST="outputs/$LOGDIR/manifest.tsv"
echo -e "run_name\tgroup\tdevice\tlifted_dim\trollout\tseed\tbatch\tinterval\tintegral_w\twidths_ic\tlogfile" > "$MANIFEST"

# ----------------------------- run builder ----------------------------------
# UNIFORM NAMING SCHEME.  Every run name encodes every swept axis, in fixed
# order, with the value attached:
#
#     d{lifted}_r{rollout}_s{seed}_b{batch}_q{interval_pct}_p{phys_pct}_w{widths}
#
#   e.g.  d128_r251_s42_b2500_q95_p0_w128x64x128   (baseline, physics off)
#         d128_r251_s42_b2500_q95_p25_w128x64x128  (physics on, integral_weight=0.25)
#         d128_r251_s42_b1000_q95_p0_w128x64x128   (ablation: batch=1000)
#         d128_r251_s42_b2500_q95_p0_w128x32x128   (ablation: bottleneck)
#
# Rules:
#   * ALL seven fields always present (even baseline values) -> names parse
#     positionally into a dataframe with split('_'); no field is ever implied.
#   * Integer / dot-free values: q95 not q0.95, p25 not p0.25 (filesystem- and
#     logger-safe). widths_ic uses "x" as the separator (128x64x128) for the
#     same reason -- commas are awkward in filenames/loggers.
#   * Physics is a *field* (p{pct}), not a special suffix. Same treatment for
#     widths_ic: a field (w{...}), not a special suffix.
#   * "Which arm / ablation" is metadata (the `group` column), NOT in the name.
#     Ablation membership is recovered later by FILTERING (e.g. all rows sharing
#     d128/r251/s42, varying b or q or w), so the ablation runs live in the SAME
#     namespace as everything else.
#   * The name is also passed as run_name= so logger/ckpt/logfile all key off
#     the same identifier.

# Each RUNS[] entry is a tab-separated tuple:
RUNS=()
declare -A SEEN   # name -> 1, for dedup (e.g. ablation baseline == a main run)

add_run () {  
  local lifted="$1" rollout="$2" seed="$3" batch="$4" interval="$5" integral_w="$6" group="$7"
  local widths_ic="${8:-$ANCHOR_WIDTHS_IC}"
  # integer-ize the name fields (dot-free)
  local q_pct p_pct w_tag name
  q_pct=$(python -c "print(round(${interval}*100))")     # 0.95 -> 95
  p_pct=$(python -c "print(round(${integral_w}*100))")   # 0.25 -> 25, 0.0 -> 0
  w_tag="w${widths_ic//,/x}"                              # "128,64,128" -> "w128x64x128"
  name="d${lifted}_r${rollout}_s${seed}_b${batch}_q${q_pct}_p${p_pct}_${w_tag}"
  if [[ -n "${SEEN[$name]:-}" ]]; then
    echo "  (dedup) '$name' already queued; skipping duplicate from group=$group"
    return
  fi
  SEEN[$name]=1
  RUNS+=("$name	$lifted	$rollout	$seed	$batch	$interval	$integral_w	$group	$widths_ic")
}

# --- Main arm 1: lifted_dim x seed @ rollout 251, batch 2500, 95% (9 runs) ---
for d in "${LIFTED_DIMS[@]}"; do
  for s in "${SEEDS[@]}"; do
    add_run "$d" "$ANCHOR_ROLLOUT" "$s" 2500 0.95 0.0 main_dim
  done
done

# --- Main arm 2: rollout variants @ lifted 128, 3 seeds, batch 2500, 95% (6) -
for r in "${ROLLOUTS_EXTRA[@]}"; do
  for s in "${SEEDS[@]}"; do
    add_run "$ANCHOR_LIFTED" "$r" "$s" 2500 0.95 0.0 main_rollout
  done
done

# --- Main arm 3: physics-on contrast @ anchor, seed 42 (1 run) ---------------
add_run "$ANCHOR_LIFTED" "$ANCHOR_ROLLOUT" 42 2500 0.95 0.25 physics

# --- Ablation arm: batch x interval, one-axis-at-a-time off (2500,95%) --------
# Anchor lifted=128, rollout=251, seed=42. The (2500,95,p0) baseline point is
# IDENTICAL to the main-sweep run d128_r251_s42_b2500_q95_p0_w128x64x128, so the
# dedup guard above skips re-running it -- you compare the ablation against that
# existing run. (This is exactly the duplication the uniform naming makes visible.)
add_run "$ANCHOR_LIFTED" "$ANCHOR_ROLLOUT" 42 2500 0.95 0.0 abl_batch_interval  # == main; deduped
add_run "$ANCHOR_LIFTED" "$ANCHOR_ROLLOUT" 42 1000 0.95 0.0 abl_batch_interval
add_run "$ANCHOR_LIFTED" "$ANCHOR_ROLLOUT" 42 2000 0.95 0.0 abl_batch_interval
add_run "$ANCHOR_LIFTED" "$ANCHOR_ROLLOUT" 42 2500 0.99 0.0 abl_batch_interval

for w in "${WIDTHS_IC_BOTTLENECK[@]}"; do
  add_run "$ANCHOR_LIFTED" "$ANCHOR_ROLLOUT" 42 2500 0.95 0.0 abl_widths_bottleneck "$w"
done

for w in "${WIDTHS_IC_UNIFORM[@]}"; do
  add_run "$ANCHOR_LIFTED" "$ANCHOR_ROLLOUT" 42 2500 0.95 0.0 abl_widths_uniform "$w"
done

add_run "$ANCHOR_LIFTED" "$ANCHOR_ROLLOUT" 42 2500 0.95 0.0 abl_widths_shared_control "$WIDTHS_IC_SHARED"

echo "Total runs queued: ${#RUNS[@]}  (concurrent cap: $MAX_CONCURRENT)"

# Shuffle run order (avoids any order-dependent issue; re-runs hit a fresh order).
mapfile -t RUNS < <(printf '%s\n' "${RUNS[@]}" | shuf)

# ----------------------------- launch loop -----------------------------------
launch_one () {  # tuple, device
  local tuple="$1" dev="$2"
  IFS=$'\t' read -r name lifted rollout seed batch interval integral_w group widths_ic <<< "$tuple"
  local logf="outputs/$LOGDIR/traj/${name}_log.log"

  echo -e "${name}\t${group}\t${dev}\t${lifted}\t${rollout}\t${seed}\t${batch}\t${interval}\t${integral_w}\t${widths_ic}\t${logf}" >> "$MANIFEST"

  # interval -> quantile_levels list. 95% two-sided -> [0.025, 0.975].
  # (Conformal still reaches the *reported* target on the full calibration set;
  #  this is the trained quantile band.)
  local q_lvls widths_list
  q_lvls=$(python -c "i=${interval}; print(f'[{round((1-i)/2,4)},{round(1-(1-i)/2,4)}]')")
  widths_list="[${widths_ic//,/, }]"   # "128,64,128" -> "[128, 64, 128]" for Hydra
  echo "CMD[$name]: NUMBA_NUM_THREADS=$NUMBA_THREADS CUDA_VISIBLE_DEVICES=$dev train stage=traj trainer.strategy=auto trainer.devices=1 experiment_name=$name trainer.log_dir=${LOGDIR}/${name} model.traj.lifted_dim=$lifted model.traj.rollout_steps=$rollout model.traj.integral_weight=$integral_w model.traj.quantile_levels=$q_lvls model.traj.widths_ic=$widths_list data.batch_size=$batch seed=$seed"
  CUDA_VISIBLE_DEVICES="$dev" \
  NUMBA_NUM_THREADS="$NUMBA_THREADS" \
  OMP_NUM_THREADS="$OMP_NUM_THREADS" \
  train \
      stage=traj \
      trainer.strategy=auto \
      trainer.devices=1 \
      experiment_name="$name" \
      trainer.log_dir="${LOGDIR}" \
      model.traj.lifted_dim="$lifted" \
      model.traj.rollout_steps="$rollout" \
      model.traj.integral_weight="$integral_w" \
      model.traj.quantile_levels="$q_lvls" \
      model.traj.widths_ic="$widths_list" \
      data.batch_size="$batch" \
      seed="$seed" \
      > "$logf" 2>&1 &
}

i=0
for tuple in "${RUNS[@]}"; do
  name="$(echo "$tuple" | cut -f1)"

  ckpt_dir="checkpoints/${LOGDIR}/traj/${name}"
  #MIN_DONE_EPOCH=890   # <-- SET to (epoch budget - small margin)
  #if [ -d "$ckpt_dir" ]; then
  #  max_ep=$(ls "$ckpt_dir" 2>/dev/null | grep -oE 'epoch=[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1)
  #    echo "skip [$((i+1))/${#RUNS[@]}] already done (epoch $max_ep): $name"
  #    i=$((i+1)); continue
  #  fi
  #fi


  # Resume check: skip if this run has EVER been attempted (dir non-empty),
  if [ -d "$ckpt_dir" ] && [ -n "$(ls -A "$ckpt_dir" 2>/dev/null)" ]; then
    echo "skip [$((i+1))/${#RUNS[@]}] already tried: $name"
    i=$((i+1)); continue
  fi

  # throttle: wait for a slot if we're at the concurrency cap
  while [ "$(jobs -r | wc -l)" -ge "$MAX_CONCURRENT" ]; do
    wait -n
  done
  dev="${GPUS[$(( i % ${#GPUS[@]} ))]}"   # round-robin device
  launch_one "$tuple" "$dev"
  echo "launched [$((i+1))/${#RUNS[@]}] on GPU $dev: $name"
  i=$((i+1))
  sleep 2   # small stagger so numba JIT / data-gen don't all hit at once
done

wait   # block until every run finishes
echo "All ${#RUNS[@]} runs complete. Logs + manifest in $LOGDIR"