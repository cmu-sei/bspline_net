#!/usr/bin/env bash
# ============================================================================
# link_ckpt_names.sh
# ----------------------------------------------------------------------------
# One-time (repeatable) step: for each OLD-scheme checkpoint dir, create a
# NEW-scheme symlink pointing at it, so the "already done" resume check in
# run_sweep.sh (which looks up by new-scheme name) finds it -- WITHOUT moving
# or renaming anything. Original directories are never touched.
#
#   checkpoints/sweep_logs/traj/d128_r251_s42_b2500_q95_p0/          (real dir, untouched)
#   checkpoints/sweep_logs/traj/d128_r251_s42_b2500_q95_p0_w128x64x128 -> d128_r251_s42_b2500_q95_p0/   (symlink)
#
# SAFE / IDEMPOTENT:
#   * Only links dirs matching the OLD 6-field pattern
#       d{lifted}_r{rollout}_s{seed}_b{batch}_q{pct}_p{pct}
#   * If the new-scheme name already exists (as a real dir OR a symlink),
#     it's left alone -- re-running is a no-op for anything already linked.
#   * Dry-run by default. Pass --apply to actually create the links.
#   * Verified: run_sweep.sh's resume check (`if [ -d "$ckpt_dir" ]` +
#     `ls "$ckpt_dir" | grep -oE 'epoch=[0-9]+'`) follows symlinks correctly,
#     so runs behind a link are recognized as "already done" exactly like a
#     real directory would be.
#   * To undo: the created links are plain symlinks, so
#       find checkpoints/sweep_logs/traj -maxdepth 1 -type l -delete
#     removes them without touching any real data.
#
# Usage:
#   ./link_ckpt_names.sh            # dry run, prints what it WOULD do
#   ./link_ckpt_names.sh --apply    # actually creates the symlinks
# ============================================================================
set -uo pipefail

CKPT_ROOT="checkpoints/sweep_logs/traj"
ANCHOR_WIDTHS_IC_TAG="w128x64x128"   # matches ANCHOR_WIDTHS_IC="128,64,128" in run_sweep.sh

APPLY=0
if [[ "${1:-}" == "--apply" ]]; then
  APPLY=1
fi

# old-scheme pattern: d<...>_r<...>_s<...>_b<...>_q<...>_p<...>   (exactly 6 fields)
OLD_PATTERN='^d[0-9]+_r[0-9]+_s[0-9]+_b[0-9]+_q[0-9]+_p[0-9]+$'

if [[ ! -d "$CKPT_ROOT" ]]; then
  echo "No checkpoint root found at $CKPT_ROOT -- nothing to link."
  exit 0
fi

n_linked=0
n_skipped=0

for dir in "$CKPT_ROOT"/*/; do
  [[ -d "$dir" ]] || continue
  # skip symlinked dirs themselves (don't chain-link a link)
  [[ -L "${dir%/}" ]] && continue

  base="$(basename "$dir")"

  if [[ "$base" =~ $OLD_PATTERN ]]; then
    new_name="${base}_${ANCHOR_WIDTHS_IC_TAG}"
    new_path="${CKPT_ROOT}/${new_name}"

    if [[ -e "$new_path" || -L "$new_path" ]]; then
      echo "SKIP (target already exists): $new_name"
      n_skipped=$((n_skipped+1))
      continue
    fi

    if [[ "$APPLY" -eq 1 ]]; then
      ln -s "$base" "$new_path"   # relative symlink, same directory
      echo "linked: $new_name -> $base"
    else
      echo "[dry-run] would link: $new_name -> $base"
    fi
    n_linked=$((n_linked+1))
  else
    n_skipped=$((n_skipped+1))
  fi
done

echo "----"
echo "linked: $n_linked   skipped (already linked / already new / unrecognized): $n_skipped"
if [[ "$APPLY" -eq 0 ]]; then
  echo "This was a DRY RUN. Re-run with --apply to actually create symlinks."
fi
