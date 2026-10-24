# bspline_net

Code for:
- **IEEE 2026**: "A Neural Network Approach towards Real-Time Knot Point Density Estimates for the
  Nonlinear Quadrotor" (knot placement)
- **TMLR**: "Spectral and Conformal Control Point Locations with Splines"

Earlier work (CDC 2024) is archived in `src/legacy/`.

## Setup
### Docker (GPU)
```
docker build -t <your image name/> .
docker run -it --rm --network=host --shm-size 8G -e MY_UID=$(id -u) -e MY_GID=$(id -g) --gpus all --name test \
  -v `pwd`/src:/home/hspn-user/src -v `pwd`/tests:/home/hspn-user/tests -v `pwd`/data:/home/hspn-user/data \
  -v `pwd`/models:/home/hspn-user/models -v `pwd`/checkpoints:/home/hspn-user/checkpoints \
  -v `pwd`/outputs:/home/hspn-user/outputs -v `pwd`/analysis_results:/home/hspn-user/analysis_results \
  <your image name/>
```
Or: copy `docker-compose.generic.yml` to `docker-compose.yml` and `.env.example` to `.env`, fill in your ids,
then `docker compose run --rm hspn`.

### Local (CPU)
```
uv venv --python 3.10 && source .venv/bin/activate && uv pip install -e ".[dev]" tensorboard
```
Some configs use container paths (`/home/hspn-user/...`); override them on the command line when running locally.

Training and evaluation data are generated on the fly (quadrotor simulations in `src/bsplinenet/data.py`);
there is no dataset to download.

## IEEE 2026: knot placement
```
train stage=knots
mkdir -p analyses/knots && knot_analysis ckpt_path=analysis_results/knots/22.4k_small_model.ckpt
```

## TMLR
```
train stage=traj                                              # trajectory model (sweep: run_sweep.sh)
python -m bsplinenet.analysis.traj_conformal_analysis          # conformal calibration
python -m bsplinenet.analysis.traj_noise_propagation --config src/configs/conformal_config.yaml
python -m bsplinenet.analysis.final_spline_pred                # control-point intervals (uniform=true: evenly spaced knots)
run_end_to_end hydra.job.chdir=false                           # control-point margin check
```

## Override parameters
```
train stage=knots model.knots.lr=0.001 trainer.max_epochs=50
```
note! the cp_model and the joint_model (now in `src/legacy/`) are code culs-de-sac. Ideas that did not contribute to a paper. Do not use! The correct model is traj_model with the traj training stage.

## ONNX export (for on-device timing)
```
onnxexport
```

## Original instructions (old pipeline)

The original commands for the earlier `.ini`-based pipeline (CDC 2024), kept as written for reference.
That code is now in `src/legacy/` (the paths below are from before the move) and does not run on this
branch; see `src/legacy/README.md`.

## Base examples (old instructions)
Example run of ff_net
```
python ff_net.py 
```

Example run of train

```
python train.py config.ini
```

Example of running plot (must have saved model and data matched to config file)
```
python -i  plot.py -m models/example -d data/example_data.pt -c config.ini -i 3
```
### generate data
```
python3 scripts/generate_data.py -c configs/config.ini
```
### running python training script in docker container
```
python3 scripts/train.py configs/config.ini
```
similarly for using the lightning script:
```
python3 scripts/train_lightning.py -c <config.ini> -t <train.nc> -v <valid.nc>  -r <n_repeat> -m <model_name> -n <net_type>
```
example of plotting
```
python3 -i scripts/plot.py  -m models/trial1_240130/data__3_2_1-100_100-10_40_800_200_do.pt  -d data/trial1_240130/data__3_2_1-100_100-10_40_100.nc  -c configs/trial1/config_train.ini  
```
example of stats
```
python3 scripts/stats.py  -m models/trial1_240130/data__3_2_1-100_100-10_40_1600_400_do.pt  -d data/trial1_240130/data__3_2_1-100_100-10_40_100.nc  -c configs/trial1/config_train.ini 
```
