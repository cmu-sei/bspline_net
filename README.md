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
# Docker
## Building the docker image
```
docker build -t <your image name/> .
```
## Running docker container
```
docker run -it --rm --network=host --shm-size 8G -e UID=$(id -u) -e GID=$(id -g) --gpus all --name test -v `pwd`/data:/home/hspn-user/data -v `pwd`/scripts:/home/hspn-user/scripts -v `pwd`/src:/home/hspn-user/src -v `pwd`/models:/home/hspn-user/models -v `pwd`/configs:/home/hspn-user/configs <your image name/>
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
### running python plot script in docker container


```bash

 # Train knot model
python -m projectm.train stage=knots

# Train CP model
python -m projectm.train stage=cp

# Joint fine-tuning
python -m projectm.train stage=joint knot_checkpoint=checkpoints/knots/last.ckpt cp_checkpoint=checkpoints/cp/last.ckpt

# Override params
python -m projectm.train stage=knots model.knots.lr=0.001 trainer.max_epochs=50
```