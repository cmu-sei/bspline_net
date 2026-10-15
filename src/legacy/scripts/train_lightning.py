"""
bspline_net (Github)

Copyright 2026 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to its own license.

DM26-0750
"""
import argparse
import configparser
import sys
from pathlib import Path
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader, TensorDataset

from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.accelerators import find_usable_cuda_devices
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from bsplinenet.net import LightningNet 
from bsplinenet.Bspline_functions import BsKnots, rotate, to_cylindrical
from train import load_data, get_data_params
from generate_data import as_list


# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a console handler with level INFO
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a file handler with level WARNING
file_handler = logging.FileHandler('log.log','w')
file_handler.setLevel(logging.INFO)

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


if __name__ == "__main__":
    # for A100 GPUs
    torch.set_float32_matmul_precision("medium")
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", required=True, help="Path to the training data file")
    parser.add_argument("-v", "--valid", required=True, help="Path to the validation data file")
    parser.add_argument("-c", "--config", required=True, help="Path to the config file")
    parser.add_argument("-r", "--repeat", 
                        required=False, 
                        default = 1,  
                        type=int,
                        help = "times to repeat training")
    
    parser.add_argument("-m", "--model", required=False, help="Where to save model (overrides config)")
    parser.add_argument("-n", "--network", required=False, help="network type (overrides config)")
    
    # Check if config file name is provided
    if len(sys.argv) < 2:
        print("Usage: python scripts/train_lightning.py -t <train.nc> -v <valid.nc> -c <config.ini> -m <model.pt>")
        sys.exit()

    args = parser.parse_args()
    # Read the config file
    config = configparser.ConfigParser()
    config.read(args.config)

    # Get the parameters
    use_gpu     = config.getboolean("general", "use_gpu")
    save_model  = config.getboolean("general", "save_model")
    model_name  = config.get("general", "model_name").strip("'")
    lr          = config.getfloat("train", "learning_rate")
    epoch_train = config.getint("train", "epoch_train")
    batch_size  = config.getint("train", "batch_size")
    loss_func   = config.get("train", "loss").strip("'")
    loss_trans  = config.getboolean("train", "loss_transform")
    optimizer   = config.get("train", "optimizer").strip("'")
    patience    = config.getint("train", "patience")
    if config.has_option("train","rotate"): 
        angles_deg  = as_list(config.get("train","rotate"))
    else:
        angles_deg = []
    
    if config.has_option("train","coords"): 
        coords  = config.get("train","coords").strip("'")
    else:
        coords = "cartesian"
        
    sch_step    = config.getint("train", "scheduler_step_size")
    sch_gamma   = config.getfloat("train", "scheduler_gamma")
    print_every = config.getfloat("train", "print_out_every")

    n_hidden    = config.getint("nnet", "n_hidden")
    n_layers    = config.getint("nnet", "n_layers")
    activation  = config.get("nnet", "activation").strip("'")  
    network_type = config.get("nnet", "network_type").strip("'") 

    #override if given
    
    if args.model is not None:
        model_name = args.model
        
    if args.network is not None:
        network_type = args.network

    # get data
    X_train, Y_train = load_data(args.train)
    X_valid, Y_valid = load_data(args.valid)
   
    n_state_dim = X_train.shape[1]
    n_control_points = int(Y_train.shape[1] / X_train.shape[1])
    logger.debug(f'\n\n network type:\t{network_type}')
    logger.debug(f' state space dimensions:\t{n_state_dim}')
    logger.debug(f' control point number:\t{n_control_points}\n\n') 
    # data augmentation
    for i in angles_deg:
        logger.debug(f'augmenting by rotating around Z by {i} degrees')
        angle = np.pi*i/180
        rot = rotate(0, 0, angle, every_other = True)
        X_train = torch.vstack([X_train, rot(X_train, n_cp=1)])
        Y_train = torch.vstack([Y_train, rot(Y_train, n_cp =n_control_points )])
        X_valid = torch.vstack([X_valid, rot(X_valid, n_cp=1)])
        Y_valid = torch.vstack([Y_valid, rot(Y_valid, n_cp =n_control_points )])
    
    if coords == "cylindrical":
        logger.debug(f'NEW! change to cylindrical coordinates')
        X_train = to_cylindrical(X_train,1)
        Y_train = to_cylindrical(Y_train,n_control_points)
        X_valid = to_cylindrical(X_valid,1)
        Y_valid = to_cylindrical(Y_valid,n_control_points)
    elif coords == "cartesian":
        logger.debug("Cartesian coordinates, do nothing.")
    else:
        logger.error(f"Unknown coordinate system {coords}")
        exit()
    
    print(X_train.shape, Y_train.shape)
    # Define the activation function
    if activation == "relu":
        activation_function = nn.ReLU()
    else:
        raise ValueError(f"Unknown activation function: {activation}")

    # Create the network
    
    transform_matrix = None
    if loss_trans:
        
        params = get_data_params(args.train,['bspline_order','time_interval'])
        logger.debug(f'calculating transformation of control points to to 1000 points')
        _,_,transform_matrix = BsKnots(n_control_points,params["bspline_order"],1000)
        transform_matrix = torch.tensor(transform_matrix, dtype=torch.float32)
        
    net = LightningNet(n_state_dim, 
                       n_control_points, 
                       n_layers, 
                       n_hidden, 
                       activation_function,
                       nn_type = network_type,
                       transform_matrix =  transform_matrix,
                       time_step = float(params['time_interval'])/n_control_points)
    # Define the loss function
    if loss_func == "mse":
        #criterion = nn.MSELoss(reduction = "mean")
        net.set_loss(nn.MSELoss(reduction='mean'))
    else:
        raise ValueError(f"Unknown loss function: {loss_func}")

    
    # If GPU is available, move the network to GPU
    #if use_gpu and torch.cuda.is_available():
    #    net = net.cuda()
    #    X_train = X_train.to("cuda:0")
    #    Y_train = Y_train.to("cuda:0")
    #    X_valid = X_valid.to("cuda:0")
    #    Y_valid = Y_valid.to("cuda:0")

    # Create Dataloaders from the training/validation data
    train_data = TensorDataset(X_train, Y_train)
    valid_data = TensorDataset(X_valid, Y_valid)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=12)#, num_workers=2, persistent_workers=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=12)#, num_workers=2, persistent_workers=True)
    
    
    for i in range(args.repeat):
            
        # Define the optimizer
        if optimizer == "adam":
            net.set_optimizer(optim.Adam, learning_rate = lr)
            #optimizer = optim.Adam(net.parameters(), learning_rate =lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        # Define the learning rate scheduler
        net.set_scheduler(torch.optim.lr_scheduler.StepLR,
                        sch_step, 
                        sch_gamma)
        # Define early stopping
        early_stop_callback = EarlyStopping(monitor="val_loss", 
                                            min_delta=0.0001, 
                                            patience=patience, 
                                            verbose=False, 
                                            mode="min")
        
        # Define logging
        loggertb = TensorBoardLogger(f"logs/{model_name.split('/')[0]}",
                                name = model_name.split('/')[-1])
        
        loggercsv = CSVLogger(f"logs/{model_name.split('/')[0]}",
                                name = model_name.split('/')[-1])
    
        # Define checkpointing
        fname = f"{model_name.split('/')[-1]}"+\
            f"_{network_type}"+\
            f"_{coords}"+\
            f"_ncp={n_control_points}_t={params['time_interval']}_{i}"+\
            "_{val_loss:2.6f}"
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k = 1,
            save_weights_only=False,
            save_last=False,
            every_n_epochs=3,
            dirpath=f"models/{model_name.split('/')[0]}",
            filename=fname
        )
        # Define early stopping
        early_stop_callback = EarlyStopping(monitor="val_loss", 
                                            min_delta=0.000001, 
                                            patience=patience, 
                                            verbose=False, 
                                            mode="min")
        
        
        # Define a trainer
        trainer = pl.Trainer(max_epochs = epoch_train,
                            callbacks=[early_stop_callback, checkpoint_callback],
                            log_every_n_steps = int(min(X_train.shape[0]/batch_size, print_every))) 
        
        if use_gpu and torch.cuda.device_count()>0:
            trainer = pl.Trainer(accelerator="cuda", 
                                strategy="ddp",
                                logger = [loggertb, loggercsv],
                                devices=find_usable_cuda_devices(3),
                                max_epochs = epoch_train,
                                callbacks=[early_stop_callback, checkpoint_callback],
                                log_every_n_steps = int(min(X_train.shape[0]/batch_size, print_every)),
                                max_time={"minutes":120}) 
 

        logger.debug(f'\n---\nRunning {i+1}')
        net.initialize_weights(i)
        start_time = time.time()
        trainer.fit(net, 
                    train_dataloaders=train_loader, 
                    val_dataloaders=valid_loader)
        #if trainer2 is not None:
        #        
        #    trainer2.fit(net, 
        #                train_dataloaders=train_loader, 
        #                val_dataloaders=valid_loader)
        end_time = time.time()
        
        # checkpoints are better maybe
        # Save model
        #if save_model:
        #    if args.repeat ==1:
        #        logger.debug(f"Saving model")
        #        torch.save(net.state_dict(), f"{Path('models') / Path(model_name)}.pt")
        #    else:
        #        logger.debug(f"Saving model {i+1} out of {args.repeat}")
        #        torch.save(net.state_dict(), f"{Path('models') / Path(model_name)}_{i}.pt")    
