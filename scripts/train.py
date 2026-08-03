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
from numpy.ma import getdata
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader, TensorDataset

from bsplinenet.net import Net
from bsplinenet.Bspline_functions import BsKnots
from netCDF4 import Dataset


# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a console handler with level INFO
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a file handler with level WARNING
file_handler = logging.FileHandler("log.log", "w")
file_handler.setLevel(logging.INFO)

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# takes in a module and applies the specified weight initialization
def init_weights(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        m.weight.data.uniform_(-1.0/np.sqrt(m.in_features), 1.0/np.sqrt(m.in_features))
        m.bias.data.fill_(0)

def load_data(nc_file):
    if not Path(nc_file).is_file():
        raise ValueError(f"Cannot find file {nc_file}")
    data = Dataset(nc_file, "r")
    attributes = data.ncattrs() 
    if 'system' in attributes:
        logger.debug(f" system: { data.getncattr('system')}")
    else :
        logger.debug(f" system: { 'unknown (duffing)'}")
        
    logger.debug(f" bspline order: { data.getncattr('bspline_order')}")
    logger.debug(f" sampling interval: { data.getncattr('sampling_interval')}")
    logger.debug(f" time interval: { data.getncattr('time_interval')}")
    logger.debug(f" number of control points { data.getncattr('n_control')}")
    logger.debug(f" dimensions of state { data.getncattr('n_state_dimension')}")
    logger.debug(f" number of initial conditions { data.getncattr('n_initial_conditions')}")
    x = torch.from_numpy(getdata(data.variables["X"][:]))
    y = torch.from_numpy(getdata(data.variables["Y"][:]))
    
    return x, y

def get_data_params(nc_file, pars, defaults = {'system':'duffing'}):

    if not Path(nc_file).is_file():
        raise ValueError(f"Cannot find file {nc_file}")
    data = Dataset(nc_file, "r")
    ret = dict()
    for i in pars:
        val = defaults.get(i)
        
        if i in data.ncattrs():
            val = data.getncattr(i)
        else:
            logger.debug(f'parameter {i} not specified, using default value {val}')    
        
        ret[i] = val
    
    return ret
    
    
 
if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", required=True, help="Path to the training data file")
    parser.add_argument("-v", "--valid", required=True, help="Path to the validation data file")
    parser.add_argument("-c", "--config", required=True, help="Path to the config file")
    parser.add_argument("-m", "--model", required=False, help="Where to save model (overrides config)")
    parser.add_argument("-n", "--network", required=False, help="network type (overrides config)")
    
    # Check if config file name is provided
    if len(sys.argv) < 2:
        print("Usage: python scripts/train.py -t <train.nc> -v <valid.nc> -c <config.ini> -m <model.pt>")
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

    
    X_train, Y_train = load_data(args.train)
    X_valid, Y_valid = load_data(args.valid)
    
    # Define the activation function
    if activation == "relu":
        activation_function = nn.ReLU()
    else:
        raise ValueError(f"Unknown activation function: {activation}")

    # Create the network
    n_state_dim = X_train.shape[1]
    n_control_points = int(Y_train.shape[1] / X_train.shape[1])
    
    
    transform_matrix = None
    if loss_trans:
        params = get_data_params(args.train,['bspline_order'])
        logger.debug(f'calculating translation to 1000 points')
        _,_,transform_matrix = BsKnots(n_control_points,params["bspline_order"],1000)
        transform_matrix = torch.tensor(transform_matrix, dtype=torch.float32).to('cuda:0')
        
    net = Net(n_state_dim, n_control_points, n_layers, n_hidden, activation_function, network_type, transform_matrix)
    net.apply(init_weights)

    # Define the loss function
    if loss_func == "mse":
        criterion = nn.MSELoss(reduction="mean")
    else:
        raise ValueError(f"Unknown loss function: {loss_func}")

    # Define the optimizer
    if optimizer == "adam":
        optimizer = optim.Adam(net.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    # Define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=sch_step, gamma=sch_gamma
    )

    # If GPU is available, move the network to GPU
    if use_gpu and torch.cuda.is_available():
        net = net.cuda()
        X_train = X_train.to("cuda:0")
        Y_train = Y_train.to("cuda:0")
        X_valid = X_valid.to("cuda:0")
        Y_valid = Y_valid.to("cuda:0")

    # Create a TensorDataset from the training dat
    train_data = TensorDataset(X_train, Y_train)
    # Create a DataLoader from the TensorDatase
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    best_loss = float("inf")  # Best loss achieved so far
    epochs_no_improve = 0  # Number of epochs with no improvement

    # Training loop
    start_time = time.time()
        
    for epoch in range(epoch_train):
        for inputs, targets in train_loader:
            # Forward pass
            outputs = net(inputs)
            outputs = net.transform(outputs)
            targets = net.transform(targets)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validate every epoch
        with torch.no_grad():
            outputs_test = net(X_valid)
            outputs_test = net.transform(outputs_test)
            targets_test = net.transform(Y_valid)
            loss_train = loss.detach().cpu()
            loss_test = criterion(outputs_test, targets_test).detach().cpu()

        # print every N epochs
        if epoch % print_every == 0:
            logger.debug(
                f"Epoch {epoch}, Loss: {loss_train.item():2.5f} Validation Loss: {loss_test.item():2.5f}"
            )

        # Check for improvement
        if loss_test < best_loss:
            best_loss = loss_test
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.debug("Early stopping!")
                break
    end_time = time.time()

    if save_model:
        logger.debug("Saving model")
        torch.save(net.state_dict(), f"{Path('models') / Path(model_name)}.pt")

    ## Summarize run
    run_summary = f"""
    Summary of training run
    Epochs: {epoch}
    Time: {np.round(end_time-start_time)/60:2.2f} minutes, or {(end_time-start_time)/epoch:2.2f} seconds per epoch
    Training Data Size: {X_train.shape[0]}
    Input Size: {X_train.shape[1]}
    Output Size: {outputs.shape[1]}
    
    Parameters: """
    param_print = ""
    for section in config.sections():
        param_print += f"\n\n    [{section}]\n"

        for key in config[section]:
            param_print += f"\n        {key} = {config[section][key]}"

    logger.info(f"{run_summary}\n {param_print}")
