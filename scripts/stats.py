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
import numpy as np
#import pandas as pd
import scipy 
import torch 
import torch.nn as nn
from train import load_data
from pathlib import Path

from bsplinenet.Bspline_functions import Bs_comp, duffing_sys
from bsplinenet.net import Net
from train import load_data


#if __name__=='__main__':
parser = argparse.ArgumentParser()

parser.add_argument("-m", "--model", required=True, help="Path to the model")
parser.add_argument("-d", "--data", required=True, help="Path to the test data")
parser.add_argument("-c", "--config", required=True, help="Path to config file")

args = parser.parse_args()
data = Path(args.data)
model = Path(args.model)
if not data.is_file():
    raise ValueError(f"file {args.data} does not exist")

if not model.is_file():
    raise ValueError(f"file {args.model} does not exist") 

config = configparser.ConfigParser()
config.read(args.config)

n_hidden = config.getint("nnet", "n_hidden")
n_layers = config.getint("nnet", "n_layers")
activation = config.get("nnet", "activation").strip("'")

if activation == "relu":
    activation_function = nn.ReLU()
else:
    raise ValueError(f"Unknown activation function: {activation}")

X, Y_true = load_data(data)


# Load the model
n_state_space = X.shape[1]
n_control_points = int(Y_true.shape[1] / X.shape[1])
model_saved = torch.load(args.model)
if 'branch' in set([i for key in model_saved.keys() for i in key.split('.')]):
    network_type = 'deeponet'
else:
    #!note no way to identify recurrent right now.
    network_type = 'feedforward'
    
print(f'network type {network_type}')  
model = Net(n_state_space, n_control_points, n_layers, n_hidden, activation_function, network_type)
model.load_state_dict(model_saved)
model.eval()
with torch.no_grad():
    Y_pred = model(X)
    
x = X.detach().cpu().numpy()
y_true = Y_true.detach().cpu().numpy() 
y_pred = Y_pred.detach().cpu().numpy() 

dif = y_pred - y_true

print(mean_dif = np.mean(dif,axis=1))
print(mean_dif2 = np.mean(dif**2, axis=1))
print((dif**2).mean())
