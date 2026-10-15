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

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import odeint
from train import get_data_params, load_data

from bsplinenet.Bspline_functions import Bs_comp, duffing_sys
from bsplinenet.ff_net import ODESYS
from bsplinenet.net import Net

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True, help="Path to model file")
parser.add_argument("-d", "--data", required=True, help="Path to data file")
parser.add_argument("-c", "--model-config", required=True, help="Path to model config file")
parser.add_argument("-i", "--index", required=False, help="index to look at", type=int, default=-1
)
args = parser.parse_args()

# Read the configuration file
config = configparser.ConfigParser()
config.read(args.model_config)

# Get the parameters
n_hidden = config.getint("nnet", "n_hidden")
n_layers = config.getint("nnet", "n_layers")
activation = config.get("nnet", "activation").strip("'") 


if activation == "relu":
    activation_function = nn.ReLU()
else:
    raise ValueError(f"Unknown activation function: {activation}")



# Load the data
X, Y_true, ode_sys  = load_data(args.data)
ode_system = ODESYS[ode_sys.upper()].get_ode()

params = get_data_params(args.data, ['bspline_order',
                                     'time_interval',
                                     'sampling_interval',
                                     'n_control',
                                     'n_state_dimension'])
print(params)
# Load the model
n_state_space = X.shape[1]
n_control_points = int(Y_true.shape[1] / X.shape[1])
model_saved = torch.load(args.model)

if 'branch' in set([i for key in model_saved.keys() for i in key.split('.')]):
    network_type = 'deeponet'
else:
    #!note no way to identify recurrent right now.
    network_type = 'feedforward'
    
model = Net(n_state_space, n_control_points, n_layers, n_hidden, activation_function, False, network_type)
model.load_state_dict(model_saved)
model.eval()

# pick random IC to look at
if args.index < 0:
    idx = np.random.randint(0, X.shape[0])
else:
    idx = args.index

print(f"Looking at index {idx}")
time_range = np.arange(0,params.get('time_interval')+
                       params.get('sampling_interval'),
                       params.get('sampling_interval'))
Ns = time_range.shape[0]
### KLUDGE!!
x1 = np.array([0,5,
                0,0,
                0,5,
                0,0,
                0,0,
                0,0])
ode_ = ode_system(X[idx, :], time_range,(x1,) )

# Get the model output
print(f'Get prediction {X[idx:idx+1,:].shape}')
with torch.no_grad():
    Y_pred = model(X[idx:idx+1,:])
#print(Y_pred.shape, Y_true.shape, X.shape, idx)
# Get the B-spline
print(f'Get the b-spline')
spline_true = Bs_comp(Ns, Y_true.T[:, 0], params.get('n_control'), params.get('n_state_dimension'), Bit)
spline_pred = Bs_comp(Ns, Y_pred.T[:, 0], params.get('n_control'), params.get('n_state_dimension'), Bit)

# Plot the output
print('Plot the output')
plt.figure(figsize=(10, 5))

h = params.get('time_interval')/params.get('n_control')
time_range_c = np.arange(0, params.get('time_interval'), h)

fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axs[0].plot(time_range, spline_true[0, :], label="LS Bspline")
axs[0].plot(time_range, ode_[:, 0], label="True $x_1$")
axs[0].plot(time_range_c, Y_true[0, :params.get('n_control')], "ro", label="LS cp")
axs[0].set_ylabel("$x_1$")

axs[1].plot(time_range, spline_true[1, :], label="LS Bspline")
axs[1].plot(time_range, ode_[:, 1], label="True $x_2$")
axs[1].plot(time_range_c, Y_true[0, params.get('n_control'):params.get('n_control')*2], "ro", label="LS cp")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("$x_2$")

axs[0].legend(loc="upper right")
axs[1].legend(loc="upper right")

plt.suptitle("Least-Squares Bsplines")
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axs[0].plot(time_range, spline_pred[0, :], label="NN Bspline")
axs[0].plot(time_range, ode_[:, 0], label="True $x_2$")
axs[0].plot(time_range_c, Y_pred[0, :params.get('n_control')], "ro", label="NN cp")
axs[0].set_ylabel("$x_1$")

axs[1].plot(time_range, spline_pred[1, :], label="NN Bspline")
axs[1].plot(time_range, ode_[:, 1], label="True $x_2$")
axs[1].plot(time_range_c, Y_pred[0, params.get('n_control'):params.get('n_control')*2], "ro", label="NN cp")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("$x_2$")

axs[0].legend(loc="upper right")
axs[1].legend(loc="upper right")

plt.suptitle("Predicted BSplines")
plt.show()
plt.savefig("test.png")
print('Done!')