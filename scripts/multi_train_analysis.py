"""
bspline_net (Github)

Copyright 2026 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to its own license.

DM26-0750
"""

from glob import glob
from pathlib import Path
from argparse import ArgumentParser
import torch
from typing import List
import numpy as np
from bsplinenet.net import LightningNet
from train import load_data
def get_net_params(mdl):

    state_dim =-1
    n_control_points=-1
    n_layers=-1
    n_hidden=-1
    nn_type = 'feedforward'
    
    keys = mdl.keys()
    nets = set([i.split('.')[0] for i in keys])
    ends = [(0,max([int(i.split('.')[1]) for net in nets for i in keys if i.split('.')[0] ==net]))]
    
    def n(mdl,w,i):
        return (mdl.get(f'{w}.{i[0]}.weight').shape,
                mdl.get(f'{w}.{i[1]}.weight').shape)
    
    z = {w:n(mdl,w,i) for w, i in zip(nets,ends)}
    if len(z.keys())>1:
        nn_type = 'deeponet'
        state_dim = z.get('mlayers')[0][1]+ z.get('branch')[0][1]
        n_control_points = z.get('leaf')[1][0]/state_dim
        
    elif z.get('mlayers')[0][1] == z.get('mlayers')[1][1]:
        nn_type = 'recurrent'
        state_dim = z.get('mlayers')[0][1]
    else:
        nn_type = 'feedforward'
        state_dim = z.get('mlayers')[0][1]
        n_control_points = z.get('mlayers')[1][0]/state_dim
    n_layers = sum([i[1] for i in ends])+1
    n_hidden = z.get('mlayers')[1][1]
        
    return  {
        "state_dim":state_dim,
        "n_control_points":n_control_points,
        "n_layers":n_layers,
        "n_hidden":n_hidden,
        "activation_function": torch.nn.ReLU(),
        "nn_type": nn_type
        }

class mod():
    def __init__(self, models:List[LightningNet]):
        self.models = models

    def predict(self, input, mod_i=None,val_i=None)-> np.array:
        vals = []
        for enum, m in enumerate(self.models):
            if mod_i and enum!=mod_i:
                continue
            vals.append([m(input).detach().cpu().numpy()])
        if val_i:
            return np.vstack(vals)[:,:,val_i]
        return np.vstack(vals)
        
        
if __name__=="__main__":
    
    ap = ArgumentParser()
    ap.add_argument('-d','--directory',required = True, help = 'The directory containing multiples of the same model')
    ap.add_argument('-f','--file',required = True, help='The data file to use')
    ap.add_argument('-g','--glob',required = False, default='*.pt', help = 'Filter to files with this structure in the folder')
    
    args = ap.parse_args()
    
    X, y = load_data(args.file)
    
    models = []
    for model_name in Path(args.directory).glob(args.glob):
        mdl = torch.load(model_name)
        net = LightningNet(**get_net_params(mdl))
        net.load_state_dict(state_dict = mdl)
        models.append(net.eval())