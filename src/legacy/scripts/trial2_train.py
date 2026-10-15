"""
bspline_net (Github)

Copyright 2026 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to its own license.

DM26-0750
"""
#!/usr/bin/python3
import subprocess
from pathlib import Path 
network_types = {'ff':'feedforward','do':'deeponet'}
time_interval = [2.5,5,10]
n_control_points = [5,10,20,40,80]
datasets = [(2400,600),(1600,400), (800,200)]
folder = "trial2_240205"
prefix = f'{folder}/data__3_2_1-100'
Path(f"models/{folder}").mkdir(parents=True, exist_ok=True)

for ti in time_interval:
    for nc in n_control_points:
        for nt,nv in datasets:
            for nn,net in network_types.items():
                if nc<2*ti or nc>10*ti:
                    continue
                
                train_data_file = Path(f"data/{prefix}_{(int(ti*10))}-10_{nc}_{nt}.nc")
                if not train_data_file.is_file():
                    raise ValueError(f"file {train_data_file} does note exist")
                
                valid_data_file = Path(f"data/{prefix}_{(int(ti*10))}-10_{nc}_{nv}.nc")
                if not valid_data_file.is_file():
                    raise ValueError(f"file {valid_data_file} does not exist")
                
                model = f"{prefix}_{(int(ti*10))}-10_{nc}_{nt}_{nv}_{nn}"
                mpath = Path(f'models/{model}.pt')
                if mpath.is_file():
                    print(f"FOUND FILE {model}.pt")
                    continue
                command_ =["python3",
                                 "scripts/train_lightning.py",
                                 "-t", str(train_data_file),
                                 "-v", str(valid_data_file),
                                 "-c", "configs/trial2/config_train.ini",
                                 "-m", model,
                                 "-n", net
                                 ]
                print("\n\n============\n"+" ".join(command_))
                subprocess.run(command_, timeout=1200)
                
