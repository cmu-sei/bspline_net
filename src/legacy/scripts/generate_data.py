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
import json
import time
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from bsplinenet.ff_net import gen_data_np, ODESYS
from quadrotor.ellipsoid import gen_ic_ellips
 
def gen_data(
    n_control:int, 
    bspline_order:int, 
    state_space_dim:int, 
    time_interval:float,
    sampling_interval:float, 
    n_ic:int,
    ode_system:ODESYS,
    randomize_knots: bool = False,
    n_knot_arrays : int = 1, 
    use_integrated: bool = False, 
):
    tt = np.arange(
        0, time_interval + sampling_interval, sampling_interval
    )  # vector of time
    Ns = tt.size  # total number of samples

    if ode_system==ODESYS.QUADROTOR:
        
        max_x = 2 # m
        max_dx = 2 # m/s
        max_ag = np.pi/4 # radians
        max_da = 5 # rad/sec
        R = np.array([max_dx,max_x,
                      max_dx,max_x,
                      max_dx,max_x,
                      max_da,max_ag,
                      max_da,max_ag,
                      max_da,max_ag])
        if state_space_dim>12 or state_space_dim<12:
            raise ValueError("The number of state space dimension for quadrotor is 12.")
        A = np.eye(state_space_dim,state_space_dim)*R
        x0 = gen_ic_ellips(state_space_dim, A, n_ic).T
        x1 = np.zeros(12)
        
    else:
        # Generation of the initial conditions
        a = np.array([1, 1])
        b = -a
        x0 = a + np.random.rand(n_ic, state_space_dim) * (b - a)
        x1 = x0 # is ignored for duffing system
    Y_T, param_errors, rmse = gen_data_np(ode_system, n_control, bspline_order,tt, x0, x1, randomize_knots, n_knot_arrays, use_integrated)
    Y = Y_T.T

    # save as numpy - ignore torch conversion
    # X = torch.from_numpy(x0).float()
    # Y = torch.from_numpy(Y).float()
    return x0, Y, param_errors, rmse


def as_list(val):

    if type(val) == list:
        return val
    elif type(val) == int or type(val) == float:
        return [val]
    elif type(val) == str:
        v = json.loads(val)
        if type(v) == list:
            return v
        else:
            return [v]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="config file")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    folder = config.get("general", "folder").strip("'")
    prefix = config.get("general", "file_name_prefix").strip("'")
    ode_system = config.get("general", "system").strip("'")
    
    Path(f"data/{folder}").mkdir(parents=True, exist_ok=True)
    if ode_system.upper() not in ODESYS._member_names_:
        raise Exception(ValueError, "This isn't a valid ODE system")
    ode_system = ODESYS[ode_system.upper()]
    

    sec = "parameters"
    bspline_ord_list = as_list(config.get( sec, "bspline_order"    ))
    st_spc_dim_list  = as_list(config.get( sec, "state_space_dim"  ))
    samp_intrvl_list = as_list(config.get( sec, "sampling_interval"))
    n_control_list   = as_list(config.get( sec, "n_control"        ))
    time_intrvl_list = as_list(config.get( sec, "time_interval"    ))
    n_ic_list        = as_list(config.get( sec, "n_ic"             ))
    n_knot_arrays = config.getint(sec, "n_randomized_knots", fallback=0)
    use_integrated = config.getboolean(sec, "use_integrated", fallback = False)
    randomize_knots =  n_knot_arrays>0
    if not randomize_knots:
        n_knot_arrays = 1

    ignore = config.get(sec, "sample_ignore").strip("'")

    total_nic = np.sum(n_ic_list)

    for bs in bspline_ord_list:
        for sd in st_spc_dim_list:
            for si in samp_intrvl_list:
                for ti in time_intrvl_list:
                    for nc in n_control_list:
                        if eval(ignore):
                            continue
                        attr = f"""
    bspline_order = {bs}
    state_space_dim = {sd}
    sampling_interval = {si}
    time_interval = {ti}
    control_points = {nc}
    total_size = {total_nic} broken into {n_ic_list}
    use_integrated = {int(use_integrated)}
    """
                        print(attr)
                        # generate data
                        X, Y,param_errors, rmse= gen_data(nc, bs, sd, ti, si, total_nic, ode_system, randomize_knots, n_knot_arrays, use_integrated)
                        file_name = f"data/{folder}/{prefix}_{ode_system}_{bs}_dim={sd}_samp={si}_time={ti}_ctrl={nc}_int={int(use_integrated)}"
                        for i0, i1 in list(
                            zip(np.cumsum([0] + n_ic_list), np.cumsum(n_ic_list))
                        ):
                            x_ = X[i0:i1,:]
                            y_ = Y[i0:i1,:]

                            ncout = Dataset(f"{file_name}_ic={i1-i0}.nc", "w", "NETCDF4")
                            ncout.setncatts(
                                {
                                    "system":str(ode_system),
                                    "n_control": nc,
                                    "bspline_order": bs,
                                    "n_state_dimension": sd,
                                    "time_interval": ti,
                                    "sampling_interval": si,
                                    "n_initial_conditions": i1 - i0,
                                    "n_knots": nc-bs+1,
                                    "use_integrated": 1 if use_integrated else 0
                                }
                            )
                            ncout.setncattr_string(
                                "description", f"{prefix} data made on {time.ctime()}"
                            )
                            ncout.createDimension("n_control_points_per_state", nc * sd)
                            ncout.createDimension("state_dimensions", sd)
                            ncout.createDimension("n_initial_conditions", i1 - i0)
                            ncout.createDimension("n_knots", nc-bs+1)
                            ncout.createDimension("n_knot_vectors", n_knot_arrays)
                            Xvar = ncout.createVariable(
                                "X",
                                "float32",
                                ("n_knot_vectors", "n_initial_conditions", "state_dimensions"),
                            )
                            Yvar = ncout.createVariable(
                                "Y",
                                "float32",
                                ("n_knot_vectors", "n_initial_conditions", "n_control_points_per_state"),
                            )
                            
                            Kvar = ncout.createVariable(
                                "Knots",
                                "float32",
                                ("n_knot_vectors", "n_initial_conditions", "n_knots"),
                            )
                            Xvar[:] = x_
                            Yvar[:] = y_[:,:,:nc*sd]
                            Kvar[:] = y_[:,:,nc*sd+1:-1]
                            ncout.close()
