"""
bspline_net (Github)

Copyright 2026 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to its own license.

DM26-0750
"""
from functools import partial

import torch
import torch.nn as nn
from lightning import pytorch as pl


class Net(pl.LightningModule):
    def __init__(
        self,
        state_dim,
        n_control_points,
        n_layers,
        n_hidden,
        activation_function,
        batch_norm=False,
        nn_type="fullyconnected",
        transform_matrix=None,
        time_step=None,
        weighting=None,
    ):
        super(Net, self).__init__()
        self.state_dim = state_dim
        self.n_control_points = n_control_points
        self.act = activation_function
        self.nn_type = nn_type
        self.time_step = time_step

        if weighting is None:
            self.w = torch.ones(
                self.state_dim * self.n_control_points,
                requires_grad=False,
                device=self.device,
            )
        else:
            self.w = weighting.to(self.device)
            self.w.requires_grad = False

        self.save_hyperparameters()

        if self.nn_type == "fullyconnected":
            n_input = self.state_dim
            n_output = self.state_dim * self.n_control_points
            n_layers = n_layers - 1

        elif self.nn_type == "recurrent":
            n_input = n_hidden
            n_output = self.state_dim  # get next state and control point
            n_layers = n_layers - 1

            # create a GRU "branch" with a hidden size of n_hidden
            self.rnn = torch.nn.GRUCell(self.state_dim, n_hidden)
            self.h0 = torch.zeros(1, self.rnn.hidden_size)

        elif self.nn_type == "deeponet":
            n_input = self.state_dim
            n_output = n_hidden
            if n_layers < 3:
                raise ValueError("For DeepONets, need at least 3 layers")
            n_layers = int((n_layers - 1) / 2)  # reduce b/c 2 networks

            # need a second network for deep_o_net
            self.branch = nn.ModuleList()
            nn_shape = [1] + [n_hidden] * n_layers
            for n_in, n_out in list(zip(nn_shape[:-1], nn_shape[1:])):
                self.branch.append(nn.Linear(n_in, n_out))

            # need a third network to combine two networks
            self.leaf = nn.ModuleList()
            nn_shape = [n_hidden, self.state_dim]
            self.leaf.append(nn.Linear(*nn_shape))

        elif self.nn_type == "resnet":

            n_input = n_hidden
            n_output = self.state_dim  # get next state and control point
            n_layers = n_layers - 1

            # create a GRU "branch" with a hidden size of n_hidden
            # self.rnn = torch.nn.GRUCell(self.state_dim,n_hidden)
            self.h0 = torch.zeros(1, self.rnn.hidden_size)
        else:
            raise ValueError(f"nn_type value {nn_type} unknown.")

        # create layers
        self.mlayers = nn.ModuleList()
        nn_shape = [n_input] + [n_hidden] * (n_layers) + [n_output]
        nnio = list(zip(nn_shape[:-1], nn_shape[1:]))

        for enum, (n_in, n_out) in enumerate(nnio):
            self.mlayers.append(nn.Linear(n_in, n_out))
            if enum == 0 and batch_norm:
                self.mlayers.append(nn.BatchNorm1d(n_out))

        # initialize weights
        self.initialize_weights()

        if transform_matrix is not None:
            self.transform_matrix = transform_matrix.to(self.device)
        else:
            self.transform_matrix = None

    def forward_fullyconnected(self, x, which_net="mlayers"):
        net = self.__getattr__(which_net)
        for i, layer in enumerate(net):
            x = layer(x) if i == 0 else layer(self.act(x))
        return x

    # @torch.jit.script
    def recurse(self, x, h):
        h = self.rnn(x, h)
        x = self.forward_fullyconnected(h, "mlayers")
        return x, h

    def forward_recurrent(self, x):
        # predict the last control point from the initial condtion
        h = self.h0.repeat(x.shape[0], 1).type_as(x)
        ret_x = [x[:, :, None]]

        for i in range(self.n_control_points - 1):
            x, h = self.recurse(x, h)
            ret_x.append(x[:, :, None])

        ret_x = torch.cat(ret_x, dim=-1)
        return ret_x.reshape(x.shape[0], -1)

    def forward_deeponet(self, x):
        basis = self.forward_fullyconnected(x, "mlayers")
        ret_x = torch.tensor([])
        ret_x = ret_x.type_as(x)

        ones = torch.ones(x.shape[0], 1)
        ones = ones.type_as(x)

        for i_cp in range(self.n_control_points):
            input_ = ones * i_cp
            cp_coeff = self.forward_fullyconnected(input_, "branch")
            leaf_input = torch.einsum("ij,ij->ij", basis, cp_coeff)
            ret_ = self.forward_fullyconnected(leaf_input, "leaf")
            ret_x = torch.cat([ret_x, ret_], 1)
        return ret_x

    def forward(self, x):
        if self.nn_type == "fullyconnected":
            return self.forward_fullyconnected(x)
        if self.nn_type == "recurrent":
            return self.forward_recurrent(x)
        if self.nn_type == "deeponet":
            return self.forward_deeponet(x)

    def transform(self, y):

        if self.transform_matrix is None:
            return y
        if self.transform_matrix.device != self.device:
            self.transform_matrix = self.transform_matrix.to(self.device)

        y = y.reshape(-1, self.state_dim, self.n_control_points).transpose(2, 1)

        return self.transform_matrix @ y

    def initialize_weights(self, seed=None):
        if seed:
            torch.manual_seed(seed)

        initialize_weights(self.mlayers, self.act.__class__.__name__.lower())

        # if self.nn_type == "recurrent":
        #    initialize_weights(self.rnn, self.act.__class__.__name__.lower())

        if self.nn_type == "deeponet":
            initialize_weights(self.branch, self.act.__class__.__name__.lower())
            initialize_weights(self.leaf, self.act.__class__.__name__.lower())


class LightningNet(Net):
    def __init__(
        self,
        state_dim,
        n_control_points,
        n_layers,
        n_hidden,
        activation_function,
        nn_type="fullyconnected",
        transform_matrix=None,
        time_step=None,
        weighting=None,
    ):
        Net.__init__(
            self,
            state_dim=state_dim,
            n_control_points=n_control_points,
            n_layers=n_layers,
            n_hidden=n_hidden,
            activation_function=activation_function,
            nn_type=nn_type,
            transform_matrix=transform_matrix,
            time_step=time_step,
            weighting=weighting,
        )

        self.optim = None
        self.scheduler = None
        self.lr = 0.01

    def set_optimizer(self, optimizer, learning_rate):
        self.learning_rate = learning_rate
        self.optim = optimizer(self.parameters(), lr=self.learning_rate)

    def set_scheduler(self, scheduler, schedule_step, schedule_gamma):
        self.schedule_step = schedule_step
        self.schedule_gamma = schedule_gamma
        self.scheduler = scheduler(self.optim, step_size=schedule_step, gamma=schedule_gamma)

    def set_loss(self, loss_function):
        self.loss_function = loss_function

    def training_step(self, batch, batch_idx):
        self.w = self.w.to(self.device)
        x, y = batch
        y_hat = self.transform(self(x) * self.w)  # transform of the output of the NN
        y = self.transform(y * self.w)  # transform of the "truth" cp that we were looking at.

        loss = self.loss_function(y_hat, y)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):

        self.w = self.w.to(self.device)
        x, y = batch
        y_hat = self.transform(self(x) * self.w)
        y = self.transform(y * self.w)

        loss = self.loss_function(y_hat, y)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        if not self.optim:
            raise ValueError("No optimizer")
        if not self.scheduler:
            return self.optim

        return [self.optim], [self.scheduler]


def initialize_weights(nnet, nonlinearity="relu"):
    gain = torch.nn.init.calculate_gain(nonlinearity)

    for layer in nnet:

        if isinstance(layer, nn.Linear) and nonlinearity == "relu":
            torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            torch.nn.init.normal_(layer.bias, 0.0, 0.01)
        elif isinstance(layer, nn.Linear):
            torch.nn.init.normal_(layer.weight, 0.0, 0.1)
            torch.nn.init.normal_(layer.bias, 0.0, 0.1)


def convert_to_meters(x, dt=1):
    """
    Example usage:
        m = torch.vmap(partial(reweight,dt=dt))(x)
    """
    x2 = x[[1, 3, 5]] * x[[1, 3, 5]]
    rx = (x2[2] + x2[1]) ** 0.5
    ry = (x2[0] + x2[2]) ** 0.5
    rz = (x2[1] + x2[0]) ** 0.5
    dt = torch.tensor(dt)
    l = torch.tensor(1)

    weight = torch.hstack([dt, l, dt, l, dt, l, dt * rx, rx, dt * ry, ry, dt * rz, rz])
    return x * weight
