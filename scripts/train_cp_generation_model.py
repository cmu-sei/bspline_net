"""
bspline_net (Github)

Copyright 2026 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to its own license.

DM26-0750
"""
# input: initial condition, number of control points
# output: suggested location of control points (n_cp-length vector)
# optimize: mse fit + distribution of points from uniform.

import numpy as np
import torch
from bsplinenet.ff_net import gen_data_np, ODESYS
from quadrotor.ellipsoid import gen_ic_ellips
from matplotlib import pyplot as plt

def bspline_basis(t, knots, n_cp, d):
    """
    Compute B-spline basis matrix in PyTorch (differentiable w.r.t. knots).
    
    t: (Ns,) - evaluation points in [0, 1]
    knots: (n_cp + d + 1,) - knot vector
    n_cp: number of control points
    d: degree
    
    Returns: (Ns, n_cp) basis matrix
    """
    Ns = t.shape[0]
    n_knots = knots.shape[0]
    
    # Initialize degree 0 basis functions
    # B_i,0(t) = 1 if knots[i] <= t < knots[i+1], else 0
    
    B = torch.zeros(Ns, n_knots - 1, device=knots.device)
    
    for i in range(n_knots - 1):
        left = t >= knots[i]
        right = t < knots[i + 1]
        B[:, i] = (left & right).to(t.dtype)
    
    # Handle last interval (include right endpoint)
    B[:, -1] = torch.where(t == knots[-1], torch.ones_like(t), B[:, -1])
    
    # Cox-de Boor recursion for higher degrees
    for k in range(1, d + 1):
        B_new = torch.zeros(Ns, n_knots - 1 - k, device=knots.device)
        
        for i in range(n_knots - 1 - k):
            # Left term: (t - knots[i]) / (knots[i+k] - knots[i]) * B[i, k-1]
            denom_left = knots[i + k] - knots[i]
            if denom_left > 1e-10:
                left = (t - knots[i]) / denom_left * B[:, i]
            else:
                left = torch.zeros(Ns, device=knots.device)
            
            # Right term: (knots[i+k+1] - t) / (knots[i+k+1] - knots[i+1]) * B[i+1, k-1]
            denom_right = knots[i + k + 1] - knots[i + 1]
            if denom_right > 1e-10:
                right = (knots[i + k + 1] - t) / denom_right * B[:, i + 1]
            else:
                right = torch.zeros(Ns, device=knots.device)
            
            B_new[:, i] = left + right
        
        B = B_new
    
    return B  # (Ns, n_cp)

def compute_fit_loss(knots, trajectories, n_cp, d):
    """
    knots: (batch, n_cp + d + 1)
    trajectories: (batch, Ns, n_state)
    
    Returns: scalar MSE, predicted_trajectories, predicted_controlpoints, knots
    """
    batch, Ns, n_state = trajectories.shape
    t = torch.linspace(0, 1, Ns, device=knots.device)
    
    total_loss = 0
    predicted_trajectories = torch.zeros_like(trajectories)
    predicted_controlpoints = torch.zeros(batch, n_cp, n_state, device=knots.device)
    
    for b in range(batch):
        Bit = bspline_basis(t, knots[b], n_cp, d)
        Y = trajectories[b]
        
        BtB = Bit.T @ Bit + 1e-6 * torch.eye(n_cp, device=knots.device)
        BtY = Bit.T @ Y
        coeffs = torch.linalg.solve(BtB, BtY)
        Y_fit = Bit @ coeffs
        total_loss += torch.mean((Y - Y_fit) ** 2)
        
        predicted_trajectories[b, :, :] = Y_fit
        predicted_controlpoints[b, :, :] = coeffs
    
    return total_loss / batch, predicted_trajectories, predicted_controlpoints, knots

def gen_ic_shell(n_dim, A, n_samples, min_radius=0.0):
    # Random directions on unit sphere
    directions = np.random.randn(n_dim, n_samples)
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    
    # Random radii in [min_radius, 1.0], uniform in volume
    # For uniform in shell volume: r ~ (min_r^n + u*(1 - min_r^n))^(1/n)
    u = np.random.rand(n_samples)
    radii = (min_radius**n_dim + u * (1 - min_radius**n_dim)) ** (1/n_dim)
    
    # Scale directions by radii and transform by A
    points = directions * radii
    return A @ points

class knot_model_lstm(torch.nn.Module):
    def __init__(self, dimensions=12, hidden_size=128, n_lstm_layers=2, n_cp_max=128, d=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_cp_max = n_cp_max
        self.d = d
        
        # Encode initial condition into LSTM hidden state
        self.ic_encoder = torch.nn.Sequential(
            torch.nn.Linear(dimensions, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size * n_lstm_layers * 2)  # h0 and c0 for each layer
        )
        self.n_lstm_layers = n_lstm_layers
        
        # LSTM: input is current cumulative position, output is next spacing
        self.lstm = torch.nn.LSTM(
            input_size=1,  # current position
            hidden_size=hidden_size,
            num_layers=n_lstm_layers,
            batch_first=True
        )
        
        # Output spacing
        self.spacing_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Softplus()
        )
        
        with torch.no_grad():
            # softplus(x) ~ x for x > 0, so small bias = small output
            self.spacing_head[0].weight.mul_(0.5)
            target_spacing = 1/(n_cp_max+1)  # want ~20 knots to reach 1.0
            bias_value = torch.log(torch.exp(torch.tensor(target_spacing)) - 1).item()
            self.spacing_head[0].bias.fill_(bias_value)
    
    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device
        
        # Encode IC into initial hidden state
        encoded = self.ic_encoder(x)
        encoded = encoded.view(batch_size, self.n_lstm_layers, 2, self.hidden_size)
        h0 = encoded[:, :, 0, :].permute(1, 0, 2).contiguous()
        c0 = encoded[:, :, 1, :].permute(1, 0, 2).contiguous()
        
        # Generate spacings autoregressively
        spacings = []
        current_pos = torch.zeros(batch_size, 1, 1, device=device)
        h, c = h0, c0
        
        max_interior = self.n_cp_max - self.d
        
        for i in range(max_interior):
            lstm_out, (h, c) = self.lstm(current_pos, (h, c))
            spacing = self.spacing_head(lstm_out[:, 0, :])
            spacings.append(spacing)
            current_pos = current_pos + spacing.unsqueeze(1)
        
        spacings = torch.cat(spacings, dim=1) + 1e-8  # (batch, max_interior)
        
        # Cumsum - NO normalization, can exceed 1
        #vec = torch.tanh(torch.cumsum(spacings, dim=1))*1.0
        vec = torch.cumsum(spacings, dim=1)
        vec = vec.clamp(max=1.0)

        # Build knot vector
        knots = torch.zeros(batch_size, self.n_cp_max + self.d + 1, device=device)
        knots[:, :self.d + 1] = 0.0
        knots[:, self.d + 1:-self.d] = vec
        knots[:, -self.d - 1:] = vec[:, -1:].clamp(min=1.0)  # ensure end knots > all interior
        return knots

n_samples =500
n_dim = 12
n_epochs = 100


#---- generate initial conditions
#initial_conditions = np.random.rand(n_samples, n_dim)
max_x = 15#2 # m
max_dx = 15#2 # m/s
max_ag = np.pi/4 # radians
max_da = 5 # rad/sec
R = np.array([max_dx,max_x,
                max_dx,max_x,
                max_dx,max_x,
                max_da,max_ag,
                max_da,max_ag,
                max_da,max_ag])
if n_dim>12 or n_dim<12:
    raise ValueError("The number of state space dimension for quadrotor is 12.")
A = np.eye(n_dim,n_dim)*R
d = 3
n_cp_max = 25
ti = 2.5
si = 0.005
Ns = int(ti / si) + 1

kmodel = knot_model_lstm(n_dim, n_cp_max = n_cp_max)

optimizer = torch.optim.Adam(kmodel.parameters(), lr=1e-3)

for epoch in range(n_epochs+1):    
    # Ramp minimum radius from 0 to 0.8 over training
    min_radius = min(0.1, epoch*2 /n_epochs)
    
    # Sample from shell between min_radius and 1.0
    initial_conditions = gen_ic_shell(n_dim, A, n_samples, min_radius=min_radius).T
    ics = torch.from_numpy(initial_conditions).to(torch.float32)
    knot_point_vector= kmodel(ics)
    spacings = knot_point_vector.diff()[:,d:-d-1]
    
    # Soft count of active knots
    min_value = torch.sigmoid(torch.tensor(20*(0-.9))) 
    max_value = torch.sigmoid(torch.tensor(20*(1-.9)))
    n_effective =n_cp_max-(((torch.sigmoid(20*(knot_point_vector-.9))-min_value)/max_value).sum(dim=1)-d-1)
    tenth_uniform_spacing = 1.0 / (n_effective - d - 1)/10
    ode_sys = ODESYS.QUADROTOR 
    ode_func = ode_sys.get_ode()
    tt = np.linspace(0, ti, Ns)
    trajectories = np.zeros((n_samples, Ns, n_dim))
    for i in range(n_samples):
        x1 = np.zeros( n_dim) 
        trajectories[i,:,:] = ode_func(initial_conditions[i,:], tt, (x1,)).astype(np.float32)
    
    loss_fit, predicted_trajectories, predicted_controlpoints, effective_knots = compute_fit_loss(knot_point_vector, 
                                torch.from_numpy(trajectories).to(torch.float32), 
                                n_cp_max, 
                                d) 
                                #threshold=tenth_uniform_spacing)
    
    
    optimizer.zero_grad()
    weight = 1e-2
    loss_total = loss_fit+weight*n_effective.mean()/n_cp_max
    print(f"\r Epoch {epoch:04d}  Loss: {loss_total:2.7f} Fit: {loss_fit:2.7f} {n_effective.detach().mean():2.4f} / {n_cp_max} min/max/std {n_effective.min():2.2f} {n_effective.max():2.2f} {n_effective.std():2.2f} ", end="")
    
    if epoch % 2 == 0:
        print()
        # Pick a random sample
        idx = np.random.randint(n_samples)
        
        # Get data for this sample
        y_true = trajectories[idx, :, :]  # (Ns, n_dim)
        y_spline = predicted_trajectories[idx, :, :].detach().numpy()  # (Ns, n_dim)
        cps = predicted_controlpoints[idx].detach().numpy()  # (n_cp, n_dim)
        knots_sample = knot_point_vector[idx].detach().numpy()
        
        # Time points
        t_eval = np.linspace(0, 1, Ns)
        
        # Control point positions (Greville abscissae)
        cp_positions = np.array([knots_sample[i+1:i+d+1].mean() for i in range(n_cp_max)])
        
        fig, axes = plt.subplots(4, 4, figsize=(14, 12))
        
        # First 12 subplots: trajectory dimensions
        for i in range(n_dim):
            ax = axes.flatten()[i]
            ax.plot(t_eval, y_true[:, i], 'b-', label="True")
            ax.plot(t_eval, y_spline[:, i], 'r--', label="Spline")
            ax.scatter(cp_positions, cps[:, i], c='g', s=30, zorder=5, label="CPs")
            ax.set_title(f"Dim {i}")
            if i == 0:
                ax.legend(fontsize=8)
        
        # 13th subplot: knot spacing heatmap (single sample)
        ax = axes.flatten()[12]
        interior_knots = knot_point_vector[idx, d+1:-d].detach().numpy()
        spacings_sample = np.diff(interior_knots)
        ax.imshow(spacings_sample.reshape(1, -1), aspect='auto', cmap='viridis', origin="lower")
        ax.set_title(f"Spacings (sample {idx})")
        ax.set_yticks([])
        ax.set_xlabel("Knot index")
        
        # 14th subplot: knot spacing heatmap (all samples)
        ax = axes.flatten()[13]
        all_spacings = knot_point_vector[:, d+1:-d].diff(dim=1).detach().numpy()
        ax.imshow(all_spacings, aspect='auto', cmap='viridis')  # first 20 samples
        ax.set_title("Spacings (batch)")
        ax.set_ylabel("Sample")
        ax.set_xlabel("Knot index")
        
        # 15th subplot: knot positions on line
        ax = axes.flatten()[14]
        for i in range(n_samples):
            positions = knot_point_vector[i, d+1:-d].detach().numpy()
            ax.hlines(i, 0, 1, colors='gray', alpha=0.1)
            ax.scatter(positions, np.ones_like(positions) * i, s=7)
        ax.set_xlim(-0.05, 1.05)
        ax.set_title("Knot positions")
        ax.set_xlabel("Position [0,1]")
        
        # 16th subplot: empty or loss curve
        ax = axes.flatten()[15]
        ax.text(0.5, 0.5, f"Epoch {epoch}\nloss_fit: {loss_fit:.6f}\nn_eff: {n_effective.detach().mean():.2f}", 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"scripts/fit_epoch_{epoch:04d}.png")
        plt.close()
    
    loss_total.backward()
    optimizer.step()


        



