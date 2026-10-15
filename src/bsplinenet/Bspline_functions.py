"""
bspline_net (Github)

Copyright 2026 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to its own license.

DM26-0750
"""
# PAPER: shared
import numpy as np
import torch

"""
BsFun - BsKnots - leastSquaresInversion are used to approximate a signal of n components
over a finite period of time. 

For each component there is a Bspline defined over the time interval, where the 
control points are uniformly distributed over the time interval (x-axis), and the Bspline parameter
t used to compute curve, corresponds to the time.

While the control points are fixed in time (x-axis), they vary on the y-axis. The functions
BsFun and BsKnots are used to generates the matrices to compute the pseudo-inverse that 
generates the y-axis of each control points.

For a signal of n components and n_cp control points each, the unknown vector is (n x n_cp) x 1 
vector. 

"""


def BsFun(i, d, t, Ln) -> float:
    """BSFUN Computes the value of the B-spline function (Cox - De Boor Algorithm)

    Args:
        i (_type_): index of the interval
        d (_type_): B-spline order
        t (_type_): parameter
        Ln (_type_): knots vector

    Returns:
        float: B - B-spline function value
    """
    if d == 0:
        return int(Ln[i] <= t < Ln[i + 1])

    a = 0 if (Ln[d + i] - Ln[i]) == 0 else (t - Ln[i]) / (Ln[d + i] - Ln[i])
    b = 0 if (Ln[d + i + 1] - Ln[i + 1]) == 0 else (Ln[d + i + 1] - t) / (Ln[d + i + 1] - Ln[i + 1])
    B = a * BsFun(i, d - 1, t, Ln) + b * BsFun(i + 1, d - 1, t, Ln)

    return B


def BsKnots(n_cp, d, Ns, Ln):
    """
    B-spline basis matrix with user-provided normalized knot vector Ln
    Ln is normalized between 0 and 1, and it contains n_cp+d+1 knot points
    where the last and initial d+1 points are equal to 1 and 0
    respectively.

    Returns:
        tk: B-spline parameter vector;
        Ln: Knots Vector;
        Bit: B-spline functions, each column corresponds a control point
    """
    # Parameter vector using final knot (usually = 1)
    tk = np.linspace(0, 1 - 1e-10, Ns)
    # tk = np.linspace(0, Ln[-1] - Ln[-1] / Ns, Ns)

    # Evaluate basis functions
    Bit = np.zeros((Ns, n_cp))
    for j in range(n_cp):
        for i in range(Ns):
            Bit[i, j] = BsFun(j, d, tk[i], Ln)

    return tk, Ln, Bit


# def BsKnots(n_cp, d, Ns):
#    """BSKNOTS B-spline parameter vector (tk) and knots vector (Ln)
#    Args:
#        n_cp: number of control points
#        d: Bspline order
#        Ns: length of the parameter vector tk
#    Returns:
#        tk: B-spline parameter vector;
#        Ln: Knots Vector;
#        Bit: B-spline functions, each column corresponds a control point
#    """
#    k = 1
#    j = 1
#    n_knots = n_cp + d + 1
#    # Knots Vector
#    Ln = np.zeros(n_knots)
#
#    for i in range(d + 1, n_knots):
#        Ln[i] = k
#        if i < n_cp:
#            k = k + j
#
#    # Parameter Vector
#    tk = np.arange(0, Ln[-1], Ln[-1] / Ns)
#
#    Bit = np.zeros((Ns, n_cp))
#
#    for j in range(0, n_cp):
#        for i in range(0, Ns):
#            Bit[i, j] = BsFun(j, d, tk[i], Ln)
#
#    return tk, Ln, Bit


def leastSquaresInversion(Phi, Psi, T=None):
    """
    LEASTSQUARESINVERSION computes the least square solution of the
    approximated model stable inversion approach.
    Inputs:
           Phi   :Matrix containing the coefficients values C(t)
           Psi   :Vector containing the measurements values z(t)
           T     :Coordinate transformation matrix for smoothness constraints
                  in case of 3rd order polynomial basis functions
    Outputs:
           x    :Estimated basis function coefficients
    """
    # Singular Value Decomposition
    if T is None:
        T = np.eye(Phi.shape[0])

    U, S, V_T = np.linalg.svd(T.T @ Phi @ T, full_matrices=True)
    x = T @ V_T.T @ ((U.T @ T.T) @ Psi / S)
    return x


def Bs_comp(Ns, x, n_cp, q, Bit):
    """
    Computes the values of B-spline functions using Cox - De Boor Algorithm.

    Parameters:
    - Ns: Number of samples
    - x: Coefficient vector
    - n_cp: Number of control points
    - q: state dimension

    Returns:
    - sp1: Matrix of computed B-spline functions (q x Ns)
    """
    xx = x.reshape((n_cp, q), order="F")
    sp1 = (Bit @ xx).T
    # sp1 = np.zeros((q, Ns))
    ##tk, Ln, Bit = BsKnots(n_cp, q, Ns)
    # for i in range(0, Ns):
    #    for k in range(q):
    #        for j in range(n_cp):
    #            sp1[k, i] += Bit[i, j] * x[j + k * n_cp]
    return sp1


# Example of usage:
# (UNCOMMENT THIS PART TO TEST THE FUNCTIONS)


def duffing_sys(x, t, *args):
    # Define the Duffing system here
    return np.array([x[1], x[0] - x[0] ** 3])


def rotate(p, q, r, every_other=True, nd=3):
    if not nd == 3:
        raise ValueError("this only works for 3 dimensions")
    Rp = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])

    Rq = np.array([[np.cos(q), 0, np.sin(q)], [0, 1, 0], [-np.sin(q), 0, np.cos(q)]])

    Rr = np.array([[np.cos(r), np.sin(r), 0], [-np.sin(r), np.cos(r), 0], [0, 0, 1]])

    def rot(inp_matrix, n_cp=50):
        matrix_type = type(inp_matrix)
        ndim = inp_matrix.ndim
        if ndim == 3:
            inp_matrix = inp_matrix.reshape(inp_matrix.shape[0], -1)

        if every_other:
            skip = 2
            if np.mod(inp_matrix.shape[1], n_cp * nd * 2) > 0:
                raise ValueError(
                    "must be a multiple of the number of control points and dimensions x 2"
                )
        else:
            skip = 1
            if np.mod(inp_matrix.shape[1], n_cp * nd) > 0:
                raise ValueError(
                    "must be a multiple of the number of control points and dimensions"
                )
        z = n_cp * nd * skip  # 50 * 3 * 2 = 300

        list_mat = []
        for i in np.arange(0, inp_matrix.shape[1], z):
            # 0, 300 (for skip =2), 0,150,300,450 (for skip=1)
            for j in np.arange(0, skip * n_cp, n_cp):
                #  0, 50 (for skip=2), 0 (for skip=1)
                idx = i + j
                # print([k for k in np.arange(idx, idx+n_cp*nd*skip,n_cp*skip)])
                list_mat.append(
                    [
                        inp_matrix[:, k : k + n_cp]
                        for k in np.arange(idx, idx + n_cp * nd * skip, n_cp * skip)
                    ]
                )

        list_mat_new = [Rr @ (Rq @ (Rp @ np.hstack(l).reshape(-1, nd, n_cp))) for l in list_mat]
        if skip == 1:
            list_mat_new = [i for i in list_mat_new]
        if skip == 2:
            tc = int(len(list_mat_new) / 2)
            list_mat_ = []
            for k in range(0, tc + 1, 2):
                for i in range(nd):
                    for j in range(2):
                        list_mat_.append(list_mat_new[k + j][:, i, :])
            list_mat_new = list_mat_

        ret = np.hstack(list_mat_new)
        if matrix_type == torch.Tensor:
            ret = torch.tensor(ret).to(torch.float32)
        if ndim == 3:
            ret = ret.reshape(ret.shape[0], -1, n_cp)
        return ret

    return rot


def to_cylindrical(input_t, ncp):
    """
    for dx,x,dy,y,dz,z,da,a,db,b,dc,c ordered vector
    changes dx to dR
    changes x to R
    changes dy to dTheta
    changes y to theta

    """
    if type(input_t) == np.ndarray:
        X = input_t.copy()
        arccos = np.arccos
        minimum = np.minimum
        abs_ = np.abs
        pi = np.pi
        zeros = np.zeros

    if type(input_t) == torch.Tensor:
        X = input_t.detach().clone()
        arccos = torch.arccos
        minimum = torch.minimum
        abs_ = torch.abs
        pi = torch.pi
        zeros = torch.zeros

    x = X[:, ncp : 2 * ncp]
    y = X[:, 3 * ncp : 4 * ncp]

    dx = X[:, 0:ncp]
    dy = X[:, 2 * ncp : 3 * ncp]
    R = np.sqrt(x**2 + y**2)
    # print(R.shape, dx.shape, x.shape)
    dR = (dx * x + dy * y) / R

    cos_theta = x / R
    sin_theta = y / R
    q = minimum(sin_theta / abs_(sin_theta), zeros(sin_theta.shape))
    theta = -2 * q * pi + arccos(cos_theta) * (1 + 2 * q)
    dtheta = (dy * x - dx * y) / R

    X[:, 0:ncp] = dR
    X[:, ncp : 2 * ncp] = R
    X[:, 2 * ncp : 3 * ncp] = dtheta
    X[:, 3 * ncp : 4 * ncp] = theta

    return X


"""
# Simulation parameters
Ts = 0.01     #sampling time
T = 10        #time interval  
tt = np.arange(0, T + Ts, Ts)  #vector of time
Ns = tt.size  #total number of samples
n = 2         #dimension of the state space

# Simulations
N_ic = 1      # number of initial conditions
# Generation of the initial conditions
a = np.array([1, 1])
b = -a
#x0 = a + np.random.rand(N_ic, n) * (b - a)
#x0 = np.array([0.5047, -0.9708])
x0 = np.array([0.5047, -0.9708]).reshape((1, 2))


# Bspline definition
n_cp = 30      #number of control points
d = 3          #Bspline order

Ln = np.linspace(0,1, Ncp)
tk, Ln, Bit = BsKnots(n_cp, d, Ns, Ln)  #Bit contains the values of the Bspline function over the time (Ns x n_cp)

print(Bit.shape)
#Least-Squares formulation
#C = np.block([[Bit, np.zeros_like(Bit)], [np.zeros_like(Bit), Bit]])
C = np.block([[Bit if i == j else np.zeros_like(Bit) for j in range(n)] for i in range(n)])
Phi = C.T @ C

x_tot = np.zeros((n_cp * n, N_ic)) #It saves the control points for each component and for each IC

print('Progress:    ', end='')

for j in range(0, N_ic):
    # Data generation
    xx = odeint(duffing_sys, x0[j, :], tt)
    ym = np.concatenate((xx[:, 0], xx[:, 1]))
    # Least squares computation
    Psi = C.T @ ym
    x, sp1 = leastSquaresInversion(Phi, Psi, n_cp, n, Ns, Bit)
    x_tot[:, j] = x

    # Update progress bar
    progress = (j+1) / N_ic * 100
    print('\b\b\b\b%3.0f%%' % progress, end='')


print('\n')

# x-coordinate of the control points
h = T / n_cp
ci_x = np.arange(0, T, h)  

# Plot the results

fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

axs[0].plot(tt, sp1[0, :])
axs[0].plot(tt, xx[:, 0])
axs[0].plot(ci_x, x[:n_cp], 'ro')
axs[0].set_ylabel('$x_1$')

axs[1].plot(tt, sp1[1, :])
axs[1].plot(tt, xx[:, 1])
axs[1].plot(ci_x, x[n_cp:], 'ro')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('$x_1$')

plt.subtitle('Least Squares Inversion Coefficients')
plt.show()
"""
