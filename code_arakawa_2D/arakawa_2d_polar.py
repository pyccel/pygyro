import os
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import eye as sparse_id

import matplotlib.pyplot as plt

from utils_polar import plot_gridvals, plot_time_diag, make_movie
from discrete_brackets_polar import assemble_Jpp, assemble_Jpx, assemble_Jxp

# Choose which bracket to use
#bracket = '++'
#bracket = '+x'
#bracket = 'x+'
bracket = 'akw'

# if code should print information during execution
verbose = True

# Choose if explicit scheme should be used or not
explicit = False

# Number of grid points for each variable
N0_nodes_theta = 200
N0_nodes_r = 100

# Number of time steps, total time, and time stepping
Nt = 400
T = 2
dt = T/Nt

if verbose:
    print(f'dt = {dt}')

# Parameters for the plot
nb_plots = 100
movie_duration = 8

# If plots should be shown or just saved
show_plots = False


def init_f(rt):
    """
    Initialises f as a Gaussian function

    Parameters
    ----------
        rt : list
            rt[0] is theta-value, rt[1] is rho-value

    Returns
    -------
        f : float
            the  value for f at [theta, rho]
    """
    # center values (shifts) for Guassian
    f0_c = [0, -.1]

    # standard deviation for Gaussian
    f0_s = .1

    x = rt[1] * np.cos(rt[0])
    y = rt[1] * np.sin(rt[0])

    return np.exp(-((x - f0_c[0])**2 + (y - f0_c[1])**2) / (2 * f0_s**2))


def phi_ex(rt):
    """
    Initialises phi as a Gaussian function

    Parameters
    ----------
        rt : list
            rt[0] is theta-value, rt[1] is rho-value

    Returns
    -------
        f : float
            the  value for f at [theta, rho]
    """
    # shift and standard ddeviation values for Gaussian
    phi_c = [0.5, 0.5]
    phi_s = .5

    x = rt[1] * np.cos(rt[0])
    y = rt[1] * np.sin(rt[0])

    return np.exp(-((x - phi_c[0])**2 + (y - phi_c[1])**2) / (2 * phi_s**2))

# ---- ---- ---- ---- ---- ---- ---- ----
# grid
# indexing: f[i0 + i1*N0_nodes] = f(i0*h,i1*h)


# initialise the grid
grid_theta = np.linspace(0, 2*np.pi, num=N0_nodes_theta)  # theta
grid_r = np.linspace(0.01, 1.01, num=N0_nodes_r)  # r

if verbose:
    print(f'\ngrid for theta : \n {grid_theta}')
    print(f'\ngrid for r : \n {grid_r} \n')

# compute grid spacing
dtheta = (grid_theta[-1] - grid_theta[0]) / (len(grid_theta) - 1)
dr = (grid_r[-1] - grid_r[0]) / (len(grid_r) - 1)

if verbose:
    print(f'dtheta  = {dtheta}')
    print(f'dr  = {dr}')

# total number of nodes
N_nodes = N0_nodes_theta * N0_nodes_r

grid = np.array([[grid_theta[k % N0_nodes_theta], grid_r[k//N0_nodes_theta]]
                for k in range(N_nodes)])

if verbose:
    print(grid)

#grid_cart = np.array([pol2cart(grid[k][0], grid[k][1]) for k in range(len(grid))])
# grid0_cart = np.linspace(0, 1, num=N0_nodes) #x
# grid1_cart = np.linspace(-1, 1, num=N0_nodes) #y
#plt.scatter([grid_cart[k][0] for k in range(len(grid))], [grid_cart[k][1] for k in range(len(grid))] )
# plt.show()

# ---- ---- ---- ---- ---- ---- ---- ----
# plots

method = bracket
if explicit:
    method += '_exp'
else:
    method += '_imp'

plot_dir = './plots/polar_run_' + method + '/'
if not os.path.exists(plot_dir):
    if verbose:
        print('creating directory ' + plot_dir)
    os.makedirs(plot_dir)

if nb_plots > 0:
    plot_period = max(int(Nt/nb_plots), 1)
else:
    plot_period = Nt

# movie_fn = plot_dir+'f_movie.gif'
movie_fn = plot_dir + 'f_movie.mp4'
frames_list = []


# ---- ---- ---- ---- ---- ---- ---- ----
# operators
# scaling for the integrals
r_scaling = [grid_r[k//N0_nodes_theta] for k in range(N_nodes)]

# r_mat = np.diag(r_scaling)
# print(r_mat.shape)


def get_total_f(f):
    """
    Compute the integral of f over the whole domain

    Parameters
    ----------
        f : array[float]
            array containing the values of f

    Returns
    -------
        float
            the sum over all values of f
    """
    f_s = f @ np.diag(r_scaling)

    return np.sum(f_s)


def get_total_f2(f):
    """
    Compute the integral of f^2 over the whole domain

    Parameters
    ----------
        f : array[float]
            array containing the values of f

    Returns
    -------
        float
            the sum over all squared values of f
    """
    f_s = f @ np.diag(r_scaling)

    # pointwise multiplication
    f2 = np.multiply(f, f_s)

    return np.sum(f2)


def get_total_energy(f, phi):
    """
    Compute the totoal energy, i.e. the integral of f times phi over the whole domain

    Parameters
    ----------
        f : array[float]
            array containing the values of f

        phi : array[float]
            array containing the values of phi

    Returns
    -------
        float
            the total energy
    """
    phi_s = phi @ np.diag(r_scaling)

    # pointwise multiplication
    f_phi = np.multiply(f, phi_s)

    return np.sum(f_phi)


phi = np.array(list(map(phi_ex, grid)))


#X, Y = np.meshgrid(grid0, grid1)
#phi = phi.reshape(X.shape)
#fig = plt.figure()
#ax = fig.gca(polar=True)
#ax = plt.subplots(subplot_kw=dict(projection='polar'))
#ax.contour(X, Y, phi)
# fig.savefig('hi')
# plt.show()

plot_gridvals(grid_theta, grid_r, [phi], f_labels=['phi'], title='phi',
              show_plot=show_plots, plt_file_name=plot_dir+'phi.png')

# assemble discrete brackets as sparse matrices
# for f -> J(phi,f) = d_y phi * d_x f - d_x phi * d_y f

Jpp_phi = assemble_Jpp(phi, N0_nodes_theta, N0_nodes_r, grid_r)
Jpx_phi = -assemble_Jpx(phi, N0_nodes_theta, N0_nodes_r, grid_r)
Jxp_phi = -assemble_Jxp(phi, N0_nodes_theta, N0_nodes_r, grid_r)

Jpp_phi /= 4 * dr * dtheta
Jpx_phi /= 4 * dr * dtheta
Jxp_phi /= 4 * dr * dtheta

if bracket == '++':
    J_phi = Jpp_phi
elif bracket == '+x':
    J_phi = Jpx_phi
elif bracket == 'x+':
    J_phi = Jxp_phi
elif bracket == 'akw':
    J_phi = (Jpp_phi + Jpx_phi + Jxp_phi) / 3
else:
    raise NotImplementedError(f'{bracket} is not a valid bracket')


if verbose:
    print('tests:')
    print(
        f'max of J_phi times unit matrix : {max(abs(J_phi @ np.ones(len(grid))))}')
    # print(f'max of J_phi times scaled phi: {max(abs(J_phi @ np.diag(r_scaling_inv) @ phi))}')
    print(f'max of J_phi times phi: {max(abs(J_phi @ phi))}')


if explicit:
    I = A = B = None
else:
    I = sparse_id(N_nodes)
    A = I - dt/2 * J_phi
    B = I + dt/2 * J_phi

# f0 on grid
f = np.array(list(map(init_f, grid)))
plt_fn = plot_dir + 'f0.png'
plot_gridvals(grid_theta, grid_r, [f], f_labels=['f0'], title='f0',
              show_plot=show_plots, plt_file_name=plt_fn)
frames_list.append(plt_fn)

# diags
total_f = np.zeros(Nt + 1)
total_energy = np.zeros(Nt + 1)
total_f2 = np.zeros(Nt + 1)

total_energy[0] = get_total_energy(f, phi)
total_f[0] = get_total_f(f)
total_f2[0] = get_total_f2(f)

for nt in range(Nt):
    if verbose:
        print(f"\n computing f^n for n={nt+1} of {Nt}")
        print(f'total energy : {total_energy[nt]}')
        print(f'total f : {total_f[nt]}')
        print(f'total f^2 : {total_f2[nt]}')
        print(f'sum of f*J_phi*f : {abs(sum(np.multiply(f, (J_phi.dot(f)))))}')

    total_energy[nt+1] = get_total_energy(f, phi)
    total_f[nt+1] = get_total_f(f)
    total_f2[nt+1] = get_total_f2(f)

    if explicit:
        f[:] += dt * J_phi.dot(f)
    else:
        # print('solving source problem with scipy.spsolve...')
        f[:] = spsolve(A, B.dot(f))

    # visualisation
    nt_vis = nt + 1
    if nt_vis % plot_period == 0 or nt_vis == Nt:
        plt_fn = plot_dir + 'f_{}.png'.format(nt_vis)
        plot_gridvals(grid_theta, grid_r, [f], f_labels=['f_end'], title='f end',
                      show_plot=show_plots, plt_file_name=plt_fn)
        frames_list.append(plt_fn)


plot_time_diag(total_energy, Nt, dt, plot_dir, name="energy")
plot_time_diag(total_f, Nt, dt, plot_dir, name="f")
plot_time_diag(total_f2, Nt, dt, plot_dir, name="f2")

make_movie(frames_list, movie_fn, frame_duration=movie_duration/nb_plots)
