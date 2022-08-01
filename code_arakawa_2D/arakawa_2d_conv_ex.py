import os
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import eye as sparse_id

from numpy.linalg import norm
import matplotlib.pyplot as plt

from utils import plot_gridvals, plot_time_diag, make_movie
from discrete_brackets import assemble_Jpp, assemble_Jpx, assemble_Jxp

# Choose which bracket to use
#bracket = '++'
#bracket = '+x'
#bracket = 'x+'
bracket = 'akw'

# if code should print information during execution
verbose = False

# Choose if explicit scheme should be used or not
explicit = False


# center values (shifts) for Guassian
f0_c = [.4, .5]

# standard deviation for Gaussian
f0_s = .07

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

    x = rt[0] 
    y = rt[1] 
    return np.exp(-((x - f0_c[0])**2 + (y - f0_c[1])**2) / (2 * f0_s**2))


v1 = 0.3
v2 = -.2

def phi_ex(rt):
    x = rt[0] 
    y = rt[1] 
    return -y*v1 + x*v2 


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

    return h_grid**2 * sum(f)


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

    # pointwise multiplication
    f2 = np.multiply(f, f)

    return h_grid**2 * sum(f2)


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
    # pointwise multiplication
    f_phi = np.multiply(f, phi)

    return h_grid**2 *sum(f_phi)

# ---- ---- ---- ---- ---- ---- ---- ----
# plots

method = bracket
if explicit:
    method += '_exp'
else:
    method += '_imp'

plot_dir = './plots/conv_' + method + '/'
if not os.path.exists(plot_dir):
    if verbose:
        print('creating directory ' + plot_dir)
    os.makedirs(plot_dir)

# ---- ---- ---- ---- ---- ---- ---- ----
# grid
# indexing: f[i0 + i1*N0_nodes] = f(i0*h,i1*h)

    # Number of grid points for each variable
N0 = [50, 100, 150, 200]
dof = np.multiply(np.sqrt(N0), np.sqrt(N0))
time_steps = [100, 100, 100, 100]
T = 1

def f_ex(T, rt):

    x = rt[0] -T*v1
    y = rt[1] -T*v2

    return init_f([x, y])

error_f = []
error_energy = []
error_int = []
error_int2 = []

for k in range(len(N0)):
    N0_nodes = N0[k]

    #cfl = dt/dx
    #CFL = 0.5
    # Number of time steps, total time, and time stepping
    #dt = CFL * 1/(N0_nodes-1)
    
    
    Nt = time_steps[k]
    dt = T/Nt

    # initialise the grid
    h_grid = 1/(N0_nodes-1)
    grid0 = np.linspace(0, 1, num=N0_nodes)
    grid1 = np.linspace(0, 1, num=N0_nodes)
    N_nodes = N0_nodes * N0_nodes
    grid = np.array([[grid0[k%N0_nodes], grid1[k//N0_nodes]] for k in range(N_nodes)])

 
    # ---- ---- ---- ---- ---- ---- ---- ----
    # operators
    # scaling for the integrals

    phi = np.array(list(map(phi_ex, grid)))

    Jpp_phi = assemble_Jpp(phi, N0_nodes)
    Jpx_phi = -assemble_Jpx(phi, N0_nodes)
    Jxp_phi = -assemble_Jxp(phi, N0_nodes)

    Jpp_phi /= 4 * h_grid**2
    Jpx_phi /= 4 * h_grid**2
    Jxp_phi /= 4 * h_grid**2

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

    if explicit:
        I = A = B = None
    else:
        I = sparse_id(N_nodes)
        A = I - dt/2 * J_phi
        B = I + dt/2 * J_phi

    # f0 on grid
    f = np.array(list(map(init_f, grid)))

    for nt in range(Nt):

        if explicit:
            f[:] += dt * J_phi.dot(f)
        else:
            # print('solving source problem with scipy.spsolve...')
            f[:] = spsolve(A, B.dot(f))


    f_x = lambda x: f_ex(Nt * dt, x)
    f_exact = np.array(list(map(f_x, grid)))

 
    err_f = max(np.abs((f - f_exact)))/max(np.abs((f_exact)))
    print(err_f)

    err_energy = np.sqrt( (get_total_energy(f, phi) - get_total_energy(f_exact, phi))**2 )
    err_int = np.sqrt( (get_total_f(f) - get_total_f(f_exact))**2 )
    err_int2 = np.sqrt( (get_total_f2(f) - get_total_f2(f_exact))**2 )

    plt_fn = plot_dir + 'f_{}.png'.format(k)
    plot_gridvals(grid0, grid1, [f], f_labels=['f'], title='f',
                      show_plot=False, plt_file_name=plt_fn)
    plt_fn = plot_dir + 'f_ex_{}.png'.format(k)
    plot_gridvals(grid0, grid1, [f_exact], f_labels=['fex'], title='fex',
                      show_plot=False, plt_file_name=plt_fn)

    error_f.append(err_f)
    error_energy.append(err_energy)
    error_int.append(err_int)
    error_int2.append(err_int2)


fig, ax = plt.subplots( )
ax.loglog()
#ax.set_title("error")
ax.plot(dof, error_f, label = "error")

ax.plot(dof, [ (5/dof[i])**1 for i in range(len(dof))], label="order 1", linestyle='--')
ax.plot(dof, [ (20/dof[i])**2 for i in range(len(dof))], label="order 2", linestyle='--')
ax.legend(loc='upper right')
ax.set_title("Translation with Arakawa in Cartesian Coordinates")
ax.set_xlabel("grid size")
ax.set_ylabel("relative error")
plt.savefig(plot_dir + 'error.png', dpi=300)

fig, ax = plt.subplots()
ax.plot(dof, error_energy, label="energy")
ax.plot(dof, error_int, label="integral")
ax.plot(dof, error_int2, label="square integral")
ax.legend(loc='upper right')
plt.savefig(plot_dir + 'conserved_quant.png', dpi=300)

