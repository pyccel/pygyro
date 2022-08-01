import os
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import eye as sparse_id

import matplotlib.pyplot as plt

from utils_polar import get_total_f, get_total_f2, get_total_energy, plot_gridvals, plot_time_diag, make_movie
from discrete_brackets_polar import assemble_bracket


def solve_Arakawa_advection(example, bc,
                            N_theta, N_r,
                            T, Nt,
                            domain_theta, domain_r,
                            nb_plots=10, movie_duration=8,
                            bracket='akw',
                            verbose=True, explicit=False,
                            show_plots=False, plot_dir=None,
                            convergence=False):
    """
    TODO

    Parameters
    ----------
        TODO

    Returns
    -------
        TODO
    """

    dt = T/Nt

    if verbose:
        print(f'dt = {dt}')

    # center values (shifts) for Guassian
    f0_c = [.3, .4]

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

        x = rt[1] * np.cos(rt[0])
        y = rt[1] * np.sin(rt[0])

        return np.exp(-((x - f0_c[0])**2 + (y - f0_c[1])**2) / (2 * f0_s**2))

    if example == 'gaussian':

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
            # shift and standard deviation values for Gaussian
            phi_c = [0, 0]
            phi_s = .5

            x = rt[1] * np.cos(rt[0])
            y = rt[1] * np.sin(rt[0])

            return np.exp(-((x - phi_c[0])**2 + (y - phi_c[1])**2) / (2 * phi_s**2))

    elif example == 'translation':
        v1 = -.3
        v2 = .4

        def phi_ex(rt):
            x = rt[1] * np.cos(rt[0])
            y = rt[1] * np.sin(rt[0])

            return y*v1 - x*v2

        def f_ex(rt):

            x = rt[1] * np.cos(rt[0]) - T*v1
            y = rt[1] * np.sin(rt[0]) - T*v2

            return np.exp(-((x - f0_c[0])**2 + (y - f0_c[1])**2) / (2 * f0_s**2))

    elif example == 'transformation':
        print("todo")
    # ---- ---- ---- ---- ---- ---- ---- ----
    # grid
    # indexing: f[i0 + i1*N0_nodes] = f(i0*h,i1*h)

    # initialise the grid
    grid_theta = np.linspace(
        domain_theta[0], domain_theta[1], num=N_theta)  # theta
    grid_r = np.linspace(domain_r[0], domain_r[1], num=N_r)  # r

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
    N_nodes = N_theta * N_r

    grid = np.array([[grid_theta[k % N_theta], grid_r[k//N_theta]]
                    for k in range(N_nodes)])

    # get boundary indices
    if bc == 'dirichlet':
        ind_bd = []
        for k in range(N_nodes):
            if k//N_theta == 0 or k//N_theta == N_r - 1:
                ind_bd.append(k)
        ind_bd = np.array(ind_bd)

    # ---- ---- ---- ---- ---- ---- ---- ----
    # plots
    if nb_plots > 0:

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
    r_scaling = [grid_r[k//N_theta] for k in range(N_nodes)]

    # phi on grid
    phi = np.array(list(map(phi_ex, grid)))

    # f0 on grid
    f = np.array(list(map(init_f, grid)))
    # enforce dirichlet BC
    if bc == 'dirichlet':
        f[ind_bd] = np.zeros(len(ind_bd))
        phi[ind_bd] = np.zeros(len(ind_bd))

    if nb_plots > 0:
        plot_gridvals(grid_theta, grid_r, [phi], f_labels=['phi'], title='phi',
                      show_plot=show_plots, plt_file_name=plot_dir + 'phi.png')

    # assemble discrete brackets as sparse matrices
    # for f -> J(phi,f) = d_y phi * d_x f - d_x phi * d_y f
    phi_hh = 1/(4 * dr * dtheta) * phi
    J_phi = assemble_bracket(bracket, bc, phi_hh, N_theta, N_r, grid_r)

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

    if nb_plots > 0:
        plt_fn = plot_dir + 'f0.png'
        plot_gridvals(grid_theta, grid_r, [f], f_labels=['f0'], title='f0',
                      show_plot=show_plots, plt_file_name=plt_fn)
        frames_list.append(plt_fn)

    total_f = np.zeros(Nt + 1)
    total_energy = np.zeros(Nt + 1)
    total_f2 = np.zeros(Nt + 1)

    total_energy[0] = get_total_energy(dr, dtheta, r_scaling, f, phi)
    total_f[0] = get_total_f(dr, dtheta, r_scaling, f)
    total_f2[0] = get_total_f2(dr, dtheta, r_scaling, f)

    for nt in range(Nt):
        if verbose:
            print(f"\n computing f^n for n={nt + 1} of {Nt}")
            print(f'total energy : {total_energy[nt]}')
            print(f'total f : {total_f[nt]}')
            #print( trapezoidal_rule_2d(np.multiply(r_scaling, f)) )
            print(f'total f^2 : {total_f2[nt]}')
            print(
                f'sum of f*J_phi*f : {abs(sum(np.multiply(f, (J_phi.dot(f)))))}')

        total_energy[nt + 1] = get_total_energy(dr, dtheta, r_scaling, f, phi)
        total_f[nt + 1] = get_total_f(dr, dtheta, r_scaling, f)
        total_f2[nt + 1] = get_total_f2(dr, dtheta, r_scaling, f)

        if explicit:
            f[:] += dt * J_phi.dot(f)
        else:
            # print('solving source problem with scipy.spsolve...')
            f[:] = spsolve(A, B.dot(f))

        # visualisation
        if nb_plots > 0:
            nt_vis = nt + 1
            if nt_vis % plot_period == 0 or nt_vis == Nt:
                plt_fn = plot_dir + 'f_{}.png'.format(nt_vis)
                plot_gridvals(grid_theta, grid_r, [f], f_labels=['f_end'], title='f end',
                              show_plot=show_plots, plt_file_name=plt_fn)
                frames_list.append(plt_fn)

    if nb_plots > 0:
        plot_time_diag(total_energy, Nt, dt, plot_dir, name="energy")
        plot_time_diag(total_f, Nt, dt, plot_dir, name="f")
        plot_time_diag(total_f2, Nt, dt, plot_dir, name="f2")

    if movie_duration > 0:
        make_movie(frames_list, movie_fn,
                   frame_duration=movie_duration/nb_plots)

    if convergence == True:
        f_exact = np.array(list(map(f_ex, grid)))
        f_exact[ind_bd] = np.zeros(len(ind_bd))
        if nb_plots > 0:
            plt_fn = plot_dir + 'f_ex_{}.png'.format(k)
            plot_gridvals(grid_theta, grid_r, [f_exact], f_labels=['fex'], title='fex',
                          show_plot=False, plt_file_name=plt_fn)

        dof = np.sqrt(N_nodes)
        err = max(np.abs(f - f_exact)) / max(np.abs(f_exact))

        return dof, err


def main():
    # what example to run
    #example = 'gaussian'
    example = 'translation'

    domain_theta = [0, 2 * np.pi]
    domain_r = [0.1, 1.1]

    # Number of grid points for each variable
    N_theta = 100
    N_r = 100

    # Number of time steps, total time, and time stepping
    Nt = 100
    T = 1

    # Choose which bracket to use
    #bracket = '++'
    #bracket = '+x'
    #bracket = 'x+'
    bracket = 'akw'

    #bc = 'periodic'
    bc = 'dirichlet'

    # if code should print information during execution
    verbose = False

    # Choose if explicit scheme should be used or not
    explicit = False

    # Check for error (needs f_ex)
    convergence = True

    # Parameters for the plot
    nb_plots = 2
    movie_duration = 0

    method = bracket
    if explicit:
        method += '_exp'
    else:
        method += '_imp'

    if convergence:
        method += "_conv"

    plot_dir = './plots/polar_'+example+'_' + method + '_'+bc + '/'
    if not os.path.exists(plot_dir):
        if verbose:
            print('creating directory ' + plot_dir)
        os.makedirs(plot_dir)

    # If plots should be shown or just saved
    show_plots = False

    if not convergence:
        solve_Arakawa_advection(example, bc, N_theta, N_r, T, Nt, domain_theta, domain_r,
                                nb_plots, movie_duration, bracket, verbose,
                                explicit, show_plots, plot_dir, convergence)

    else:
        # Number of grid points for each variable
        N_theta = [50, 100, 150]
        N_r = [50, 100, 150]

        # Number of time steps, total time, and time stepping
        Nt = [100, 100, 100, 100]
        T = 1

        err_f = []
        dofs = []

        for k in range(len(N_theta)):

            if not os.path.exists(plot_dir+'{}/'.format(k)):
                os.makedirs(plot_dir+'{}/'.format(k))

            dof, err = solve_Arakawa_advection(example, bc, N_theta[k], N_r[k], T, Nt[k], domain_theta, domain_r,
                                               nb_plots, movie_duration, bracket, False,
                                               explicit, False, plot_dir+'{}/'.format(k), True)

            dofs.append(dof)
            err_f.append(err)
            print(err)

        fig, ax = plt.subplots()
        ax.loglog()
        # ax.set_title("error")
        ax.plot(dofs, err_f, label="error")
        ax.set_title("Translation with Arakawa in polar Coordinates")
        ax.set_xlabel("grid size")
        ax.set_ylabel("relative error")
        ax.plot(dofs, [(7/dofs[i])**1 for i in range(len(dofs))],
                label="order 1", linestyle='--')
        ax.plot(dofs, [(40/dofs[i])**2 for i in range(len(dofs))],
                label="order 2", linestyle='--')
        ax.legend(loc='upper right')
        plt.savefig(plot_dir + 'error.png', dpi=300)


if __name__ == '__main__':
    main()
