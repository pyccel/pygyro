import argparse
import os
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')

from pygyro.initialisation.setups import setupFromFile
from pygyro.initialisation.initialiser_funcs import make_f_eq_grid


def main():
    parser = argparse.ArgumentParser(
        description='Plot the 4D distribution function')
    parser.add_argument('foldername', nargs='?', type=str, default=None,
                        help='The folder whose results should be plotted.')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    foldername = args.foldername

    if foldername == None:
        raise ValueError('foldername must be given!')
    
    if not os.path.exists(os.path.join(foldername, 'plots')):
        os.mkdir(os.path.join(foldername, 'plots'))
    
    plot_path = os.path.join(foldername, 'plots')

    # ===========================
    # ===== Create Subplots =====
    # ===========================

    fig = plt.figure(figsize=(8, 9), dpi=250)
    for k in range(6):
        if len(str(k)) == 1:
            t = '0' + str(k) + '000'
        elif len(str(k)) == 2:
            t = str(k) + '000'
        else:
            raise ValueError('some naming gone wrong')

        distribFunc, constants, _ = setupFromFile(foldername, comm=comm,
                                          allocateSaveMemory=True,
                                          timepoint=k*1000)

        distribFunc.setLayout('poloidal')

        r_grid = distribFunc.eta_grid[0]
        theta_grid = distribFunc.eta_grid[1]
        theta_grid = np.append(theta_grid, 2*np.pi)
        v_grid = distribFunc.eta_grid[3]

        layout = distribFunc.getLayout('poloidal')

        idx_r = layout.inv_dims_order[0]
        idx_v = layout.inv_dims_order[3]

        # global grids
        r = distribFunc.eta_grid[0]
        v = distribFunc.eta_grid[3]

        # local grids
        my_r = r[layout.starts[idx_r]:layout.ends[idx_r]]
        my_v = v[layout.starts[idx_v]:layout.ends[idx_v]]
    
        # Make f_eq array (only depends on v and r but has to have full size)
        f_eq = np.zeros((my_v.size, my_r.size), dtype=float)
        make_f_eq_grid(constants.CN0, constants.kN0, constants.deltaRN0, constants.rp,
                    constants.CTi, constants.kTi, constants.deltaRTi, my_v, my_r, f_eq)
        shape_f_eq = [1, 1, 1, 1]
        shape_f_eq[idx_v] = my_v.size
        shape_f_eq[idx_r] = my_r.size
        f_eq.resize(shape_f_eq)

        min_loc = np.argmin(np.abs(v_grid))

        # plot_f first axis is theta, second axis is r
        plot_f = np.empty((np.shape(distribFunc._f)[-2],
                           np.shape(distribFunc._f)[-1]))

        # total distribution function
        plot_f[:, :] = distribFunc._f[min_loc, 0, :, :] * 0.01 + 0.4
        plot_f = np.append(plot_f, plot_f[0, :]).reshape((-1, plot_f.shape[1]))

        ax = plt.subplot(3, 2, k + 1, projection='polar')
        ax.set_title("T = " + t)

        colorbarax2 = fig.add_axes([0.85, 0.1, 0.03, 0.8],)

        f_min = np.min(plot_f)
        f_max = np.max(plot_f)

        plotParams = {'vmin': f_min, 'vmax': f_max, 'cmap': "jet"}

        clevels = np.linspace(plot_f.min(), plot_f.max(), 151)

        line1 = ax.contourf(theta_grid, r_grid, plot_f.T,
                            10, **plotParams)

    plt.subplots_adjust(left=0.00, right=0.9, top=0.92, bottom=0.05)
    plt.subplots_adjust(wspace=-0.2, hspace=0.4)
    fig.canvas.draw()
    fig.colorbar(line1, cax=colorbarax2)

    plt.savefig(os.path.join(plot_path, 'all_full.png'))
    plt.close()


    fig = plt.figure(figsize=(8, 9), dpi=250)
    for k in range(6):
        if len(str(k)) == 1:
            t = '0' + str(k) + '000'
        elif len(str(k)) == 2:
            t = str(k) + '000'
        else:
            raise ValueError('some naming gone wrong')

        distribFunc, constants, _ = setupFromFile(foldername, comm=comm,
                                          allocateSaveMemory=True,
                                          timepoint=k*1000)

        distribFunc.setLayout('poloidal')

        r_grid = distribFunc.eta_grid[0]
        theta_grid = distribFunc.eta_grid[1]
        theta_grid = np.append(theta_grid, 2*np.pi)
        v_grid = distribFunc.eta_grid[3]

        layout = distribFunc.getLayout('poloidal')

        idx_r = layout.inv_dims_order[0]
        idx_v = layout.inv_dims_order[3]

        # global grids
        r = distribFunc.eta_grid[0]
        v = distribFunc.eta_grid[3]

        # local grids
        my_r = r[layout.starts[idx_r]:layout.ends[idx_r]]
        my_v = v[layout.starts[idx_v]:layout.ends[idx_v]]
    
        # Make f_eq array (only depends on v and r but has to have full size)
        f_eq = np.zeros((my_v.size, my_r.size), dtype=float)
        make_f_eq_grid(constants.CN0, constants.kN0, constants.deltaRN0, constants.rp,
                    constants.CTi, constants.kTi, constants.deltaRTi, my_v, my_r, f_eq)
        shape_f_eq = [1, 1, 1, 1]
        shape_f_eq[idx_v] = my_v.size
        shape_f_eq[idx_r] = my_r.size
        f_eq.resize(shape_f_eq)

        min_loc = np.argmin(np.abs(v_grid))

        # plot_f first axis is theta, second axis is r
        plot_f = np.empty((np.shape(distribFunc._f)[-2],
                           np.shape(distribFunc._f)[-1]))
        plot_f[:, :] = (distribFunc._f - f_eq)[min_loc, 0, :, :]
        plot_f = np.append(plot_f, plot_f[0, :]).reshape((-1, plot_f.shape[1]))

        ax = plt.subplot(3, 2, k + 1, projection='polar')
        ax.set_title("T = " + t)

        colorbarax2 = fig.add_axes([0.85, 0.1, 0.03, 0.8],)

        f_min = np.min(plot_f)
        f_max = np.max(plot_f)

        clevels = np.linspace(f_min, f_max, 151)

        line1 = ax.contourf(theta_grid, r_grid, plot_f.T,
                            clevels, cmap='seismic')

    plt.subplots_adjust(left=0.00, right=0.9, top=0.92, bottom=0.05)
    plt.subplots_adjust(wspace=-0.2, hspace=0.4)
    fig.canvas.draw()
    fig.colorbar(line1, cax=colorbarax2)

    plt.savefig(os.path.join(plot_path, 'all_diff.png'))
    plt.close()


    exit()

    # ===============================
    # ===== Create Single Plots =====
    # ===============================

    for k in range(5):
        if len(str(k)) == 1:
            t = '0' + str(k) + '000'
        elif len(str(k)) == 2:
            t = str(k) + '000'
        else:
            raise ValueError('some naming gone wrong')

        distribFunc, constants, _ = setupFromFile(foldername, comm=comm,
                                          allocateSaveMemory=True,
                                          timepoint=k*1000)

        distribFunc.setLayout('poloidal')

        r_grid = distribFunc.eta_grid[0]
        theta_grid = distribFunc.eta_grid[1]
        theta_grid = np.append(theta_grid, 2*np.pi)
        v_grid = distribFunc.eta_grid[3]

        layout = distribFunc.getLayout('poloidal')

        idx_r = layout.inv_dims_order[0]
        idx_v = layout.inv_dims_order[3]

        # global grids
        r = distribFunc.eta_grid[0]
        v = distribFunc.eta_grid[3]

        # local grids
        my_r = r[layout.starts[idx_r]:layout.ends[idx_r]]
        my_v = v[layout.starts[idx_v]:layout.ends[idx_v]]
    
        # Make f_eq array (only depends on v and r but has to have full size)
        f_eq = np.zeros((my_v.size, my_r.size), dtype=float)
        make_f_eq_grid(constants.CN0, constants.kN0, constants.deltaRN0, constants.rp,
                    constants.CTi, constants.kTi, constants.deltaRTi, my_v, my_r, f_eq)
        shape_f_eq = [1, 1, 1, 1]
        shape_f_eq[idx_v] = my_v.size
        shape_f_eq[idx_r] = my_r.size
        f_eq.resize(shape_f_eq)

        min_loc = np.argmin(np.abs(v_grid))

        # plot_f first axis is theta, second axis is r
        plot_f = np.empty((np.shape(distribFunc._f)[-2],
                           np.shape(distribFunc._f)[-1]))


        # ========================
        # ===== Create Plots =====
        # ========================

        # total distribution function
        plot_f[:, :] = distribFunc._f[min_loc, 0, :, :] * 0.01 + 0.4
        plot_f = np.append(plot_f, plot_f[0, :]).reshape((-1, plot_f.shape[1]))

        fig = plt.figure(figsize=(9.5, 7), dpi=250)
        ax = plt.subplot(111, projection='polar')
        ax.set_title("T = " + t)

        colorbarax2 = fig.add_axes([0.85, 0.1, 0.03, 0.8],)

        f_min = np.min(plot_f)
        f_max = np.max(plot_f)

        plotParams = {'vmin': f_min, 'vmax': f_max, 'cmap': "jet"}

        clevels = np.linspace(plot_f.min(), plot_f.max(), 151)

        line1 = ax.contourf(theta_grid, r_grid, plot_f.T,
                            10, **plotParams)

        fig.canvas.draw()
        fig.colorbar(line1, cax=colorbarax2)

        plt.savefig(os.path.join(plot_path, 'grid_' + t + 'full.png'))
        plt.close()

        # with f_eq subtracted
        f_min = np.min(distribFunc._f)
        f_max = np.max(distribFunc._f)

        plot_f = np.empty((np.shape(distribFunc._f)[-2],
                           np.shape(distribFunc._f)[-1]))
        plot_f[:, :] = (distribFunc._f - f_eq)[min_loc, 0, :, :]
        plot_f = np.append(plot_f, plot_f[0, :]).reshape((-1, plot_f.shape[1]))

        fig = plt.figure(figsize=(9.5, 7), dpi=250)
        ax = plt.subplot(111, projection='polar')
        ax.set_title("T = " + t)

        colorbarax2 = fig.add_axes([0.85, 0.1, 0.03, 0.8],)

        clevels = np.linspace(plot_f.min(), plot_f.max(), 151)

        line1 = ax.contourf(theta_grid, r_grid, plot_f.T,
                            clevels, cmap='seismic')

        fig.canvas.draw()
        fig.colorbar(line1, cax=colorbarax2)

        plt.savefig(os.path.join(plot_path, 'grid_' + t + 'diff.png'))
        plt.close()


if __name__ == '__main__':
    main()
