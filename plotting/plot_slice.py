import argparse
import os
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')

from pygyro.initialisation.setups import setupFromFile


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

    for k in range(11):
        if len(str(k)) == 1:
            t = '0' + str(k) + '000'
        elif len(str(k)) == 2:
            t = str(k) + '000'
        else:
            raise ValueError('some naming gone wrong')

        filename = os.path.join(foldername, 'grid_00' + t)

        distribFunc, _, _ = setupFromFile(foldername, comm=comm,
                                          allocateSaveMemory=True,
                                          timepoint=k*1000)

        distribFunc.setLayout('poloidal')

        f_min = np.min(distribFunc._f)
        f_max = np.max(distribFunc._f)

        print(f'min = {f_min} ; max = {f_max}')

        fig = plt.figure(figsize=(9.5, 7), dpi=250)
        ax = plt.subplot(111, projection='polar')
        ax.set_title("poloidalAdvectionArakawa_vortex")

        colorbarax2 = fig.add_axes([0.85, 0.1, 0.03, 0.8],)

        plotParams = {'vmin': f_min, 'vmax': f_max, 'cmap': "jet"}

        r_grid = distribFunc.eta_grid[0]
        theta_grid = distribFunc.eta_grid[1]
        theta_grid = np.append(theta_grid, 2*np.pi)
        v_grid = distribFunc.eta_grid[3]

        min_loc = np.argmin(np.abs(v_grid))

        # plot_f first axis is theta, second axis is r
        plot_f = np.empty((np.shape(distribFunc._f)[-2],
                           np.shape(distribFunc._f)[-1]))

        plot_f[:, :] = distribFunc._f[min_loc, 0, :, :]

        plot_f = np.append(plot_f, plot_f[0, :]).reshape((-1, plot_f.shape[1]))

        print(plot_f.shape)

        line1 = ax.contourf(theta_grid, r_grid, plot_f.T,
                            20, **plotParams)

        fig.canvas.draw()
        fig.colorbar(line1, cax=colorbarax2)

        plt.savefig(os.path.join(os.path.dirname(filename), 'grid_' + t + '.png'))


if __name__ == '__main__':
    main()
