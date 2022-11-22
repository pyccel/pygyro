import os
import h5py
import numpy as np
from mpi4py import MPI

import ffmpeg
import matplotlib.pyplot as plt
from matplotlib import rc as pltFont

from pygyro.initialisation.setups import setupFromFile
from pygyro.initialisation.constants import get_constants
from pygyro import splines as spl
from pygyro.tools.getSlice import get_grid_slice, get_flux_surface_grid_slice
from pygyro.tools.getPhiSlice import get_phi_slice
from pygyro.arakawa.utilities import compute_int_f, compute_int_f_squared, get_potential_energy

from pygyro.model.process_grid import compute_2d_process_grid
from pygyro.model.grid import Grid
from pygyro.model.layout import LayoutSwapper


def get_data_from_4d_f(foldername, tEnd, z=None, v=None):

    comm = MPI.COMM_WORLD

    distribFunc, constants, _ = setupFromFile(foldername, comm=comm,
                                              allocateSaveMemory=True,
                                              timepoint=tEnd)

    distribFunc.setLayout('poloidal')

    if z is None:
        z = 0
    else:
        assert z_idx < distribFunc.eta_grid[2].size

    if v is None:
        nv = distribFunc.eta_grid[3].size
        v = nv//2
    else:
        assert v < distribFunc.eta_grid[3].size

    if (v in distribFunc.getGlobalIdxVals(0) and z in distribFunc.getGlobalIdxVals(1)):
        starts = distribFunc.getLayout(distribFunc.currentLayout).starts
        i = v - starts[0]
        j = z - starts[1]
        dataset = distribFunc.get2DSlice(i, j)

        shape = dataset.shape
        data_shape = list(shape)
        data_shape[0] += 1

        data = np.ndarray(data_shape)

        data[:-1, :] = dataset[:]
        data[-1, :] = data[0, :]

        return data, constants


def get_data_from_4d_phi(foldername, tEnd, z=None):
    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()

    filename = os.path.join(foldername, "initParams.json")
    constants = get_constants(filename)

    npts = constants.npts

    degree = constants.splineDegrees[:-1]
    period = [False, True, True]
    domain = [[constants.rMin, constants.rMax], [
        0, 2*np.pi], [constants.zMin, constants.zMax]]

    nkts = [n+1+d*(int(p)-1) for (n, d, p) in zip(npts, degree, period)]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, d, p)
             for (b, d, p) in zip(breaks, degree, period)]
    bsplines = [spl.BSplines(k, d, p)
                for (k, d, p) in zip(knots, degree, period)]
    eta_grids = [bspl.greville for bspl in bsplines]

    layout_poisson = {'v_parallel_2d': [0, 2, 1],
                      'mode_solve': [1, 2, 0]}
    layout_vpar = {'v_parallel_1d': [0, 2, 1]}
    layout_poloidal = {'poloidal': [2, 1, 0]}

    nprocs = compute_2d_process_grid(npts, mpi_size)

    remapperPhi = LayoutSwapper(comm, [layout_poisson, layout_vpar, layout_poloidal],
                                [nprocs, nprocs[0], nprocs[1]], eta_grids,
                                'v_parallel_2d')

    phi = Grid(eta_grids, bsplines, remapperPhi,
               'v_parallel_2d', comm, dtype=np.complex128)
    phi.loadFromFile(foldername, tEnd, "phi")

    phi.setLayout('poloidal')

    if z is None:
        z = 0
    else:
        assert z_idx < phi.eta_grid[2].size

    if (z in phi.getGlobalIdxVals(0)):

        starts = phi.getLayout(phi.currentLayout).starts
        dataset = np.real(phi.get2DSlice(z + starts[0]))

        shape = dataset.shape
        data_shape = list(shape)
        data_shape[0] += 1

        data = np.ndarray(data_shape)

        data[:-1, :] = dataset[:]
        data[-1, :] = data[0, :]

    return data, constants


def plot_f_slice(foldername, tEnd, z=None, v=None):

    if (not os.path.isdir(foldername+"plots_f/")):
        os.mkdir(foldername+"plots_f/")

    data, constants = get_data_from_4d_f(foldername, tEnd, z=None, v=None)

    if np.isnan(data).any() or np.isinf(data).any():
        print("NaN or Inf encounter")
    else:
        npts = constants.npts[:2]
        degree = constants.splineDegrees[:2]
        period = [False, True]
        domain = [[constants.rMin, constants.rMax], [0, 2*np.pi]]

        nkts = [n+1+d*(int(p)-1)
                for (n, d, p) in zip(npts, degree, period)]
        breaks = [np.linspace(*lims, num=num)
                  for (lims, num) in zip(domain, nkts)]
        knots = [spl.make_knots(b, d, p)
                 for (b, d, p) in zip(breaks, degree, period)]
        bsplines = [spl.BSplines(k, d, p)
                    for (k, d, p) in zip(knots, degree, period)]
        eta_grid = [bspl.greville for bspl in bsplines]

        theta = np.repeat(np.append(eta_grid[1], 2*np.pi), npts[0]) \
            .reshape(npts[1]+1, npts[0])
        r = np.tile(eta_grid[0], npts[1]+1) \
            .reshape(npts[1]+1, npts[0])

        x = r*np.cos(theta)
        y = r*np.sin(theta)

        font = {'size': 16}
        pltFont('font', **font)
        _, ax = plt.subplots(1)
        ax.set_title('T = {}'.format(tEnd))
        clevels = np.linspace(data.min(), data.max(), 101)
        im = ax.contourf(x, y, data, clevels, cmap='jet')
        for c in im.collections:
            c.set_edgecolor('face')
        plt.colorbar(im)

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        plt.savefig(foldername+'plots_f/t_{:06}'.format(tEnd))
        plt.close()


def plot_phi_slice(foldername, tEnd, z=None):
    if (not os.path.isdir(foldername+"plots_phi/")):
        os.mkdir(foldername+"plots_phi/")

    data, constants = get_data_from_4d_phi(foldername, tEnd, z=None)

    if np.isnan(data).any() or np.isinf(data).any():
        print("NaN or Inf encounter")
    else:

        npts = constants.npts[:2]
        degree = constants.splineDegrees[:2]
        period = [False, True]
        domain = [[constants.rMin, constants.rMax], [0, 2*np.pi]]

        nkts = [n+1+d*(int(p)-1)
                for (n, d, p) in zip(npts, degree, period)]
        breaks = [np.linspace(*lims, num=num)
                  for (lims, num) in zip(domain, nkts)]
        knots = [spl.make_knots(b, d, p)
                 for (b, d, p) in zip(breaks, degree, period)]
        bsplines = [spl.BSplines(k, d, p)
                    for (k, d, p) in zip(knots, degree, period)]
        eta_grid = [bspl.greville for bspl in bsplines]

        theta = np.repeat(np.append(eta_grid[1], 2*np.pi), npts[0]) \
            .reshape(npts[1]+1, npts[0])
        r = np.tile(eta_grid[0], npts[1]+1) \
            .reshape(npts[1]+1, npts[0])

        x = r*np.cos(theta)
        y = r*np.sin(theta)

        font = {'size': 16}
        pltFont('font', **font)
        _, ax = plt.subplots(1)
        ax.set_title('T = {}'.format(tEnd))
        clevels = np.linspace(data.min(), data.max(), 101)
        im = ax.contourf(x, y, data, clevels, cmap='jet')
        for c in im.collections:
            c.set_edgecolor('face')
        plt.colorbar(im)

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        plt.savefig(foldername+'plots_phi/t_{:06}'.format(tEnd))
        plt.close()


def plot_all_slices(foldername, z=None, v=None):
    """
    Given a folder of poloidal slices,
    we plot all of them in a subfolder called plots/.

    Parameters
    ----------
    foldername  : str
                The folder containing the Poloidal Slices

    """

    filelist = os.listdir(foldername)
    for f in filelist:
        isgrid = f.find("grid_")
        isphi = f.find("phi_")
        if isgrid != -1:
            t = int(f[-8:-3])
            plot_f_slice(foldername, t, z, v)

        elif isphi != -1:
            t = int(f[-7:-3])
            plot_phi_slice(foldername, t, z)


def make_movie(foldername):
    """
    Given a folder with some frames called t_xxxxx.png,
    we create a movie.

    Parameters
    ----------
    foldername : str
                The folder containing the plotted frames
    """
    stream = ffmpeg.input(foldername+"t_*.png",
                          pattern_type="glob", framerate=30)
    stream = ffmpeg.output(stream, foldername+'/movie.mp4')
    ffmpeg.run(stream)


def plot_conservation(foldername):
    """
    Calculates and plots the conserved quantities
    (integral of f, integral of f^2 and energy)
    of a simulation at all time-steps.

    Parameters
    ----------
    simulationfolder  : str
                        The folder containing the simulation
    """

    if (not os.path.isdir(foldername+"plots/")):
        os.mkdir(foldername+"plots/")

    filelist = os.listdir(foldername)

    t_f = []
    int_f = []
    int_f2 = []
    int_en = []

    for fname in filelist:
        if fname.find("grid_") != -1:
            #filename = foldername+fname
            t_str = fname[fname.rfind('_')+1:fname.rfind('.')]
            t = int(t_str)
            t_f.append(t)

            f, constants = get_data_from_4d_f(foldername, t, z=None, v=None)
            phi, _ = get_data_from_4d_phi(foldername, t, z=None)

            npts = constants.npts[:2]
            degree = constants.splineDegrees[:2]
            period = [False, True]
            domain = [[constants.rMin, constants.rMax], [0, 2*np.pi]]

            nkts = [n+1+d*(int(p)-1)
                    for (n, d, p) in zip(npts, degree, period)]
            breaks = [np.linspace(*lims, num=num)
                      for (lims, num) in zip(domain, nkts)]
            knots = [spl.make_knots(b, d, p)
                     for (b, d, p) in zip(breaks, degree, period)]
            bsplines = [spl.BSplines(k, d, p)
                        for (k, d, p) in zip(knots, degree, period)]
            eta_grid = [bspl.greville for bspl in bsplines]

            r = eta_grid[0]
            dr = eta_grid[0][1] - eta_grid[0][0]
            dtheta = eta_grid[1][1] - eta_grid[1][0]

            int_f.append(compute_int_f(f.ravel(), dtheta, dr, r))
            int_f2.append(compute_int_f_squared(f.ravel(), dtheta, dr, r))
            int_en.append(get_potential_energy(
                f.ravel(), phi.ravel(), dtheta, dr, r))

    p = np.array(t_f).argsort()
    t_f = np.array(t_f)[p]
    int_f = np.array(int_f)[p]
    int_f2 = np.array(int_f2)[p]
    int_en = np.array(int_en)[p]

    _, ax = plt.subplots(1)
    ax.set_title('Integral of $f$')
    ax.set_xlabel("t")
    ax.set_ylabel("$ \int f$")
    ax.plot(t_f, int_f)
    plt.savefig(foldername+'plots/integral_f.png')
    plt.close()

    _, ax = plt.subplots(1)
    ax.set_title('Integral of $f^2$')
    ax.set_xlabel("t")
    ax.set_ylabel("$ \int f^2$")
    ax.plot(t_f, int_f2)
    plt.savefig(foldername+'plots/integral_f_squared.png')
    plt.close()

    _, ax = plt.subplots(1)
    ax.set_title('Potential Energy')
    ax.set_xlabel("t")
    ax.set_ylabel("$ \int \phi f$")
    ax.plot(t_f, int_en)
    plt.savefig(foldername+'plots/energy.png')
    plt.close()


def plot_L2(foldername):
    """
    Plots the L2 norm of phi.

    Parameters
    ----------
    simulationfolder  : str
                        The folder containing the simulation
    """
    filename = os.path.join(foldername, 'phiDat.txt')

    p = 3.54e-3
    m = 4e-5

    dataset = np.atleast_2d(np.loadtxt(filename))
    sorted_times = np.sort(dataset[:, 0])

    plt.figure()
    plt.semilogy(sorted_times, m*np.exp(p*sorted_times),
                 label=str(m)+'*exp('+str(p)+'*x)')

    dataset = np.atleast_2d(np.loadtxt(os.path.join(foldername, 'phiDat.txt')))
    shape = dataset.shape
    times = np.ndarray(shape[0])
    norm = np.ndarray(shape[0])
    times[:] = dataset[:, 0]
    norm[:] = dataset[:, 1]
    plt.semilogy(times, norm, '.', label=foldername)

    plt.xlabel('time')
    plt.ylabel('$\|\phi\|_2$')
    plt.grid()
    plt.legend()
    plt.savefig(foldername+'plots/L2_phi.png')
    plt.close()


if __name__ == "__main__":
    foldername = "simulation_0/"

    plot_all_slices(foldername)
    make_movie(foldername+"plots_f/")
    make_movie(foldername+"plots_phi/")
    plot_conservation(foldername)
    plot_L2(foldername)
