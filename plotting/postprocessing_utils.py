import os
import h5py
import numpy as np
import ffmpeg
import matplotlib.pyplot as plt
from matplotlib import rc as pltFont

from pygyro.initialisation.constants import get_constants
from pygyro import splines as spl
from pygyro.tools.getSlice import get_grid_slice, get_flux_surface_grid_slice
from pygyro.tools.getPhiSlice import get_phi_slice
from pygyro.arakawa.utilities import compute_int_f, compute_int_f_squared, get_total_energy


def unpack_all(foldername):
    """
    Unpacks all Poloidal slices from the grid_xxxxxx.h5 and phi_xxxxxx.h5 
    files in the given folder. For v and z we use the default parameters 
    nv//2 and 0, as used in plot_poloidal_slice.py.

    Parameters
    ----------
    foldername  : str
                The folder containing the simulation
    """
    filelist = os.listdir(foldername)
    ind_f = []
    ind_phi = []
    for f in filelist:
        isgrid = f.find("grid_")
        if isgrid != -1:
            f_n = int(f[5:-3])
            get_grid_slice(foldername, f_n, "Slices_f")
            get_flux_surface_grid_slice(foldername, f_n, "FluxSlices_f")

        isphi = f.find("phi_")
        if isphi != -1:
            phi_n = int(f[4:-3])
            get_phi_slice(foldername, phi_n, "Slices_phi")


def plot_all_slices(foldername):
    """
    Given a folder of poloidal slices, we plot all of them in a subfolder called plots/. 

    Parameters
    ----------
    foldername  : str
                The folder containing the Poloidal Slices

    """

    if (not os.path.isdir(foldername+"plots/")):
        os.mkdir(foldername+"plots/")

    filelist = os.listdir(foldername)
    for fname in filelist:
        if fname != "plots":
            filename = foldername+fname
            t_str = filename[filename.rfind('_')+1:filename.rfind('.')]
            t = int(t_str)

            file = h5py.File(filename, 'r')
            dataset = file['/dset']

            shape = dataset.shape
            data_shape = list(shape)
            data_shape[0] += 1

            data = np.ndarray(data_shape)

            data[:-1, :] = dataset[:]
            data[-1, :] = data[0, :]

            file.close()
            if np.isnan(data).any() or np.isinf(data).any():
                print("NaN or Inf encounter")
            else:
                superfolder = foldername[0:foldername.find('/')]
                constantFile = os.path.join(superfolder, 'initParams.json')
                if not os.path.exists(constantFile):
                    raise RuntimeError(
                        "Can't find constants in simulation folder")

                constants = get_constants(constantFile)

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
                fig, ax = plt.subplots(1)
                ax.set_title('T = {}'.format(t))
                clevels = np.linspace(data.min(), data.max(), 101)
                im = ax.contourf(x, y, data, clevels, cmap='jet')
                for c in im.collections:
                    c.set_edgecolor('face')
                plt.colorbar(im)

                ax.set_xlabel("x [m]")
                ax.set_ylabel("y [m]")
                plt.savefig(foldername+'plots/t_{:06}'.format(t))
                plt.close()
                # plt.show()


def make_movie(foldername):
    """
    Given a folder with some frames called t_xxxxx.png, 
    we create a movie.

    Parameters
    ----------
    foldername : str
                The folder containing the plotted frames
    """
    stream = ffmpeg.input(foldername+"/t_*.png",
                          pattern_type="glob", framerate=30)
    stream = ffmpeg.output(stream, foldername+'/movie.mp4')
    ffmpeg.run(stream)


def plot_conservation(simulationfolder):
    """
    Calculates and plots the conserved quantities 
    (integral of f, integral of f^2 and energy) 
    of a simulation at all time-steps. 

    Parameters
    ----------
    simulationfolder  : str
                        The folder containing the simulation
    """
    foldername = simulationfolder+"Slices_f/"

    if (not os.path.isdir(simulationfolder+"plots/")):
        os.mkdir(simulationfolder+"plots/")

    filelist = os.listdir(foldername)
    n = len(filelist)

    t_f = []
    int_f = []
    int_f2 = []
    int_en = []

    for fname in filelist:
        if fname != "plots":
            filename = foldername+fname
            t_str = filename[filename.rfind('_')+1:filename.rfind('.')]
            t = int(t_str)
            t_f.append(t)

            file = h5py.File(filename, 'r')
            dataset = file['/dset']

            shape = dataset.shape
            data_shape = list(shape)
            data_shape[0] += 1

            f = np.ndarray(data_shape)

            f[:-1, :] = dataset[:]
            f[-1, :] = f[0, :]

            file.close()

            filename_phi = simulationfolder + \
                "Slice_phi/"+"PhiSlice_{:06}.h5".format(t)
            file = h5py.File(filename, 'r')
            dataset = file['/dset']

            shape = dataset.shape
            data_shape = list(shape)
            data_shape[0] += 1

            phi = np.ndarray(data_shape)

            phi[:-1, :] = dataset[:]
            phi[-1, :] = phi[0, :]

            file.close()

            superfolder = foldername[0:foldername.find('/')]
            constantFile = os.path.join(superfolder, 'initParams.json')
            if not os.path.exists(constantFile):
                raise RuntimeError("Can't find constants in simulation folder")

            constants = get_constants(constantFile)

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
            int_en.append(get_total_energy(
                f.ravel(), phi.ravel(), dtheta, dr, r))

    p = np.array(t_f).argsort()
    t_f = np.array(t_f)[p]
    int_f = np.array(int_f)[p]
    int_f2 = np.array(int_f2)[p]
    int_en = np.array(int_en)[p]

    fig, ax = plt.subplots(1)
    ax.set_title('Integral of $f$')
    ax.set_xlabel("t")
    ax.set_ylabel("$ \int f$")
    ax.plot(t_f, int_f)
    plt.savefig(simulationfolder+'plots/integral_f.png')
    plt.close()

    fig, ax = plt.subplots(1)
    ax.set_title('Integral of $f^2$')
    ax.set_xlabel("t")
    ax.set_ylabel("$ \int f^2$")
    ax.plot(t_f, int_f2)
    plt.savefig(simulationfolder+'plots/integral_f_squared.png')
    plt.close()

    fig, ax = plt.subplots(1)
    ax.set_title('Energy')
    ax.set_xlabel("t")
    ax.set_ylabel("$ \int \phi f$")
    ax.plot(t_f, int_en)
    plt.savefig(simulationfolder+'plots/energy.png')
    plt.close()


if __name__ == "__main__":
    foldername = "simulation_1/"

    unpack_all(foldername)
    plot_all_slices(foldername+"Slices_f/")
    plot_all_slices(foldername+"Slices_phi/")
    make_movie(foldername+"Slices_f/plots")
    make_movie(foldername+"Slices_phi/plots")
    plot_conservation(foldername)
