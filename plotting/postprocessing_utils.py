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


def unpack_all(foldername):
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


def plot_all(foldername):

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


def make_movie(foldername, name):
    stream = ffmpeg.input(foldername+"/t_*.png",
                          pattern_type="glob", framerate=30)
    stream = ffmpeg.output(stream, name+'.mp4')
    ffmpeg.run(stream)


if __name__ == "__main__":
    foldername = "simulation_0/"

    unpack_all(foldername)
    plot_all(foldername+"Slices_f/")
    plot_all(foldername+"Slices_phi/")
    make_movie(foldername+"Slices_f/plots", foldername+"Slices_f/plots/movie")
    make_movie(foldername+"Slices_phi/plots",
               foldername+"Slices_phi/plots/movie")
