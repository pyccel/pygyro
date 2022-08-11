from mpi4py import MPI
from math import pi
import numpy as np
import h5py
import argparse

from pygyro.initialisation.constants import get_constants
from pygyro import splines as spl
from pygyro.model.process_grid import compute_2d_process_grid
from pygyro.model.layout import getLayoutHandler


parser = argparse.ArgumentParser(description='Process foldername')
parser.add_argument('initF', metavar='initF', nargs=1, type=str,
                    help='init file')
parser.add_argument('f1', metavar='f1', nargs=1, type=str,
                    help='file1')
parser.add_argument('f2', metavar='f2', nargs=1, type=str,
                    help='file2')

args = parser.parse_args()
init_filename = args.initF[0]
filename1 = args.f1[0]
filename2 = args.f2[0]

constants = get_constants(init_filename)


comm = MPI.COMM_WORLD

allocateSaveMemory = False
dtype = float

rank = comm.Get_rank()

layout_comm = comm

mpi_size = layout_comm.Get_size()


npts = constants.npts

domain = [[constants.rMin, constants.rMax], [0, 2*pi],
          [constants.zMin, constants.zMax], [constants.vMin, constants.vMax]]
degree = constants.splineDegrees
period = [False, True, True, False]

# Compute breakpoints, knots, spline space and grid points
nkts = [n+1+d*(int(p)-1) for (n, d, p) in zip(npts, degree, period)]
breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
knots = [spl.make_knots(b, d, p) for (b, d, p) in zip(breaks, degree, period)]
bsplines = [spl.BSplines(k, d, p) for (k, d, p) in zip(knots, degree, period)]
eta_grids = [bspl.greville for bspl in bsplines]

# Compute 2D grid of processes for the two distributed dimensions in each layout
nprocs = compute_2d_process_grid(npts, mpi_size)

# Create dictionary describing layouts
layouts = {'flux_surface': [0, 3, 1, 2],
           'v_parallel': [0, 2, 1, 3],
           'poloidal': [3, 2, 1, 0]}

remapper = getLayoutHandler(layout_comm, layouts, nprocs, eta_grids)

layout = remapper.getLayout('v_parallel')

file = h5py.File(filename1, 'r')
dataset = file['/dset']
order1 = np.array(dataset.attrs['Layout'])

slices = tuple([slice(s, e) for s, e in zip(layout.starts, layout.ends)])

data1 = np.empty(layout.size, dtype=dtype).reshape(layout.shape)
data1[:] = dataset[slices]
file.close()


file = h5py.File(filename2, 'r')
dataset = file['/dset']
order2 = np.array(dataset.attrs['Layout'])


assert ((order1 == order2).all())

data2 = np.empty_like(data1)
data2[:] = dataset[slices]
file.close()


if (rank == 0):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.25, 0.7, 0.7],)
    colorbarax2 = fig.add_axes([0.85, 0.1, 0.03, 0.8],)

    line1 = ax.pcolormesh(
        eta_grids[3], eta_grids[1], (data1-data2)[0, 0, :, :])
    fig.canvas.draw()
    fig.canvas.flush_events()

    fig.colorbar(line1, cax=colorbarax2)

    plt.show()
