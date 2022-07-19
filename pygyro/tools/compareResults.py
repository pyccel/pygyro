from mpi4py import MPI
from math import pi
import numpy as np
import h5py
import argparse

from pygyro.initialisation import constants
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
filename = args.initF[0]
filename1 = args.f1[0]
filename2 = args.f2[0]


comm = MPI.COMM_WORLD
print(filename)
save_file = h5py.File(filename, 'r', driver='mpio', comm=comm)
group = save_file['constants']
for i in group.attrs:
    constants.i = group.attrs[i]
constants.rp = 0.5*(constants.rMin + constants.rMax)

rMin = constants.rMin
rMax = constants.rMax
zMin = constants.zMin
zMax = constants.zMax
vMax = constants.vMax
vMin = constants.vMin
m = constants.m
n = constants.n
eps = constants.eps

group = save_file['degrees']
rDegree = int(group.attrs['r'])
qDegree = int(group.attrs['theta'])
zDegree = int(group.attrs['z'])
vDegree = int(group.attrs['v'])

npts = save_file.attrs['npts']

save_file.close()

allocateSaveMemory = False
dtype = float

rank = comm.Get_rank()

layout_comm = comm

mpi_size = layout_comm.Get_size()


domain = [[rMin, rMax], [0, 2*pi], [zMin, zMax], [vMin, vMax]]
degree = [rDegree, qDegree, zDegree, vDegree]
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


assert((order1 == order2).all())

data2 = np.empty_like(data1)
data2[:] = dataset[slices]
file.close()

err = comm.reduce(np.max(np.abs(data2-data1)), MPI.MAX, root=0)

if (rank == 0):
    print(filename1, filename2, err, file=open("profile/comparison.txt", "a"))
