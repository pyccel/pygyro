from mpi4py                 import MPI
import time
setup_time_start = time.clock()

from glob                   import glob

import numpy                as np
import argparse
import os
import h5py
#~ import cProfile, pstats, io

from pygyro.model.layout                    import LayoutSwapper, getLayoutHandler
from pygyro.model.grid                      import Grid
from pygyro.initialisation.setups           import setupCylindricalGrid, setupFromFile
from pygyro.advection.advection             import FluxSurfaceAdvection, VParallelAdvection, PoloidalAdvection, ParallelGradient
from pygyro.poisson.poisson_solver          import DensityFinder, QuasiNeutralitySolver
from pygyro.splines.splines                 import Spline2D
from pygyro.splines.spline_interpolators    import SplineInterpolator2D
from pygyro.utilities.savingTools           import setupSave
from l2Norm                                 import l2

loop_start = 0
loop_time = 0
diagnostic_start = 0
diagnostic_time = 0
output_start = 0
output_time = 0

parser = argparse.ArgumentParser(description='Process foldername')
parser.add_argument('foldername', metavar='foldername',nargs=1,type=str,
                    default=[""],
                   help='the name of the folder from which to load and in which to save')
parser.add_argument('tEnd', metavar='tEnd',nargs=1,type=int,
                   help='end time')
parser.add_argument('-r', dest='rDegree',nargs=1,type=int,
                    default=[3],
                   help='Degree of spline in r')
parser.add_argument('-q', dest='qDegree',nargs=1,type=int,
                    default=[3],
                   help='Degree of spline in theta')
parser.add_argument('-z', dest='zDegree',nargs=1,type=int,
                    default=[3],
                   help='Degree of spline in z')
parser.add_argument('-v', dest='vDegree',nargs=1,type=int,
                    default=[3],
                   help='Degree of spline in v')
parser.add_argument('-s', dest='saveStep',nargs=1,type=int,
                    default=[5],
                   help='Number of time steps between writing output')

args = parser.parse_args()
foldername = args.foldername[0]

assert(len(foldername)>0)

loadable = False

rDegree = args.rDegree[0]
qDegree = args.qDegree[0]
zDegree = args.zDegree[0]
vDegree = args.vDegree[0]

saveStep = args.saveStep[0]

tEnd = args.tEnd[0]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

filename = "{0}/initParams.h5".format(foldername)
save_file = h5py.File(filename,'r',driver='mpio',comm=comm)
group = save_file['constants']

npts = save_file.attrs['npts']
dt = save_file.attrs['dt']

halfStep = dt*0.5

save_file.close()
    
distribFunc = setupFromFile(foldername,comm=comm,
                            allocateSaveMemory = True,
                            timepoint = tEnd)

distribFunc.setLayout('poloidal')

nv = distribFunc.eta_grid[3].size

if (nv//2 in distribFunc.getGlobalIdxVals(0) and 0 in distribFunc.getGlobalIdxVals(1)):
    filename = "{0}/Slice_{1}.h5".format(foldername,tEnd)
    file = h5py.File(filename,'w')
    dset = file.create_dataset("dset",[distribFunc.eta_grid[1].size,distribFunc.eta_grid[0].size])
    i = nv//2 - distribFunc.getLayout(distribFunc.currentLayout).starts[0]
    dset[:]=distribFunc.get2DSlice([i,0])
    file.close()



