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

def get_grid_slice(foldername,tEnd):

    assert(len(foldername)>0)

    loadable = False

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
        
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



