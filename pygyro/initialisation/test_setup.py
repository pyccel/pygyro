from mpi4py import MPI
import numpy as np
from functools import reduce
import pytest
import h5py

from  .setups                 import setupCylindricalGrid, setupFromFile
from  .printInitialVals       import getParameters

@pytest.mark.serial
def test_setup():
    npts = [10,20,10,10]
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'flux_surface')
    
    for (coord,npt) in zip(grid.eta_grid,npts):
        assert(len(coord)==npt)
    
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'poloidal')
    
    for (coord,npt) in zip(grid.eta_grid,npts):
        assert(len(coord)==npt)
    
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'v_parallel')
    
    for (coord,npt) in zip(grid.eta_grid,npts):
        assert(len(coord)==npt)

@pytest.mark.parallel
def test_setup():
    npts = [10,20,10,10]
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'flux_surface')
    
    for (coord,npt) in zip(grid.eta_grid,npts):
        assert(len(coord)==npt)
    
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'poloidal')
    
    for (coord,npt) in zip(grid.eta_grid,npts):
        assert(len(coord)==npt)
    
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'v_parallel')
    
    for (coord,npt) in zip(grid.eta_grid,npts):
        assert(len(coord)==npt)

@pytest.mark.parallel
def test_printVals():
    npts = [10,20,10,10]
    
    test_file = h5py.File('test_init.h5','w',driver='mpio',comm=MPI.COMM_WORLD)
    getParameters(test_file,3,3,3,3,npts)
    test_file.close()
    
    test_file = h5py.File('test_init.h5','r',driver='mpio',comm=MPI.COMM_WORLD)
    grid = setupFromFile(test_file,layout = 'v_parallel')
    test_file.close()
