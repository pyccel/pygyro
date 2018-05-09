from mpi4py import MPI
import numpy as np
from functools import reduce
import pytest

from  .setups                 import setupCylindricalGrid

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
