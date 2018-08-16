from mpi4py import MPI
import numpy as np
from functools import reduce
import pytest

from  .setups                 import setupCylindricalGrid, setupFromFile

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

def compare_f(grid):
    [nEta1,nEta2,nEta3,nEta4] = grid.nGlobalCoords
    
    for i,x in grid.getCoords(0):
        for j,y in grid.getCoords(1):
            for k,z in grid.getCoords(2):
                Slice = grid.get1DSlice([i,j,k])
                for l,a in enumerate(Slice):
                    [I,J,K,L] = grid.getGlobalIndices([i,j,k,l])
                    assert(a == I*nEta4*nEta3*nEta2+J*nEta4*nEta3+K*nEta4+L)

@pytest.mark.parallel
def test_setupFromFolder():
    
    grid = setupFromFile('__test__')
    compare_f(grid)

@pytest.mark.parallel
def test_setupFromFolderAtTime():
    
    grid = setupFromFile('__test__',timepoint = 40)
    compare_f(grid)
