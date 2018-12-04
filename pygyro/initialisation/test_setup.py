from mpi4py import MPI
import numpy as np
from functools import reduce
import pytest

from  .constants              import Constants
from  .setups                 import setupCylindricalGrid, setupFromFile

@pytest.mark.parallel
def test_setup():
    npts = [10,20,10,10]
    grid,constants = setupCylindricalGrid(npts   = npts,
                                layout = 'flux_surface')
    
    for (coord,npt) in zip(grid.eta_grid,npts):
        assert(len(coord)==npt)
    
    grid,constants = setupCylindricalGrid(npts   = npts,
                                layout = 'poloidal')
    
    for (coord,npt) in zip(grid.eta_grid,npts):
        assert(len(coord)==npt)
    
    grid,constants = setupCylindricalGrid(npts   = npts,
                                layout = 'v_parallel')
    
    for (coord,npt) in zip(grid.eta_grid,npts):
        assert(len(coord)==npt)

import os

from ..model.grid          import Grid
from ..model.layout        import getLayoutHandler
from ..model.process_grid  import compute_2d_process_grid, compute_2d_process_grid_from_max
from ..utilities.savingTools import setupSave

def define_f(grid):
    [nEta1,nEta2,nEta3,nEta4] = grid.nGlobalCoords
    
    for i,x in grid.getCoords(0):
        for j,y in grid.getCoords(1):
            for k,z in grid.getCoords(2):
                Slice = grid.get1DSlice([i,j,k])
                for l,a in enumerate(Slice):
                    [I,J,K,L] = grid.getGlobalIndices([i,j,k,l])
                    Slice[l] = I*nEta4*nEta3*nEta2+J*nEta4*nEta3+K*nEta4+L

def compare_f(grid,t):
    [nEta1,nEta2,nEta3,nEta4] = grid.nGlobalCoords
    
    for i,x in grid.getCoords(0):
        for j,y in grid.getCoords(1):
            for k,z in grid.getCoords(2):
                Slice = grid.get1DSlice([i,j,k])
                for l,a in enumerate(Slice):
                    [I,J,K,L] = grid.getGlobalIndices([i,j,k,l])
                    assert(a == I*nEta4*nEta3*nEta2+J*nEta4*nEta3+K*nEta4+L+t)

@pytest.mark.parallel
def test_setupFromFolder():
    comm = MPI.COMM_WORLD
    npts = [10,20,10,10]
    nprocs = compute_2d_process_grid( npts , comm.Get_size() )
    
    eta_grids=[np.linspace(0,1,npts[0]),
               np.linspace(0,6.28318531,npts[1]),
               np.linspace(0,10,npts[2]),
               np.linspace(0,10,npts[3])]
    
    layouts = {'flux_surface': [0,3,1,2],
               'v_parallel'  : [0,2,1,3],
               'poloidal'    : [3,2,1,0]}
    remapper = getLayoutHandler( comm, layouts, nprocs, eta_grids )
    
    constants = Constants()
    constants.npts = npts
    
    grid = Grid(eta_grids,[],remapper,'flux_surface')
    
    define_f(grid)
    
    if (comm.Get_rank()==0):
        if (not os.path.isdir('testValues')):
            os.mkdir('testValues')
    
    setupSave(constants,'testValues')
    
    grid.writeH5Dataset('testValues',0)
    grid._f[:]+=20
    grid.writeH5Dataset('testValues',20)
    grid._f[:]+=20
    grid.writeH5Dataset('testValues',40)
    grid._f[:]+=20
    grid.writeH5Dataset('testValues',60)
    grid._f[:]+=20
    grid.writeH5Dataset('testValues',80)
    grid._f[:]+=20
    grid.writeH5Dataset('testValues',100)
    
    grid2,constants = setupFromFile('testValues')
    compare_f(grid2,100)

@pytest.mark.parallel
def test_setupFromFolderAtTime():
    comm = MPI.COMM_WORLD
    npts = [10,20,10,10]
    nprocs = compute_2d_process_grid( npts , comm.Get_size() )
    
    eta_grids=[np.linspace(0,1,npts[0]),
               np.linspace(0,6.28318531,npts[1]),
               np.linspace(0,10,npts[2]),
               np.linspace(0,10,npts[3])]
    
    layouts = {'flux_surface': [0,3,1,2],
               'v_parallel'  : [0,2,1,3],
               'poloidal'    : [3,2,1,0]}
    remapper = getLayoutHandler( comm, layouts, nprocs, eta_grids )
    
    constants = Constants()
    constants.npts = npts
    
    grid = Grid(eta_grids,[],remapper,'flux_surface')
    
    define_f(grid)
    
    if (comm.Get_rank()==0):
        if (not os.path.isdir('testValues')):
            os.mkdir('testValues')
    
    setupSave(constants,'testValues')
    
    grid.writeH5Dataset('testValues',0)
    grid._f[:]+=20
    grid.writeH5Dataset('testValues',20)
    grid._f[:]+=20
    grid.writeH5Dataset('testValues',40)
    grid._f[:]+=20
    grid.writeH5Dataset('testValues',60)
    grid._f[:]+=20
    grid.writeH5Dataset('testValues',80)
    grid._f[:]+=20
    grid.writeH5Dataset('testValues',100)
    
    grid2,constants = setupFromFile('testValues',timepoint = 40)
    compare_f(grid2,40)
