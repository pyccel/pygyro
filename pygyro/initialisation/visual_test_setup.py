from mpi4py import MPI
import pytest

from  .setups                 import setupCylindricalGrid
from  .initialiser            import fEq, perturbation
from ..utilities.grid_plotter import SlicePlotter4d, SlicePlotter3d, Plotter2d

@pytest.mark.serial
def test_Perturbation_FluxSurface():
    npts = [10,20,10,10]
    m = 15
    n = 20
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'flux_surface',
                                m      = m,
                                n      = n)
    for i,r in grid.getCoords(0):
        for j,v in grid.getCoords(1):
            # Get surface
            FluxSurface = grid.get2DSlice([i,j])
            # Get coordinate values
            theta = grid.getCoordVals(2)
            z = grid.getCoordVals(3)
            
            # transpose theta to use ufuncs
            theta = theta.reshape(theta.size,1)
            FluxSurface[:]=perturbation(r,theta,z,m,n)
    
    Plotter2d(grid,'r','v').show()

@pytest.mark.serial
def test_Perturbation_vPar():
    npts = [10,20,10,10]
    m = 15
    n = 20
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'v_parallel',
                                m      = m,
                                n      = n)
    for i,r in grid.getCoords(0):
        for j,z in grid.getCoords(1):
            # Get surface
            Surface = grid.get2DSlice([i,j])
            # Get coordinate values
            theta = grid.getCoordVals(2)
            v = grid.getCoordVals(3)
            
            # transpose theta to use ufuncs
            theta = theta.reshape(theta.size,1)
            Surface[:]=perturbation(r,theta,z,m,n)
    
    Plotter2d(grid,'r','v').show()

@pytest.mark.serial
def test_Perturbation_Poloidal():
    npts = [10,20,10,10]
    m = 15
    n = 20
    grid = setupCylindricalGrid(nr     = npts[0],
                                ntheta = npts[1],
                                nz     = npts[2],
                                nv     = npts[3],
                                layout = 'poloidal',
                                m      = m,
                                n      = n)
    for i,v in grid.getCoords(0):
        for j,z in grid.getCoords(1):
            # Get surface
            PoloidalSurface = grid.get2DSlice([i,j])
            # Get coordinate values
            theta = grid.getCoordVals(2)
            r = grid.getCoordVals(3)
            
            # transpose theta to use ufuncs
            theta = theta.reshape(theta.size,1)
            PoloidalSurface[:]=perturbation(r,theta,z,m,n)
    
    Plotter2d(grid,'r','v').show()

@pytest.mark.serial
def test_FieldPlot_FluxSurface():
    npts = [10,20,10,10]
    m = 15
    n = 20
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'flux_surface',
                                m      = m,
                                n      = n)
    for i,r in grid.getCoords(0):
        for j,v in grid.getCoords(1):
            # Get surface
            FluxSurface = grid.get2DSlice([i,j])
            # Get coordinate values
            theta = grid.getCoordVals(2)
            z = grid.getCoordVals(3)
            
            # transpose theta to use ufuncs
            theta = theta.reshape(theta.size,1)
            FluxSurface[:]=perturbation(r,theta,z,m,n)
    
    Plotter2d(grid,'q','z').show()

@pytest.mark.serial
def test_FieldPlot_vPar():
    npts = [10,20,10,10]
    m = 15
    n = 20
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'v_parallel',
                                m      = m,
                                n      = n)
    for i,r in grid.getCoords(0):
        for j,z in grid.getCoords(1):
            # Get surface
            Surface = grid.get2DSlice([i,j])
            # Get coordinate values
            theta = grid.getCoordVals(2)
            v = grid.getCoordVals(3)
            
            # transpose theta to use ufuncs
            theta = theta.reshape(theta.size,1)
            Surface[:]=perturbation(r,theta,z,m,n)
    
    Plotter2d(grid,'q','z').show()

@pytest.mark.serial
def test_FieldPlot_Poloidal():
    npts = [10,20,10,10]
    m = 15
    n = 20
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'poloidal',
                                m      = m,
                                n      = n)
    for i,v in grid.getCoords(0):
        for j,z in grid.getCoords(1):
            # Get surface
            PoloidalSurface = grid.get2DSlice([i,j])
            # Get coordinate values
            theta = grid.getCoordVals(2)
            r = grid.getCoordVals(3)
            
            # transpose theta to use ufuncs
            theta = theta.reshape(theta.size,1)
            PoloidalSurface[:]=perturbation(r,theta,z,m,n)
    
    Plotter2d(grid,'q','z').show()

@pytest.mark.serial
def test_Equilibrium_FluxSurface():
    npts = [10,20,10,10]
    m = 15
    n = 20
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'flux_surface',
                                m      = m,
                                n      = n)
    for i,r in grid.getCoords(0):
        for j,v in grid.getCoords(1):
            # Get surface
            FluxSurface = grid.get2DSlice([i,j])
            # Get coordinate values
            theta = grid.getCoordVals(2)
            z = grid.getCoordVals(3)
            
            # transpose theta to use ufuncs
            theta = theta.reshape(theta.size,1)
            FluxSurface[:]=fEq(r,theta,z,v,m,n)
    
    Plotter2d(grid,'r','v').show()

@pytest.mark.serial
def test_Equilibrium_vPar():
    npts = [10,20,10,10]
    m = 15
    n = 20
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'v_parallel',
                                m      = m,
                                n      = n)
    for i,r in grid.getCoords(0):
        for j,z in grid.getCoords(1):
            # Get surface
            Surface = grid.get2DSlice([i,j])
            # Get coordinate values
            theta = grid.getCoordVals(2)
            v = grid.getCoordVals(3)
            
            # transpose theta to use ufuncs
            theta = theta.reshape(theta.size,1)
            Surface[:]=fEq(r,theta,z,v,m,n)
    
    Plotter2d(grid,'r','v').show()

@pytest.mark.serial
def test_Equilibrium_Poloidal():
    npts = [10,20,10,10]
    m = 15
    n = 20
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'poloidal',
                                m      = m,
                                n      = n)
    for i,v in grid.getCoords(0):
        for j,z in grid.getCoords(1):
            # Get surface
            PoloidalSurface = grid.get2DSlice([i,j])
            # Get coordinate values
            theta = grid.getCoordVals(2)
            r = grid.getCoordVals(3)
            
            # transpose theta to use ufuncs
            theta = theta.reshape(theta.size,1)
            PoloidalSurface[:]=fEq(r,theta,z,v,m,n)
    
    Plotter2d(grid,'r','v').show()

@pytest.mark.serial
def test_FluxSurface():
    npts = [10,10,20,20]
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'flux_surface')
    SlicePlotter4d(grid).show()

@pytest.mark.serial
def test_vParallel():
    npts = [10,10,20,20]
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'v_parallel')
    SlicePlotter4d(grid).show()

@pytest.mark.serial
def test_Poloidal():
    npts = [10,10,20,20]
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'poloidal')
    SlicePlotter4d(grid).show()
