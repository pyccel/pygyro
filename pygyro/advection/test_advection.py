import pytest

from ..initialisation.setups    import setupCylindricalGrid
from .advection                 import *

@pytest.mark.serial
def test_fluxSurfaceAdvection():
    pass

@pytest.mark.serial
def test_vParallelAdvection():
    pass

@pytest.mark.serial
def test_poloidalAdvection():
    pass

@pytest.mark.serial
def test_fluxSurfaceAdvection_gridIntegration():
    npts = [10,20,10,10]
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'flux_surface')
    
    splines = grid.get2DSpline()
    
    for i,r in grid.getCoords(0):
        for j,v in grid.getCoords(1):
            fluxSurfaceAdv(grid.get2DSlice([i,j]),splines)

@pytest.mark.serial
def test_vParallelAdvection_gridIntegration():
    npts = [10,20,10,10]
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'v_parallel')
    
    spline = grid.get1DSpline()
    
    old_f=grid._f.copy()
    
    for i,r in grid.getCoords(0):
        for j,z in grid.getCoords(1):
            for k,q in grid.getCoords(2):
                vParallelAdv(grid.get1DSlice([i,j,k]),grid.getCoordVals(3),spline,0.1,0)
    
    assert(np.allclose(old_f,grid._f))

@pytest.mark.serial
def test_poloidalAdvection_gridIntegration():
    npts = [10,20,10,10]
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'poloidal')
    
    splines = grid.get2DSpline()
    
    for i,z in grid.getCoords(0):
        for j,v in grid.getCoords(1):
            poloidalAdv(grid.get2DSlice([i,j]))
