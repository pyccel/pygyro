import pytest
import matplotlib.pyplot    as plt

from ..initialisation.setups        import setupCylindricalGrid
from ..initialisation.initialiser   import fEq
from .advection                     import *
from ..                     import splines as spl

@pytest.mark.serial
def test_fluxSurfaceAdvection():
    pass

def gauss(x):
    return np.exp(-x**2/2/0.5**2)

@pytest.mark.serial
@pytest.mark.parametrize( "function,N,periodic", [(gauss,10,False),(gauss,10,True),(gauss,20,False),(gauss,20,True),
                                                    (gauss,30,False),(gauss,30,True)] )
def test_vParallelAdvection(function,N,periodic):
    npts = 100
    f = np.empty(npts)
    
    dt=0.1
    c=2
    
    
    nkts      = npts+1+3*(int(periodic)-1)
    breaks    = np.linspace( -5, 5, num=nkts )
    knots     = spl.make_knots( breaks,3,periodic )
    spline    = spl.BSplines( knots,3,periodic )
    x         = spline.greville
    
    r = 4
    fEdge = fEq(4,x[0])
    
    f = function(x)+fEdge
    
    vParAdv = vParallelAdvection(x, spline)
    
    for i in range(N):
        vParAdv.step(f,dt,c,r)
    
    fEnd = np.empty(npts)
    
    for i in range(npts):
        if (x[i]-c*dt<x[0]):
            fEnd[i]=fEdge
        else:
            fEnd[i]=fEdge+function(x[i]-c*dt*N)
    
    assert(max(f-fEnd)<2e-5)

@pytest.mark.serial
def test_poloidalAdvection():
    pass

@pytest.mark.serial
def test_fluxSurfaceAdvection_gridIntegration():
    npts = [10,20,10,10]
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'flux_surface')
    
    dt=0.1
    
    fluxAdv = fluxSurfaceAdvection(grid.eta_grid[1:3], grid.get2DSpline())
    
    for i,r in grid.getCoords(0):
        for j,v in grid.getCoords(1):
            fluxAdv.step(grid.get2DSlice([i,j]),dt,v)

@pytest.mark.serial
def test_vParallelAdvection_gridIntegration():
    npts = [4,4,4,100]
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'v_parallel')
    
    dt=0.1
    c=0
    
    old_f=grid._f.copy()
    
    vParAdv = vParallelAdvection(grid.eta_grid[3], grid.get1DSpline())
    
    for i,r in grid.getCoords(0):
        for j,z in grid.getCoords(1):
            for k,q in grid.getCoords(2):
                vParAdv.step(grid.get1DSlice([i,j,k]),dt,c,r)
    
    assert(np.allclose(old_f,grid._f))

@pytest.mark.serial
def test_poloidalAdvection_gridIntegration():
    npts = [10,20,10,10]
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'poloidal')
    
    polAdv = poloidalAdvection(grid.eta_grid[1::-1], grid.get2DSpline())
    
    #~ for i,z in grid.getCoords(0):
        #~ for j,v in grid.getCoords(1):
            #~ poloidalAdv(grid.get2DSlice([i,j]))
