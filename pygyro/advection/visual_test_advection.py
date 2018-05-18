import pytest
import matplotlib.pyplot    as plt

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
    
    dt=0.1
    
    fluxAdv = fluxSurfaceAdvection(grid.eta_grid, grid.get2DSpline())
    
    for i,r in grid.getCoords(0):
        for j,v in grid.getCoords(1):
            fluxAdv.step(grid.get2DSlice([i,j]),dt,v)

@pytest.mark.serial
def test_vParallelAdvection_gridIntegration():
    npts = [4,4,4,100]
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'v_parallel')
    
    N = 100
    
    dt=0.1
    c=1
    
    f_vals = np.ndarray([npts[3],N])
    
    vParAdv = vParallelAdvection(grid.eta_grid, grid.get1DSpline())
    
    for n in range(N):
        for i,r in grid.getCoords(0):
            vParAdv.step(grid.get1DSlice([i,0,0]),dt,c,r)
        f_vals[:,n]=grid.get1DSlice([0,0,0])
    
    plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(grid.eta_grid[3], f_vals[:,0]) # Returns a tuple of line objects, thus the comma

    for n in range(1,N):
        line1.set_ydata(f_vals[:,n])
        fig.canvas.draw()
        fig.canvas.flush_events()

@pytest.mark.serial
def test_poloidalAdvection_gridIntegration():
    npts = [10,20,10,10]
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'poloidal')
    
    polAdv = poloidalAdvection(grid.eta_grid, grid.get2DSpline())
    
    #~ for i,z in grid.getCoords(0):
        #~ for j,v in grid.getCoords(1):
            #~ poloidalAdv(grid.get2DSlice([i,j]))
