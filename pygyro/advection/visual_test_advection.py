import pytest
import matplotlib.pyplot    as plt
from mpl_toolkits.mplot3d import Axes3D
from math                 import pi

#import cProfile, pstats, io

from ..                         import splines as spl
from ..initialisation.setups    import setupCylindricalGrid
from .advection                 import *

@pytest.mark.serial
def test_fluxSurfaceAdvection():
    npts = [30,20]
    eta_vals = [np.linspace(0,1,4),np.linspace(0,2*pi,npts[0],endpoint=False),
                np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,1,4)]
    
    N = 100
    
    dt=0.1
    c=2
    
    f_vals = np.ndarray([npts[0],npts[1],N+1])
    
    domain    = [ [0,2*pi], [0,20] ]
    nkts      = [n+1                           for n          in npts ]
    breaks    = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots     = [spl.make_knots( b,3,True )    for b          in breaks]
    bsplines  = [spl.BSplines( k,3,True )      for k          in knots]
    eta_grids = [bspl.greville                 for bspl       in bsplines]
    
    eta_vals[1]=eta_grids[0]
    eta_vals[2]=eta_grids[1]
    
    fluxAdv = fluxSurfaceAdvection(eta_vals, bsplines)
    
    #f_vals[:,:,0]=np.exp(-((np.atleast_2d(eta_vals[1]).T-pi)**2+(eta_vals[2]-10)**2)/4)
    f_vals[:,:,0]=np.sin(eta_vals[2]*pi/10)
    
    # profiling:
    #pr = cProfile.Profile()
    #pr.enable()
    
    for n in range(N):
        f_vals[:,:,n+1]=f_vals[:,:,n]
        fluxAdv.step(f_vals[:,:,n+1],dt,c)
    
    #pr.disable()
    #s = io.StringIO()
    #sortby = 'cumulative'
    #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    #ps.print_stats()
    #print(s.getvalue())
    
    
    x,y = np.meshgrid(eta_vals[2], eta_vals[1])
    
    f_min = np.min(f_vals)
    f_max = np.max(f_vals)
    
    plt.ion()

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.25, 0.7, 0.7],)
    colorbarax2 = fig.add_axes([0.85, 0.1, 0.03, 0.8],)
    
    line1 = ax.pcolormesh(x,y,f_vals[:,:,0],vmin=f_min,vmax=f_max)
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    fig.colorbar(line1, cax = colorbarax2)
    
    for n in range(1,N):
        del line1
        line1 = ax.pcolormesh(x,y,f_vals[:,:,n],vmin=f_min,vmax=f_max)
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    print(np.max(f_vals[:,:,n]-f_vals[:,:,0]))

@pytest.mark.serial
def test_vParallelAdvection():
    pass

@pytest.mark.serial
def test_poloidalAdvection():
    pass

"""
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
"""

@pytest.mark.serial
def test_fluxSurfaceAdvection_gridIntegration():
    npts = [4,20,20,4]
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'flux_surface')
    
    #N = 20
    N = 2
    
    dt=0.1
    c=1
    
    f_vals = np.ndarray([npts[1],npts[2],N])
    
    fluxAdv = fluxSurfaceAdvection(grid.eta_grid, grid.get2DSpline())
    
    f_vals[:,:,0]=grid.get2DSlice([0,0])
    for n in range(1,N):
        fluxAdv.step(grid.get2DSlice([0,0]),dt,c)
        f_vals[:,:,n]=grid.get2DSlice([0,0])
    
    x,y = np.meshgrid(grid.eta_grid[1], grid.eta_grid[2])
    
    plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1 = ax.pcolormesh(x,y,f_vals[:,:,0])
    plt.show()
    """
    for n in range(1,N):
        del line1
        line1 = ax.plot_surface(x,y, f_vals[:,:,n],color = 'b')
        fig.canvas.draw()
        fig.canvas.flush_events()
    """

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
