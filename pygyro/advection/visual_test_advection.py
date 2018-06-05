from mpi4py                 import MPI
import pytest
import matplotlib.pyplot    as plt
from mpl_toolkits.mplot3d import Axes3D
from math                 import pi

from ..                         import splines as spl
from ..initialisation.setups    import setupCylindricalGrid
from .advection                 import *
from ..utilities.grid_plotter   import SlicePlotter4d

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
    
    for n in range(N):
        f_vals[:,:,n+1]=f_vals[:,:,n]
        fluxAdv.step(f_vals[:,:,n+1],dt,c)
    
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
def test_poloidalAdvection_invariantPhi():
    npts = [30,20]
    eta_vals = [np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,2*pi,npts[0],endpoint=False),
                np.linspace(0,1,4),np.linspace(0,1,4)]
    
    N = 200
    dt=0.1
    
    v=0
    
    f_vals = np.ndarray([npts[1],npts[0],N+1])
    
    deg = 3
    
    domain    = [ [0.1,14.5], [0,2*pi] ]
    periodic  = [ False, True ]
    nkts      = [n+1+deg*(int(p)-1)            for (n,p)      in zip( npts, periodic )]
    breaks    = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots     = [spl.make_knots( b,deg,p )     for b,p        in zip(breaks,periodic)]
    bsplines  = [spl.BSplines( k,deg,p )       for k,p        in zip(knots,periodic)]
    eta_grids = [bspl.greville                 for bspl       in bsplines]
    
    eta_vals[0]=eta_grids[0]
    eta_vals[1]=eta_grids[1]
    
    polAdv = poloidalAdvection(eta_vals, bsplines[::-1])
    
    phi = Spline2D(bsplines[1],bsplines[0])
    phiVals = np.empty([npts[1],npts[0]])
    phiVals[:]=3*eta_vals[0]**2 * (1+ 1e-1 * np.cos(np.atleast_2d(eta_vals[1]).T*2))
    #phiVals[:]=10*eta_vals[0]
    interp = SplineInterpolator2D(bsplines[1],bsplines[0])
    
    interp.compute_interpolant(phiVals,phi)
    
    #f_vals[:,:,0] = np.exp(-np.atleast_2d((eta_vals[1]-pi)**2).T - (eta_vals[0]-7)**2)/4 + fEq(0.1,v)
    f_vals[:,:,0] = phiVals + fEq(0.1,v)
    
    for n in range(N):
        f_vals[:,:,n+1]=f_vals[:,:,n]
        polAdv.step(f_vals[:,:,n+1],dt,phi,v)
    
    f_min = np.min(f_vals)
    f_max = np.max(f_vals)
    
    plt.ion()

    fig = plt.figure()
    ax = plt.subplot(111, projection='polar')
    #ax = fig.add_axes([0.1, 0.25, 0.7, 0.7],)
    colorbarax2 = fig.add_axes([0.85, 0.1, 0.03, 0.8],)
    
    plotParams = {'vmin':f_min,'vmax':f_max, 'cmap':"jet"}
    
    line1 = ax.contourf(eta_vals[1],eta_vals[0],f_vals[:,:,0].T,20,**plotParams)
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    fig.colorbar(line1, cax = colorbarax2)
    
    for n in range(1,N):
        for coll in line1.collections:
            coll.remove()
        del line1
        line1 = ax.contourf(eta_vals[1],eta_vals[0],f_vals[:,:,n].T,20,**plotParams)
        fig.canvas.draw()
        fig.canvas.flush_events()

@pytest.mark.serial
def test_poloidalAdvection_vortex():
    npts = [30,20]
    eta_vals = [np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,2*pi,npts[0],endpoint=False),
                np.linspace(0,1,4),np.linspace(0,1,4)]
    
    N = 200
    dt=0.1
    
    v=0
    
    f_vals = np.ndarray([npts[1],npts[0],N+1])
    
    deg = 3
    
    domain    = [ [0.1,14.5], [0,2*pi] ]
    periodic  = [ False, True ]
    nkts      = [n+1+deg*(int(p)-1)            for (n,p)      in zip( npts, periodic )]
    breaks    = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots     = [spl.make_knots( b,deg,p )     for b,p        in zip(breaks,periodic)]
    bsplines  = [spl.BSplines( k,deg,p )       for k,p        in zip(knots,periodic)]
    eta_grids = [bspl.greville                 for bspl       in bsplines]
    
    eta_vals[0]=eta_grids[0]
    eta_vals[1]=eta_grids[1]
    
    polAdv = poloidalAdvection(eta_vals, bsplines[::-1])
    
    phi = Spline2D(bsplines[1],bsplines[0])
    phiVals = np.empty([npts[1],npts[0]])
    phiVals[:]=10*eta_vals[0]
    interp = SplineInterpolator2D(bsplines[1],bsplines[0])
    
    interp.compute_interpolant(phiVals,phi)
    
    f_vals[:,:,0] = np.exp(-np.atleast_2d((eta_vals[1]-pi)**2).T - (eta_vals[0]-7)**2)/4 + fEq(0.1,v)
    
    for n in range(N):
        f_vals[:,:,n+1]=f_vals[:,:,n]
        polAdv.step(f_vals[:,:,n+1],dt,phi,v)
    
    f_min = np.min(f_vals)
    f_max = np.max(f_vals)
    
    plt.ion()

    fig = plt.figure()
    ax = plt.subplot(111, projection='polar')
    #ax = fig.add_axes([0.1, 0.25, 0.7, 0.7],)
    colorbarax2 = fig.add_axes([0.85, 0.1, 0.03, 0.8],)
    
    plotParams = {'vmin':f_min,'vmax':f_max, 'cmap':"jet"}
    
    line1 = ax.contourf(eta_vals[1],eta_vals[0],f_vals[:,:,0].T,20,**plotParams)
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    fig.colorbar(line1, cax = colorbarax2)
    
    for n in range(1,N):
        for coll in line1.collections:
            coll.remove()
        del line1
        line1 = ax.contourf(eta_vals[1],eta_vals[0],f_vals[:,:,n].T,20,**plotParams)
        fig.canvas.draw()
        fig.canvas.flush_events()

@pytest.mark.serial
def test_poloidalAdvection_constantAdv():
    npts = [30,20]
    eta_vals = [np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,2*pi,npts[0],endpoint=False),
                np.linspace(0,1,4),np.linspace(0,1,4)]
    
    N = 200
    dt=0.1
    
    v=0
    
    f_vals = np.ndarray([npts[1],npts[0],N+1])
    
    deg = 3
    
    domain    = [ [0.1,14.5], [0,2*pi] ]
    periodic  = [ False, True ]
    nkts      = [n+1+deg*(int(p)-1)            for (n,p)      in zip( npts, periodic )]
    breaks    = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots     = [spl.make_knots( b,deg,p )     for b,p        in zip(breaks,periodic)]
    bsplines  = [spl.BSplines( k,deg,p )       for k,p        in zip(knots,periodic)]
    eta_grids = [bspl.greville                 for bspl       in bsplines]
    
    eta_vals[0]=eta_grids[0]
    eta_vals[1]=eta_grids[1]
    
    polAdv = poloidalAdvection(eta_vals, bsplines[::-1])
    
    phi = Spline2D(bsplines[1],bsplines[0])
    phiVals = np.empty([npts[1],npts[0]])
    phiVals[:]=3*eta_vals[0]**2
    interp = SplineInterpolator2D(bsplines[1],bsplines[0])
    
    interp.compute_interpolant(phiVals,phi)
    
    f_vals[:,:,0] = np.exp(-np.atleast_2d((eta_vals[1]-pi)**2).T - (eta_vals[0]-7)**2)/4 + fEq(0.1,v)
    
    for n in range(N):
        f_vals[:,:,n+1]=f_vals[:,:,n]
        polAdv.step(f_vals[:,:,n+1],dt,phi,v)
    
    f_min = np.min(f_vals)
    f_max = np.max(f_vals)
    
    plt.ion()

    fig = plt.figure()
    ax = plt.subplot(111, projection='polar')
    #ax = fig.add_axes([0.1, 0.25, 0.7, 0.7],)
    colorbarax2 = fig.add_axes([0.85, 0.1, 0.03, 0.8],)
    
    plotParams = {'vmin':f_min,'vmax':f_max, 'cmap':"jet"}
    
    line1 = ax.contourf(eta_vals[1],eta_vals[0],f_vals[:,:,0].T,20,**plotParams)
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    fig.colorbar(line1, cax = colorbarax2)
    
    for n in range(1,N):
        for coll in line1.collections:
            coll.remove()
        del line1
        line1 = ax.contourf(eta_vals[1],eta_vals[0],f_vals[:,:,n].T,20,**plotParams)
        fig.canvas.draw()
        fig.canvas.flush_events()

@pytest.mark.serial
def test_vParallelAdvection():
    npts = [4,4,4,100]
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'v_parallel')
    
    N = 100
    
    dt=0.1
    
    c = 1.0
    
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

@pytest.mark.parallel
def test_equilibrium():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    npts = [20,20,10,8]
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'flux_surface',
                                eps    = 0.1,
                                comm   = comm,
                                plot_thread = True)
    
    plt.ion()
    
    plot = SlicePlotter4d(grid,False,comm,drawingRank=0,drawRankInGrid=False)
    
    N=10
    
    if (rank!=0):
        
        fluxAdv = fluxSurfaceAdvection(grid.eta_grid, grid.get2DSpline())
        vParAdv = vParallelAdvection(grid.eta_grid, grid.getSpline(3))
        polAdv = poloidalAdvection(grid.eta_grid, grid.getSpline(slice(1,None,-1)))
        
        dt=0.1
        halfStep = dt*0.5
        
        phi = Spline2D(grid.getSpline(1),grid.getSpline(0))
        phiVals = np.empty([npts[1],npts[0]])
        phiVals[:]=3
        #phiVals[:]=10*eta_vals[0]
        interp = SplineInterpolator2D(grid.getSpline(1),grid.getSpline(0))
    
    for n in range(N):
        if (rank!=0):
            for i,r in grid.getCoords(0):
                for j,v in grid.getCoords(1):
                    fluxAdv.step(grid.get2DSlice([i,j]),halfStep,v)
            
            grid.setLayout('v_parallel')
            
            for i,r in grid.getCoords(0):
                for j,z in grid.getCoords(1):
                    for k,q in grid.getCoords(2):
                        vParAdv.step(grid.get1DSlice([i,j,k]),halfStep,0,r)
            
            grid.setLayout('poloidal')
            
            for i,v in grid.getCoords(0):
                for j,z in grid.getCoords(1):
                    polAdv.step(grid.get2DSlice([i,j]),dt,phi,v)
            
            grid.setLayout('v_parallel')
            
            for i,r in grid.getCoords(0):
                for j,z in grid.getCoords(1):
                    for k,q in grid.getCoords(2):
                        vParAdv.step(grid.get1DSlice([i,j,k]),halfStep,0,r)
            
            grid.setLayout('flux_surface')
            
            for i,r in grid.getCoords(0):
                for j,v in grid.getCoords(1):
                    fluxAdv.step(grid.get2DSlice([i,j]),halfStep,v)
        
        plot.updateDraw()
        if (plot.listen()==0):
            break
