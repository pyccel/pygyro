import pytest

from ..initialisation.setups        import setupCylindricalGrid
from ..initialisation.initialiser   import fEq
from .advection                     import *
from ..                             import splines as spl
from ..initialisation               import constants

def gauss(x):
    return np.exp(-x**2/4)

def iota0():
    return 0.0

@pytest.mark.serial
@pytest.mark.parametrize( "fact,dt", [(10,1),(10,0.1), (5,1)] )
def test_fluxSurfaceAdvection(fact,dt):
    npts = [30,20]
    eta_vals = [np.linspace(0,1,4),np.linspace(0,2*pi,npts[0],endpoint=False),
                np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,1,4)]
    
    N = 100
    
    c=2
    
    f_vals = np.ndarray(npts)
    
    domain    = [ [0,2*pi], [0,20] ]
    nkts      = [n+1                           for n          in npts ]
    breaks    = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots     = [spl.make_knots( b,3,True )    for b          in breaks]
    bsplines  = [spl.BSplines( k,3,True )      for k          in knots]
    eta_grids = [bspl.greville                 for bspl       in bsplines]
    
    eta_vals[1]=eta_grids[0]
    eta_vals[2]=eta_grids[1]
    
    fluxAdv = fluxSurfaceAdvection(eta_vals, bsplines, iota0)
    
    f_vals[:,:] = np.sin(eta_vals[2]*pi/fact)
    f_end = np.sin((eta_vals[2]-c*dt*N)*pi/fact)
    
    for n in range(N):
        fluxAdv.step(f_vals,dt,c)
    
    assert(np.max(np.abs(f_vals-f_end))<1e-2)

@pytest.mark.serial
@pytest.mark.parametrize( "function,N", [(gauss,10),(gauss,20),(gauss,30)] )
def test_vParallelAdvection(function,N):
    npts = 50
    f = np.empty(npts)
    N=10
    
    dt=0.1
    c=2.0
    
    nkts      = npts-2
    breaks    = np.linspace( -5, 5, num=nkts )
    knots     = spl.make_knots( breaks,3,False )
    spline    = spl.BSplines( knots,3,False )
    x         = spline.greville
    
    r = 4
    fEdge = fEq(r,x[0])
    assert(fEq(r,x[0])==fEq(r,x[-1]))
    
    f = function(x)+fEdge
    
    vParAdv = vParallelAdvection([0,0,0,x], spline)
    
    for i in range(N):
        vParAdv.step(f,dt,c,r)
    
    fEnd = np.empty(npts)
    
    for i in range(npts):
        if ((x[i]-c*dt*N)<x[0]):
            fEnd[i]=fEq(r,(x[i]-c*dt*N))
        else:
            fEnd[i]=fEdge+function(x[i]-c*dt*N)
    
    print(max(abs(f-fEnd)))
    assert(max(abs(f-fEnd))<1e-3)

@pytest.mark.serial
@pytest.mark.parametrize( "fact,dt", [(10,1),(10,0.1), (5,1),(5,0.1), (2,1), (2,0.1)] )
def test_poloidalAdvection(fact,dt):
    npts = [30,20]
    eta_vals = [np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,2*pi,npts[0],endpoint=False),
                np.linspace(0,1,4),np.linspace(0,1,4)]
    
    N = 100
    
    f_vals = np.ndarray([npts[1],npts[0]])
    
    domain    = [ [0.1,14.5], [0,2*pi] ]
    periodic  = [ False, True ]
    nkts      = [n+1+3*(int(p)-1)              for (n,p)    in zip( npts, periodic )]
    breaks    = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots     = [spl.make_knots( b,3,p )    for b,p          in zip(breaks,periodic)]
    bsplines  = [spl.BSplines( k,3,p )      for k,p          in zip(knots,periodic)]
    eta_grids = [bspl.greville                 for bspl       in bsplines]
    
    eta_vals[0]=eta_grids[0]
    eta_vals[1]=eta_grids[1]
    
    polAdv = poloidalAdvection(eta_vals, bsplines[::-1])
    
    phi = Spline2D(bsplines[1],bsplines[0])
    phiVals = np.empty([npts[1],npts[0]])
    phiVals[:]=eta_vals[0]*2+np.atleast_2d(eta_vals[1]).T*3
    interp = SplineInterpolator2D(bsplines[1],bsplines[0])
    
    interp.compute_interpolant(phiVals,phi)
    
    f_vals[:,:] = np.atleast_2d(np.sin(eta_vals[1]*pi/fact)).T
    f_end = np.atleast_2d(np.sin((eta_vals[1]-3*dt*N)*pi/fact)).T
    
    v=0
    
    for n in range(N):
        polAdv.step(f_vals,dt,phi,v)
    
    assert(np.max(np.abs(f_vals-f_end))<1e-8)

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
    
    dt=0.1
    c=0
    
    old_f=grid._f.copy()
    
    vParAdv = vParallelAdvection(grid.eta_grid, grid.get1DSpline())
    
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
    
    basis = grid.get2DSpline()
    polAdv = poloidalAdvection(grid.eta_grid, basis)
    
    phi = Spline2D(basis[0],basis[1])
    phiVals = np.full((npts[1],npts[0]),2)
    interp = SplineInterpolator2D(basis[0],basis[1])
    
    interp.compute_interpolant(phiVals,phi)
    
    dt=0.1
    
    for i,z in grid.getCoords(0):
        for j,v in grid.getCoords(1):
            polAdv.step(grid.get2DSlice([i,j]),dt,phi,v)

@pytest.mark.parallel
def test_equilibrium():
    comm = MPI.COMM_WORLD
    
    npts = [20,20,10,8]
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'flux_surface',
                                eps    = 0.1,
                                comm   = comm)
    
    startVals = grid._f.copy()
    
    N=10
        
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
    
    print(np.max(startVals-grid._f))
    assert(np.max(startVals-grid._f)<1e-2)
