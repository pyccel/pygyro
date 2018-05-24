import pytest

from ..initialisation.setups        import setupCylindricalGrid
from ..initialisation.initialiser   import fEq
from .advection                     import *
from ..                     import splines as spl

def gauss(x):
    return np.exp(-x**2/4)

def iota0():
    return 0.0

@pytest.mark.serial
@pytest.mark.parametrize( "fact,dt", [(10,1),(10,0.1), (5,1),(5,0.1), (2,1), (2,0.1)] )
def test_fluxSurfaceAdvection(fact,dt):
    npts = [30,20]
    eta_vals = [np.linspace(0,1,4),np.linspace(0,2*pi,npts[0],endpoint=False),
                np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,1,4)]
    
    N = 100
    
    dt=1
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
    
    assert(np.max(np.abs(f_vals-f_end))<1e-8)

@pytest.mark.serial
@pytest.mark.parametrize( "function,N", [(gauss,10),(gauss,20),(gauss,30)] )
def test_vParallelAdvection(function,N):
    npts = 50
    f = np.empty(npts)
    N=10
    
    dt=0.1
    c=2
    
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
    
    polAdv = poloidalAdvection(grid.eta_grid, grid.get2DSpline())
    
    #~ for i,z in grid.getCoords(0):
        #~ for j,v in grid.getCoords(1):
            #~ poloidalAdv(grid.get2DSlice([i,j]))
