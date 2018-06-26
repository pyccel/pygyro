import pytest
import timeit
from scipy.integrate        import trapz

from .advection                     import *
from ..                             import splines as spl
from ..initialisation               import constants
from ..model.layout                 import Layout

NSteps = 10000
NGrid = 20

def iota0():
    return 0.0

@pytest.mark.serial
def test_fluxSurfaceAdvection():
    npts = [NGrid,NGrid]
    eta_vals = [np.linspace(0,1,4),np.linspace(0,2*pi,npts[0],endpoint=False),
                np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,1,4)]
    
    dt=0.1
    
    f_vals = np.ndarray(npts)
    
    domain    = [ [0,2*pi], [0,20] ]
    nkts      = [n+1                           for n          in npts ]
    breaks    = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots     = [spl.make_knots( b,3,True )    for b          in breaks]
    bsplines  = [spl.BSplines( k,3,True )      for k          in knots]
    eta_grids = [bspl.greville                 for bspl       in bsplines]
    
    c=2
    
    eta_vals[1]=eta_grids[0]
    eta_vals[2]=eta_grids[1]
    eta_vals[3][0]=c
    
    layout = Layout('flux',[1],[0,3,1,2],eta_vals,[0])
    
    fluxAdv = FluxSurfaceAdvection(eta_vals, bsplines, layout, dt, iota0)
    
    f_vals[:,:] = np.sin(eta_vals[2]*pi/10)
    
    print(timeit.Timer(lambda: fluxAdv.step(f_vals,0)).timeit(NSteps)/NSteps)

def gauss(x):
    return np.exp(-x**2/4)

@pytest.mark.serial
def test_vParallelAdvection():
    npts = NGrid
    f = np.empty(npts)
    
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
    
    f = gauss(x)+fEdge
    
    vParAdv = VParallelAdvection([0,0,0,x], spline, lambda r,v : 0)
    
    print(timeit.Timer(lambda: vParAdv.step(f,dt,c,r)).timeit(NSteps)/NSteps)


def Phi(r,theta):
    return - 5 * r**2 + np.sin(theta)

def initConditions(r,theta):
    a=4
    factor = pi/a/2
    r=np.sqrt((r-7)**2+2*(theta-pi)**2)
    
    if (r<=a):
        return np.cos(r*factor)**4
    else:
        return 0.0

initConds = np.vectorize(initConditions, otypes=[np.float])

@pytest.mark.serial
def test_explicitPoloidalAdvection():
    npts = [NGrid,NGrid]
    
    dt=0.1
    v=2
    
    eta_vals = [np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,2*pi,npts[0],endpoint=False),
                np.linspace(0,1,4),np.linspace(0,1,4)]
    
    N = int(1/dt)
    
    f_vals = np.ndarray([npts[1],npts[0]])
    final_f_vals = np.ndarray([npts[1],npts[0]])
    
    deg = 3
    
    domain    = [ [1,14.5], [0,2*pi] ]
    periodic  = [ False, True ]
    nkts      = [n+1+deg*(int(p)-1)            for (n,p)      in zip( npts, periodic )]
    breaks    = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots     = [spl.make_knots( b,deg,p )     for b,p        in zip(breaks,periodic)]
    bsplines  = [spl.BSplines( k,deg,p )       for k,p        in zip(knots,periodic)]
    eta_grids = [bspl.greville                 for bspl       in bsplines]
    
    eta_vals[0]=eta_grids[0]
    eta_vals[1]=eta_grids[1]
    eta_vals[3][0]=v
    
    polAdv = PoloidalAdvection(eta_vals, bsplines[::-1], lambda r,v : 0)
    
    phi = Spline2D(bsplines[1],bsplines[0])
    phiVals = np.empty([npts[1],npts[0]])
    phiVals[:] = Phi(eta_vals[0],np.atleast_2d(eta_vals[1]).T)
    interp = SplineInterpolator2D(bsplines[1],bsplines[0])
    
    interp.compute_interpolant(phiVals,phi)
    
    f_vals[:,:] = initConds(eta_vals[0],np.atleast_2d(eta_vals[1]).T)
    
    print(timeit.Timer(lambda: polAdv.step(f_vals,dt,phi,v)).timeit(NSteps)/NSteps)
