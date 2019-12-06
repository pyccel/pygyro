import pytest
import timeit
from scipy.integrate        import trapz
import numpy                as np

from .advection                     import FluxSurfaceAdvection, PoloidalAdvection, VParallelAdvection, ParallelGradient
from ..                             import splines as spl
from ..initialisation.constants     import get_constants
from ..model.layout                 import Layout

NSteps = 1000
NGrid = 20

def iota0():
    return 0.0

@pytest.mark.serial
def test_fluxSurfaceAdvection():
    npts = [NGrid,NGrid]
    eta_vals = [np.linspace(0,1,4),np.linspace(0,2*np.pi,npts[0],endpoint=False),
                np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,1,4)]
    
    dt=0.1
    
    f_vals = np.ndarray(npts)
    
    domain    = [ [0,2*np.pi], [0,20] ]
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
    
    constants = get_constants('testSetups/iota0.json')
    
    fluxAdv = FluxSurfaceAdvection(eta_vals, bsplines, layout, dt, constants)
    
    f_vals[:,:] = np.sin(eta_vals[2]*np.pi/10)
    
    fTime = timeit.Timer(lambda: fluxAdv.step(f_vals,0)).timeit(NSteps)
    print(fTime/NSteps," per step, ",fTime," total")
    print(fTime*NGrid*NGrid/NSteps," per grid ")

def gauss(x):
    return np.exp(-x**2/4)

@pytest.mark.serial
def test_vParallelAdvection():
    npts = NGrid
    f = np.empty(npts)
    
    constants = get_constants('testSetups/iota0.json')
    
    dt=0.1
    c=2.0
    
    nkts      = npts-2
    breaks    = np.linspace( -5, 5, num=nkts )
    knots     = spl.make_knots( breaks,3,False )
    spline    = spl.BSplines( knots,3,False )
    x         = spline.greville
    
    r = 4
    fEdge = fEq(r,x[0],constants.CN0,constants.kN0,constants.deltaRN0,
                constants.rp,constants.CTi,constants.kTi,constants.deltaRTi)
    assert(fEq(r,x[0],constants.CN0,constants.kN0,constants.deltaRN0,
                constants.rp,constants.CTi,constants.kTi,constants.deltaRTi)
            ==fEq(r,x[-1],constants.CN0,constants.kN0,constants.deltaRN0,
                constants.rp,constants.CTi,constants.kTi,constants.deltaRTi))
    
    f = gauss(x)+fEdge
    
    vParAdv = VParallelAdvection([0,0,0,x], spline, constants, edge='null')
    
    vTime = timeit.Timer(lambda: vParAdv.step(f,dt,c,r)).timeit(NSteps)
    print(vTime/NSteps," per step, ",vTime," total")
    print(vTime*NGrid*NGrid*NGrid/NSteps," per grid ")


def Phi(r,theta):
    return - 5 * r**2 + np.sin(theta)

def initConditions(r,theta):
    a=4
    factor = np.pi/a/2
    r=np.sqrt((r-7)**2+2*(theta-np.pi)**2)
    
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
    
    eta_vals = [np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,2*np.pi,npts[0],endpoint=False),
                np.linspace(0,1,4),np.linspace(0,1,4)]
    
    N = int(1/dt)
    
    f_vals = np.ndarray([npts[1],npts[0]])
    final_f_vals = np.ndarray([npts[1],npts[0]])
    
    deg = 3
    
    domain    = [ [1,14.5], [0,2*np.pi] ]
    periodic  = [ False, True ]
    nkts      = [n+1+deg*(int(p)-1)            for (n,p)      in zip( npts, periodic )]
    breaks    = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots     = [spl.make_knots( b,deg,p )     for b,p        in zip(breaks,periodic)]
    bsplines  = [spl.BSplines( k,deg,p )       for k,p        in zip(knots,periodic)]
    eta_grids = [bspl.greville                 for bspl       in bsplines]
    
    eta_vals[0]=eta_grids[0]
    eta_vals[1]=eta_grids[1]
    eta_vals[3][0]=v
    
    constants = get_constants('testSetups/iota0.json')
    
    polAdv = PoloidalAdvection(eta_vals, bsplines[::-1], constants, nulEdge=True)
    
    phi = Spline2D(bsplines[1],bsplines[0])
    phiVals = np.empty([npts[1],npts[0]])
    phiVals[:] = Phi(eta_vals[0],np.atleast_2d(eta_vals[1]).T)
    interp = SplineInterpolator2D(bsplines[1],bsplines[0])
    
    interp.compute_interpolant(phiVals,phi)
    
    f_vals[:,:] = initConds(eta_vals[0],np.atleast_2d(eta_vals[1]).T)
    
    pTime = timeit.Timer(lambda: polAdv.step(f_vals,dt,phi,v)).timeit(NSteps)
    print(pTime/NSteps," per step, ",pTime," total")
    print(pTime*NGrid*NGrid/NSteps," per grid ")
