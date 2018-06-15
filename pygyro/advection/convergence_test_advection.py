from mpi4py                 import MPI
import pytest
import numpy                as np
import matplotlib.pyplot    as plt
from scipy.integrate        import trapz

from ..initialisation.setups        import setupCylindricalGrid
from ..initialisation.initialiser   import fEq
from .advection                     import *
from ..                             import splines as spl
from ..initialisation               import constants

def gaussLike(x):
    return np.cos(pi*x*0.1)**4

@pytest.mark.serial
def test_vParallelAdvection():
    npts = 32
    
    nconvpts = 7
    
    N=100
    
    dt=0.01
    c=2.0
    
    l2 = [[npts*(2**(i+1)),0] for i in range(nconvpts)]
    linf = [[npts*(2**(i+1)),0] for i in range(nconvpts)]
    
    for j in range(nconvpts):
        npts*=2
        dt/=2
        print(npts)
        f = np.empty(npts)
        
        nkts      = npts-2
        breaks    = np.linspace( -5, 5, num=nkts )
        knots     = spl.make_knots( breaks,3,False )
        spline    = spl.BSplines( knots,3,False )
        x         = spline.greville
        
        r = 4
        
        f = gaussLike(x)
        
        vParAdv = VParallelAdvection([0,0,0,x], spline, lambda r,v : 0)
        
        for i in range(N):
            vParAdv.step(f,dt,c,r)
        
        fEnd = np.empty(npts)
        
        for i in range(npts):
            if ((x[i]-c*dt*N)<x[0]):
                #fEnd[i]=fEq(r,(x[i]-c*dt*N))
                fEnd[i]=0
            else:
                fEnd[i]=gaussLike(x[i]-c*dt*N)
        
        l2[j][1]=np.linalg.norm((f-fEnd).flatten(),2)
        linf[j][1]=np.linalg.norm((f-fEnd).flatten(),np.inf)
    
    print(l2)
    print(linf)
    l2Imp = np.ndarray(nconvpts-1)
    l2Order = np.ndarray(nconvpts-1)
    linfImp = np.ndarray(nconvpts-1)
    linfOrder = np.ndarray(nconvpts-1)
    for i in range(nconvpts-1):
        l2Imp[i]=l2[i][1]/l2[i+1][1]
        l2Order[i]=np.log2(l2Imp[i])
        linfImp[i]=linf[i][1]/linf[i+1][1]
        linfOrder[i]=np.log2(linfImp[i])
    print(l2Imp)
    print(linfImp)
    print(l2Order)
    print(linfOrder)
    print(np.mean(l2Imp))
    print(np.mean(linfImp))
    print(np.mean(l2Order))
    print(np.mean(linfOrder))

def Phi(r,theta):
    return - 5 * r**2 + np.sin(theta)

factor = pi/8
rFact = 1/36
thetaFact = 1/pi

def initConditions(r,theta):
    r=np.sqrt((r-7)**2+2*(theta-pi)**2)
    if (r<=4):
        return np.cos(r*factor)**4
    else:
        return 0.0

initConds = np.vectorize(initConditions, otypes=[np.float])

@pytest.mark.serial
def test_poloidalAdvection_constantAdv():
    npts = [16,16]
    dt=0.02
    
    nconvpts = 5
    
    l2 = np.ndarray(nconvpts)
    linf = np.ndarray(nconvpts)
    
    for i in range(nconvpts):
        npts[0]*=2
        npts[1]*=2
        dt/=2
        print(npts)
        eta_vals = [np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,2*pi,npts[0],endpoint=False),
                    np.linspace(0,1,4),np.linspace(0,1,4)]
        
        N = 100
        
        v=0
        
        f_vals = np.ndarray([npts[1],npts[0]])
        final_f_vals = np.ndarray([npts[1],npts[0]])
        
        deg = 3
        
        domain    = [ [1,13], [0,2*pi] ]
        periodic  = [ False, True ]
        nkts      = [n+1+deg*(int(p)-1)            for (n,p)      in zip( npts, periodic )]
        breaks    = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
        knots     = [spl.make_knots( b,deg,p )     for b,p        in zip(breaks,periodic)]
        bsplines  = [spl.BSplines( k,deg,p )       for k,p        in zip(knots,periodic)]
        eta_grids = [bspl.greville                 for bspl       in bsplines]
        
        eta_vals[0]=eta_grids[0]
        eta_vals[1]=eta_grids[1]
        
        polAdv = PoloidalAdvection(eta_vals, bsplines[::-1])
        
        phi = Spline2D(bsplines[1],bsplines[0])
        phiVals = np.empty([npts[1],npts[0]])
        phiVals[:] = Phi(eta_vals[0],np.atleast_2d(eta_vals[1]).T)
        interp = SplineInterpolator2D(bsplines[1],bsplines[0])
        
        interp.compute_interpolant(phiVals,phi)
        
        f_vals[:,:] = initConds(eta_vals[0],np.atleast_2d(eta_vals[1]).T)
        finalPts = ( np.ndarray([npts[1],npts[0]]), np.ndarray([npts[1],npts[0]]))
        finalPts[0][:] = np.mod(polAdv._shapedQ   +     10*dt*N/constants.B0,2*pi)
        finalPts[1][:] = np.sqrt(polAdv._points[1]**2-np.sin(polAdv._shapedQ)/5/constants.B0 \
                        + np.sin(finalPts[0])/5/constants.B0)
        final_f_vals[:,:] = initConds(finalPts[1],finalPts[0])
        
        endPts = ( np.ndarray([npts[1],npts[0]]), np.ndarray([npts[1],npts[0]]))
        endPts[0][:] = polAdv._shapedQ   +     10*dt/constants.B0
        endPts[1][:] = np.sqrt(polAdv._points[1]**2-np.sin(polAdv._shapedQ)/5/constants.B0 \
                        + np.sin(endPts[0])/5/constants.B0)
        
        for n in range(N):
            polAdv.exact_step(f_vals[:,:],endPts,v)
        
        linf[i]=np.linalg.norm((f_vals-final_f_vals).flatten(),np.inf)
        l2[i]=np.sqrt(trapz(trapz((f_vals-final_f_vals)**2,eta_grids[1],axis=0)*eta_grids[0],eta_grids[0]))
        
        print("l2:",l2[i])
        print("linf:",linf[i])
    
    print("l2:",l2)
    print("linf:",linf)
    
    print("l2 order:",l2[:-1]/l2[1:])
    print("linf order:",linf[:-1]/linf[1:])
    l2order = np.log2(l2[:-1]/l2[1:])
    linforder = np.log2(linf[:-1]/linf[1:])
    print("l2 order:",l2order)
    print("linf order:",linforder)
    
    print(32," & & ",32,"    & & $",end=' ')
    mag2Order = np.floor(np.log10(l2[0]))
    maginfOrder = np.floor(np.log10(linf[0]))
    print(str.format('{0:.2f}',l2[0]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
    print("}$ &       & $",str.format('{0:.2f}',linf[0]*10**-maginfOrder),"\\cdot 10^{", str.format('{0:n}',maginfOrder),end=' ')
    print("}$ &  \\")
    for i in range(nconvpts-2):
        n=2**(i+6)
        mag2Order = np.floor(np.log10(l2[i+1]))
        maginfOrder = np.floor(np.log10(linf[i+1]))
        print(n," & & ",n,"    & & $",end=' ')
        print(str.format('{0:.2f}',l2[i+1]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
        print("}$ & ",str.format('{0:.2f}',l2order[i])," & $",end=' ')
        print(str.format('{0:.2f}',linf[i+1]*10**-maginfOrder),"\\cdot 10^{", str.format('{0:n}',maginfOrder),end=' ')
        print("}$ & ",str.format('{0:.2f}',linforder[i])," \\ \\")

@pytest.mark.serial
def test_poloidalAdvection_constantAdv_dt():
    dt=0.2
    
    nconvpts = 5
    
    l2 = np.ndarray(nconvpts)
    linf = np.ndarray(nconvpts)
    
    npts = [200,200]
    eta_vals = [np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,2*pi,npts[0],endpoint=False),
                np.linspace(0,1,4),np.linspace(0,1,4)]
    
    for i in range(nconvpts):
        dt/=2
        print(dt)
        N = int(1/dt)
                
        v=0
        
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
        
        polAdv = PoloidalAdvection(eta_vals, bsplines[::-1])
        
        phi = Spline2D(bsplines[1],bsplines[0])
        phiVals = np.empty([npts[1],npts[0]])
        phiVals[:] = Phi(eta_vals[0],np.atleast_2d(eta_vals[1]).T)
        interp = SplineInterpolator2D(bsplines[1],bsplines[0])
        
        interp.compute_interpolant(phiVals,phi)
        
        f_vals[:,:] = initConds(eta_vals[0],np.atleast_2d(eta_vals[1]).T)
        finalPts = ( np.ndarray([npts[1],npts[0]]), np.ndarray([npts[1],npts[0]]))
        finalPts[0][:] = np.mod(polAdv._shapedQ   +     10*dt*N/constants.B0,2*pi)
        finalPts[1][:] = np.sqrt(polAdv._points[1]**2-np.sin(polAdv._shapedQ)/5/constants.B0 \
                        + np.sin(finalPts[0])/5/constants.B0)
        final_f_vals[:,:] = initConds(finalPts[1],finalPts[0])
        
        endPts = ( np.ndarray([npts[1],npts[0]]), np.ndarray([npts[1],npts[0]]))
        endPts[0][:] = polAdv._shapedQ   -     10*dt/constants.B0
        endPts[1][:] = np.sqrt(polAdv._points[1]**2-np.sin(polAdv._shapedQ)/5/constants.B0 \
                        + np.sin(endPts[0])/5/constants.B0)
        
        for n in range(N):
            polAdv.step(f_vals[:,:],dt,phi,v)
        
        linf[i]=np.linalg.norm((f_vals-final_f_vals).flatten(),np.inf)
        l2[i]=np.sqrt(trapz(trapz((f_vals-final_f_vals)**2,eta_grids[1],axis=0)*eta_grids[0],eta_grids[0]))
        
        print(N,"l2:",l2[i])
        print(N,"linf:",linf[i])
    
    print("l2:",l2)
    print("linf:",linf)
    
    print("l2 order:",l2[:-1]/l2[1:])
    print("linf order:",linf[:-1]/linf[1:])
    l2Order=np.log2(l2[:-1]/l2[1:])
    linfOrder=np.log2(linf[:-1]/linf[1:])
    print("l2 order:",l2Order)
    print("linf order:",linfOrder)
    print(np.mean(l2Order))
    print(np.mean(linfOrder))
    
    print(10," & & ",0.1,"    & & $",end=' ')
    mag2Order = np.floor(np.log10(l2[0]))
    maginfOrder = np.floor(np.log10(linf[0]))
    print(str.format('{0:.2f}',l2[0]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
    print("}$ &       & $",str.format('{0:.2f}',linf[0]*10**-maginfOrder),"\\cdot 10^{", str.format('{0:n}',maginfOrder),end=' ')
    print("}$ &  \\")
    for i in range(nconvpts-2):
        n = 20*2**i
        dt = 0.05/2**i
        mag2Order = np.floor(np.log10(l2[i+1]))
        maginfOrder = np.floor(np.log10(linf[i+1]))
        print(n," & & ",dt,"    & & $",end=' ')
        print(str.format('{0:.2f}',l2[i+1]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
        print("}$ & ",str.format('{0:.2f}',l2Order[i])," & $",end=' ')
        print(str.format('{0:.2f}',linf[i+1]*10**-maginfOrder),"\\cdot 10^{", str.format('{0:n}',maginfOrder),end=' ')
        print("}$ & ",str.format('{0:.2f}',linfOrder[i])," \\ \\")
