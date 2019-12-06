from mpi4py                 import MPI
import pytest
import numpy                as np
import matplotlib.pyplot    as plt
from scipy.integrate        import trapz

from ..initialisation.setups        import setupCylindricalGrid
from ..initialisation.mod_initialiser_funcs     import fEq
from .advection                     import *
from ..                             import splines as spl
from ..initialisation.constants     import get_constants

def gaussLike(x):
    return np.cos(pi*x*0.1)**4

@pytest.mark.serial
def test_vParallelAdvection():
    npts = 32
    
    nconvpts = 7
    
    N=100
    dt=0.01
    c=2.0
    
    l2 = np.ndarray(nconvpts)
    linf = np.ndarray(nconvpts)
    
    constants = get_constants('testSetups/iota0.json')
    
    for j in range(nconvpts):
        npts*=2
        dt/=2
        N*=2
        print(npts)
        f = np.empty(npts)
        
        nkts      = npts-2
        breaks    = np.linspace( -5, 5, num=nkts )
        knots     = spl.make_knots( breaks,3,False )
        spline    = spl.BSplines( knots,3,False )
        x         = spline.greville
        
        r = 4
        
        f = gaussLike(x)
        
        vParAdv = VParallelAdvection([0,0,0,x], spline, constants, 'null')
        
        for i in range(N):
            vParAdv.step(f,dt,c,r)
        
        fEnd = np.empty(npts)
        
        for i in range(npts):
            if ((x[i]-c*dt*N)<x[0]):
                #fEnd[i]=fEq(r,(x[i]-c*dt*N))
                fEnd[i]=0
            else:
                fEnd[i]=gaussLike(x[i]-c*dt*N)
        
        l2[j]=np.sqrt(trapz((f-fEnd).flatten()**2,x))
        linf[j]=np.linalg.norm((f-fEnd).flatten(),np.inf)
    
    print("l2:",l2)
    print("linf:",linf)
    
    print("l2 order:",l2[:-1]/l2[1:])
    print("linf order:",linf[:-1]/linf[1:])
    l2order = np.log2(l2[:-1]/l2[1:])
    linforder = np.log2(linf[:-1]/linf[1:])
    print("l2 order:",l2order)
    print("linf order:",linforder)
    
    print(64,"    & & $",end=' ')
    mag2Order = np.floor(np.log10(l2[0]))
    maginfOrder = np.floor(np.log10(linf[0]))
    print(str.format('{0:.2f}',l2[0]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
    print("}$ &       & $",str.format('{0:.2f}',linf[0]*10**-maginfOrder),"\\cdot 10^{", str.format('{0:n}',maginfOrder),end=' ')
    print("}$ &  \\\\")
    print("\\hline")
    for i in range(nconvpts-1):
        n=2**(i+7)
        mag2Order = np.floor(np.log10(l2[i+1]))
        maginfOrder = np.floor(np.log10(linf[i+1]))
        print(n,"    & & $",end=' ')
        print(str.format('{0:.2f}',l2[i+1]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
        print("}$ & ",str.format('{0:.2f}',l2order[i])," & $",end=' ')
        print(str.format('{0:.2f}',linf[i+1]*10**-maginfOrder),"\\cdot 10^{", str.format('{0:n}',maginfOrder),end=' ')
        print("}$ & ",str.format('{0:.2f}',linforder[i])," \\\\")
        print("\\hline")

def Phi_adv(r,theta, omega, xc, yc):
    return omega * (r*r/2 - r*np.sin(theta)*yc - r * np.cos(theta)*xc)

def initConditions2(r,theta):
    a=6
    factor = pi/a/2
    x=r*np.cos(theta)
    y=r*np.sin(theta)
    R1=np.sqrt((x+7)**2+8*y**2)
    R2=np.sqrt(4*(x+7)**2+0.5*y**2)
    result=0.0
    if (R1<=a):
        result+=0.5*np.cos(R1*factor)**4
    if (R2<=a):
        result+=0.5*np.cos(R2*factor)**4
    return result


def initConditions1(r,theta):
    a=2
    factor = pi/a/2
    r=np.sqrt((r-6)**2+8*(theta-pi)**2)
    
    if (r<=a):
        return np.cos(r*factor)**4
    else:
        return 0.0

@pytest.mark.serial
@pytest.mark.parametrize( "initConditions,xc,yc", [(initConditions1,0,0),(initConditions1,1,1),(initConditions2,0,0)] )
def test_poloidalAdvection_constantAdv(initConditions, xc, yc):
    initConds = np.vectorize(initConditions, otypes=[np.float])
    
    npts = [32,32]
    dt=0.01
    N=10

    omega = 1
    
    nconvpts = 5
    
    l2 = np.ndarray(nconvpts)
    linf = np.ndarray(nconvpts)
    
    constants = get_constants('testSetups/iota0.json')
    
    for i in range(nconvpts):
        print(npts)
        eta_vals = [np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,2*pi,npts[0],endpoint=False),
                    np.linspace(0,1,4),np.linspace(0,1,4)]
        
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
        
        polAdv = PoloidalAdvection(eta_vals, bsplines[::-1],constants,True)
        
        phi = Spline2D(bsplines[1],bsplines[0])
        phiVals = np.empty([npts[1],npts[0]])
        phiVals[:] = Phi_adv(eta_vals[0],np.atleast_2d(eta_vals[1]).T, omega, xc, yc)
        interp = SplineInterpolator2D(bsplines[1],bsplines[0])
        
        interp.compute_interpolant(phiVals,phi)
        
        f_vals[:,:] = initConds(eta_vals[0],np.atleast_2d(eta_vals[1]).T)

        x0 = polAdv._points[1] * np.cos(polAdv._shapedQ)
        y0 = polAdv._points[1] * np.sin(polAdv._shapedQ)

        x = xc + (x0 - xc) * np.cos(omega * -dt * N) - (y0 - yc) * np.sin(omega * -dt * N)
        y = yc + (x0 - xc) * np.sin(omega * -dt * N) + (y0 - yc) * np.cos(omega * -dt * N)

        finalPts = ( np.ndarray([npts[1],npts[0]]), np.ndarray([npts[1],npts[0]]))
        finalPts[0][:] = np.mod(np.arctan2(y, x), 2 * np.pi)
        finalPts[1][:] = np.sqrt(x * x + y * y)
        final_f_vals[:,:] = initConds(finalPts[1],finalPts[0])
        
        x = xc + (x0 - xc) * np.cos(omega * -dt) - (y0 - yc) * np.sin(omega * -dt)
        y = yc + (x0 - xc) * np.sin(omega * -dt) + (y0 - yc) * np.cos(omega * -dt)

        endPts = ( np.ndarray([npts[1],npts[0]]), np.ndarray([npts[1],npts[0]]))
        endPts[0][:] = np.mod(np.arctan2(y, x), 2 * np.pi)
        endPts[1][:] = np.sqrt(x * x + y * y)
        
        polAdv.allow_tests()  # Vectorises exact_step
        for n in range(N):
            polAdv.exact_step(f_vals[:,:],endPts,v)

        linf[i]=np.linalg.norm((f_vals-final_f_vals).flatten(),np.inf)
        l2[i]=np.sqrt(trapz(trapz((f_vals-final_f_vals)**2,eta_grids[1],axis=0)*eta_grids[0],eta_grids[0]))
        
        print("l2:",l2[i])
        print("linf:",linf[i])
        npts[0]*=2
        npts[1]*=2
        dt/=2
        N*=2
    
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
    print("}$ &  \\\\")
    print("\\hline")
    for i in range(nconvpts-1):
        n=2**(i+6)
        mag2Order = np.floor(np.log10(l2[i+1]))
        maginfOrder = np.floor(np.log10(linf[i+1]))
        print(n," & & ",n,"    & & $",end=' ')
        print(str.format('{0:.2f}',l2[i+1]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
        print("}$ & ",str.format('{0:.2f}',l2order[i])," & $",end=' ')
        print(str.format('{0:.2f}',linf[i+1]*10**-maginfOrder),"\\cdot 10^{", str.format('{0:n}',maginfOrder),end=' ')
        print("}$ & ",str.format('{0:.2f}',linforder[i])," \\\\")
        print("\\hline")

@pytest.mark.serial
def test_poloidalAdvection_constantAdv_dt():
    dt=1

    omega = 4
    xc = 1
    yc = 1
    
    nconvpts = 5
    
    l2 = np.ndarray(nconvpts)
    linf = np.ndarray(nconvpts)
    
    constants = get_constants('testSetups/iota0.json')
    
    npts = [200,200]
    eta_vals = [np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,2*pi,npts[0],endpoint=False),
                np.linspace(0,1,4),np.linspace(0,1,4)]
    
    initConds = np.vectorize(initConditions1, otypes=[np.float])
    
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
        
        polAdv = PoloidalAdvection(eta_vals, bsplines[::-1],constants,True)
        
        phi = Spline2D(bsplines[1],bsplines[0])
        phiVals = np.empty([npts[1],npts[0]])
        phiVals[:] = Phi_adv(eta_vals[0],np.atleast_2d(eta_vals[1]).T, omega, xc, yc)
        interp = SplineInterpolator2D(bsplines[1],bsplines[0])
        
        interp.compute_interpolant(phiVals,phi)
    
        f_vals[:,:] = initConds(eta_vals[0],np.atleast_2d(eta_vals[1]).T)
        
        x0 = polAdv._points[1] * np.cos(polAdv._shapedQ)
        y0 = polAdv._points[1] * np.sin(polAdv._shapedQ)
        
        x = xc + (x0 - xc) * np.cos(omega * -dt * N) - (y0 - yc) * np.sin(omega * -dt * N)
        y = yc + (x0 - xc) * np.sin(omega * -dt * N) + (y0 - yc) * np.cos(omega * -dt * N)

        finalPts = ( np.ndarray([npts[1],npts[0]]), np.ndarray([npts[1],npts[0]]))
        finalPts[0][:] = np.mod(np.arctan2(y, x), 2 * np.pi)
        finalPts[1][:] = np.sqrt(x * x + y * y)
        final_f_vals[:,:] = initConds(finalPts[1],finalPts[0])
        
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
    print("}$ &  \\\\")
    print("\\hline")
    for i in range(nconvpts-1):
        n = 20*2**i
        dt = 0.05/2**i
        mag2Order = np.floor(np.log10(l2[i+1]))
        maginfOrder = np.floor(np.log10(linf[i+1]))
        print(n," & & ",dt,"    & & $",end=' ')
        print(str.format('{0:.2f}',l2[i+1]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
        print("}$ & ",str.format('{0:.2f}',l2Order[i])," & $",end=' ')
        print(str.format('{0:.2f}',linf[i+1]*10**-maginfOrder),"\\cdot 10^{", str.format('{0:n}',maginfOrder),end=' ')
        print("}$ & ",str.format('{0:.2f}',linfOrder[i])," \\\\")
        print("\\hline")

def initConditionsFlux(theta,z):
    a=4
    factor = pi/a/2
    r=np.sqrt((z-10)**2+2*(theta-pi)**2)
    if (r<=a):
        return np.cos(r*factor)**6
    else:
        return 0.0

initCondsF = np.vectorize(initConditionsFlux, otypes=[np.float])

def iota0(r = 6.0):
    return np.full_like(r,0.0,dtype=float)

@pytest.mark.serial
def test_fluxAdvection():
    dt=0.1
    zStart=32
    thetaStart=32
    npts = [thetaStart,zStart]
    
    nconvpts = 5
    
    l2 = np.ndarray(nconvpts)
    linf = np.ndarray(nconvpts)
    dts = np.ndarray(nconvpts)
    
    constants = get_constants('testSetups/iota0.json')
    
    for i in range(nconvpts):
        dts[i]=dt
        N = int(1/dt)
        print(npts,dt,N)
                
        v=0
        
        eta_vals = [np.linspace(0,1,4),np.linspace(0,2*pi,npts[0],endpoint=False),
                np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,2,4)]
        
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
        eta_vals[3][0]=c
        
        dz = eta_vals[2][1]-eta_vals[2][0]
        dtheta = iota0()*dz/constants.R0
        
        bz = dz/np.sqrt(dz**2+dtheta**2)
        btheta = dtheta/np.sqrt(dz**2+dtheta**2)
        
        layout = Layout('flux',[1],[0,3,1,2],eta_vals,[0])
        
        fluxAdv = FluxSurfaceAdvection(eta_vals, bsplines, layout, dt, constants, zDegree=3)
        
        f_vals[:,:] = initCondsF(np.atleast_2d(eta_vals[1]).T,eta_vals[2])
        finalPts=[eta_vals[1]-c*N*dt*btheta,eta_vals[2]-c*N*dt*bz]
        final_f_vals = initCondsF(np.atleast_2d(finalPts[0]).T,finalPts[1])
        
        for n in range(N):
            fluxAdv.step(f_vals,0)
        
        linf[i]=np.linalg.norm((f_vals-final_f_vals).flatten(),np.inf)
        l2[i]=np.sqrt(trapz(trapz((f_vals-final_f_vals)**2,eta_vals[1],axis=0),eta_vals[2]))
        
        print(N,"l2:",l2[i])
        print(N,"linf:",linf[i])
        dt/=2
        npts[0]*=2
        npts[1]*=2
    
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
    
    print(thetaStart," & & ",zStart," & & ",dts[0],"    & & $",end=' ')
    mag2Order = np.floor(np.log10(l2[0]))
    maginfOrder = np.floor(np.log10(linf[0]))
    print(str.format('{0:.2f}',l2[0]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
    print("}$ &       & $",str.format('{0:.2f}',linf[0]*10**-maginfOrder),"\\cdot 10^{", str.format('{0:n}',maginfOrder),end=' ')
    print("}$ &  \\\\")
    print("\\hline")
    for i in range(nconvpts-1):
        q = thetaStart*2**(i+1)
        z = zStart*2**(i+1)
        dt = dts[i+1]
        mag2Order = np.floor(np.log10(l2[i+1]))
        maginfOrder = np.floor(np.log10(linf[i+1]))
        print(q," & & ",z," & & ",dt,"    & & $",end=' ')
        print(str.format('{0:.2f}',l2[i+1]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
        print("}$ & ",str.format('{0:.2f}',l2Order[i])," & $",end=' ')
        print(str.format('{0:.2f}',linf[i+1]*10**-maginfOrder),"\\cdot 10^{", str.format('{0:n}',maginfOrder),end=' ')
        print("}$ & ",str.format('{0:.2f}',linfOrder[i])," \\\\")
        print("\\hline")

def iota8(r = 6.0):
    return np.full_like(r,0.8,dtype=float)

@pytest.mark.serial
def test_fluxAdvectionAligned():
    dt=0.1
    zStart=32
    thetaStart=32
    npts = [thetaStart,zStart]
    
    constants = get_constants('testSetups/iota8.json')
    
    nconvpts = 5
    
    l2 = np.ndarray(nconvpts)
    linf = np.ndarray(nconvpts)
    dts = np.ndarray(nconvpts)
    
    for i in range(nconvpts):
        dts[i]=dt
        N = int(1/dt)
        print(npts,dt,N)
                
        v=0
        
        eta_vals = [np.linspace(0,1,4),np.linspace(0,2*pi,npts[0],endpoint=False),
                np.linspace(0,2*pi*constants.R0,npts[1],endpoint=False),np.linspace(0,2,4)]
        
        c=2
        
        f_vals = np.ndarray(npts)
        
        domain    = [ [0,2*pi], [0,2*pi*constants.R0] ]
        nkts      = [n+1                           for n          in npts ]
        breaks    = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
        knots     = [spl.make_knots( b,3,True )    for b          in breaks]
        bsplines  = [spl.BSplines( k,3,True )      for k          in knots]
        eta_grids = [bspl.greville                 for bspl       in bsplines]
        
        eta_vals[1]=eta_grids[0]
        if (eta_grids[1][0]>eta_grids[1][1]):
            eta_vals[2]=np.array([*eta_grids[1][1:], eta_grids[1][0]])
        else:
            eta_vals[2]=eta_grids[1]
        eta_vals[3][0]=c
        
        layout = Layout('flux',[1],[0,3,1,2],eta_vals,[0])
        
        fluxAdv = FluxSurfaceAdvection(eta_vals, bsplines, layout, dt, constants, zDegree=3)
        
        m, n = (5, 4)
        theta = eta_grids[0]
        phi = eta_grids[1]*2*pi/domain[1][1]
        f_vals[:,:] = 0.5 + 0.5 * np.sin( m*theta[:,None] - n*phi[None,:] )
        
        final_f_vals = f_vals.copy()
        
        f_max = np.max(f_vals)
        f_min = np.min(f_vals)
        
        for n in range(N):
            fluxAdv.step(f_vals,0)
            f_max = np.max([f_max,np.max(f_vals)])
            f_min = np.min([f_min,np.min(f_vals)])
        
        linf[i]=np.linalg.norm((f_vals-final_f_vals).flatten(),np.inf)
        l2[i]=np.sqrt(trapz(trapz((f_vals-final_f_vals)**2,eta_vals[1],axis=0),eta_vals[2]))
        
        print(N,"l2:",l2[i])
        print(N,"linf:",linf[i])
        dt/=2
        npts[0]*=2
        npts[1]*=2
    
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
    
    print(thetaStart," & & ",zStart," & & ",dts[0],"    & & $",end=' ')
    mag2Order = np.floor(np.log10(l2[0]))
    maginfOrder = np.floor(np.log10(linf[0]))
    print(str.format('{0:.2f}',l2[0]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
    print("}$ &       & $",str.format('{0:.2f}',linf[0]*10**-maginfOrder),"\\cdot 10^{", str.format('{0:n}',maginfOrder),end=' ')
    print("}$ &  \\\\")
    print("\\hline")
    for i in range(nconvpts-1):
        q = thetaStart*2**(i+1)
        z = zStart*2**(i+1)
        dt = dts[i+1]
        mag2Order = np.floor(np.log10(l2[i+1]))
        maginfOrder = np.floor(np.log10(linf[i+1]))
        print(q," & & ",z," & & ",dt,"    & & $",end=' ')
        print(str.format('{0:.2f}',l2[i+1]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
        print("}$ & ",str.format('{0:.2f}',l2Order[i])," & $",end=' ')
        print(str.format('{0:.2f}',linf[i+1]*10**-maginfOrder),"\\cdot 10^{", str.format('{0:n}',maginfOrder),end=' ')
        print("}$ & ",str.format('{0:.2f}',linfOrder[i])," \\\\")
        print("\\hline")

def Phi(theta,z):
    #return np.cos(z*pi*0.1) + np.sin(theta)
    return np.sin(z*pi*0.1)**2 + np.cos(theta)**2

def dPhi(r,theta,z,btheta,bz):
    #return -np.sin(z*pi*0.1)*pi*0.1*bz + np.cos(theta)*btheta
    return 2*np.sin(z*pi*0.1)*np.cos(z*pi*0.1)*pi*0.1*bz - 2*np.cos(theta)*np.sin(theta)*btheta/r

@pytest.mark.serial
def test_Phi_deriv_dtheta():
    nconvpts = 7
    npts = [128,8,1024]
    
    l2=np.empty(nconvpts)
    linf=np.empty(nconvpts)
    
    constants = get_constants('testSetups/iota8.json')
    
    for i in range(nconvpts):
        breaks_theta = np.linspace(0,2*pi,npts[1]+1)
        spline_theta = spl.BSplines(spl.make_knots(breaks_theta,3,True),3,True)
        breaks_z = np.linspace(0,20,npts[2]+1)
        spline_z = spl.BSplines(spl.make_knots(breaks_z,3,True),3,True)
        
        eta_grid = [np.array([1]), spline_theta.greville, spline_z.greville]
        
        dz = eta_grid[2][1]-eta_grid[2][0]
        dtheta = constants.iota()*dz/constants.R0
        
        r = eta_grid[0]
        
        bz = 1 / np.sqrt(1+(r * constants.iota(r)/constants.R0)**2)
        #bz = dz/np.sqrt(dz**2+dtheta**2)
        btheta = r * constants.iota(r)/constants.R0 / np.sqrt(1+(r * constants.iota(r)/constants.R0)**2)
        # ~ btheta = dtheta/np.sqrt(dz**2+r*dtheta**2)
        
        phiVals = np.empty([npts[2],npts[1]])
        phiVals[:] = Phi(eta_grid[1][None,:],eta_grid[2][:,None])
        
        theLayout = Layout('full',[1,1,1],[0,2,1],eta_grid,[0,0,0])
        
        pGrad = ParallelGradient(spline_theta,eta_grid,theLayout,constants)
        
        approxGrad = np.empty([npts[2],npts[1]])
        pGrad.parallel_gradient(phiVals,0,approxGrad)
        exactGrad = dPhi(eta_grid[0][:,None,None],eta_grid[1][None,:],eta_grid[2][:,None],btheta,bz)
        
        err = approxGrad-exactGrad
        
        l2[i]=np.sqrt(np.trapz(np.trapz(err**2,dx=dz),dx=dtheta))
        linf[i]=np.linalg.norm(err.flatten(),np.inf)
        
        npts[1]*=2
    
    l2Order = np.log2(l2[:-1]/l2[1:])
    linfOrder = np.log2(linf[:-1]/linf[1:])
    
    print(l2)
    print(linf)
    print(l2Order)
    print(linfOrder)
    
    mag2Order = np.floor(np.log10(l2[0]))
    maginfOrder = np.floor(np.log10(linf[0]))
    print(8," & & $",end=' ')
    print(str.format('{0:.2f}',l2[0]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
    print("}$ &       & $",end=' ')
    print(str.format('{0:.2f}',linf[0]*10**-maginfOrder),"\\cdot 10^{", str.format('{0:n}',maginfOrder),end=' ')
    print("}$ &        \\\\")
    print("\\hline")
    for i in range(1,nconvpts):
        n=8*2**i
        mag2Order = np.floor(np.log10(l2[i]))
        maginfOrder = np.floor(np.log10(linf[i]))
        print(n," & & $",end=' ')
        print(str.format('{0:.2f}',l2[i]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
        print("}$ & ",str.format('{0:.2f}',l2Order[i-1])," & $",end=' ')
        print(str.format('{0:.2f}',linf[i]*10**-maginfOrder),"\\cdot 10^{", str.format('{0:n}',maginfOrder),end=' ')
        print("}$ & ",str.format('{0:.2f}',linfOrder[i-1])," \\\\")
        print("\\hline")

@pytest.mark.serial
def test_Phi_deriv_dz():
    nconvpts = 7
    npts = [1,32,32]
    
    l2=np.empty(nconvpts)
    linf=np.empty(nconvpts)
    
    constants = get_constants('testSetups/iota8.json')
    
    for i in range(nconvpts):
        breaks_theta = np.linspace(0,2*pi,npts[1]+1)
        spline_theta = spl.BSplines(spl.make_knots(breaks_theta,3,True),3,True)
        breaks_z = np.linspace(0,20,npts[2]+1)
        spline_z = spl.BSplines(spl.make_knots(breaks_z,3,True),3,True)
        
        eta_grid = [np.array([1]), spline_theta.greville, spline_z.greville]
        
        dz = eta_grid[2][1]-eta_grid[2][0]
        dtheta = constants.iota()*dz/constants.R0
        
        r = eta_grid[0]
        
        bz = 1 / np.sqrt(1+(r * constants.iota(r)/constants.R0)**2)
        btheta = r * constants.iota(r)/constants.R0 / np.sqrt(1+(r * constants.iota(r)/constants.R0)**2)
        #bz = dz/np.sqrt(dz**2+dtheta**2)
        #btheta = dtheta/np.sqrt(dz**2+r*dtheta**2)
        
        phiVals = np.empty([npts[2],npts[1]])
        phiVals[:] = Phi(eta_grid[1][None,:],eta_grid[2][:,None])
        
        theLayout = Layout('full',[1,1,1],[0,2,1],eta_grid,[0,0,0])
        
        pGrad = ParallelGradient(spline_theta,eta_grid,theLayout,constants,order = 3)
        
        approxGrad = np.empty([npts[2],npts[1]])
        pGrad.parallel_gradient(phiVals,0,approxGrad)
        exactGrad = dPhi(eta_grid[0][:,None,None],eta_grid[1][None,:],eta_grid[2][:,None],btheta,bz)
        
        err = approxGrad-exactGrad
        
        l2[i]=np.sqrt(np.trapz(np.trapz(err**2,dx=dz),dx=dtheta))
        linf[i]=np.linalg.norm(err.flatten(),np.inf)
        
        npts[2]*=2
    
    l2Order = np.log2(l2[:-1]/l2[1:])
    linfOrder = np.log2(linf[:-1]/linf[1:])
    
    print(l2)
    print(linf)
    print(l2Order)
    print(linfOrder)
    
    mag2Order = np.floor(np.log10(l2[0]))
    maginfOrder = np.floor(np.log10(linf[0]))
    print(8," & & $",end=' ')
    print(str.format('{0:.2f}',l2[0]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
    print("}$ &       & $",end=' ')
    print(str.format('{0:.2f}',linf[0]*10**-maginfOrder),"\\cdot 10^{", str.format('{0:n}',maginfOrder),end=' ')
    print("}$ &        \\\\")
    print("\\hline")
    for i in range(1,nconvpts):
        n=8*2**i
        mag2Order = np.floor(np.log10(l2[i]))
        maginfOrder = np.floor(np.log10(linf[i]))
        print(n," & & $",end=' ')
        print(str.format('{0:.2f}',l2[i]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
        print("}$ & ",str.format('{0:.2f}',l2Order[i-1])," & $",end=' ')
        print(str.format('{0:.2f}',linf[i]*10**-maginfOrder),"\\cdot 10^{", str.format('{0:n}',maginfOrder),end=' ')
        print("}$ & ",str.format('{0:.2f}',linfOrder[i-1])," \\\\")
        print("\\hline")

