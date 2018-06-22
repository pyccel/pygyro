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

"""
def initConditions(r,theta):
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
"""

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
def test_poloidalAdvection_constantAdv():
    npts = [128,128]
    dt=0.0025
    
    nconvpts = 4
    
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
    for i in range(nconvpts-1):
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
    for i in range(nconvpts-1):
        n = 20*2**i
        dt = 0.05/2**i
        mag2Order = np.floor(np.log10(l2[i+1]))
        maginfOrder = np.floor(np.log10(linf[i+1]))
        print(n," & & ",dt,"    & & $",end=' ')
        print(str.format('{0:.2f}',l2[i+1]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
        print("}$ & ",str.format('{0:.2f}',l2Order[i])," & $",end=' ')
        print(str.format('{0:.2f}',linf[i+1]*10**-maginfOrder),"\\cdot 10^{", str.format('{0:n}',maginfOrder),end=' ')
        print("}$ & ",str.format('{0:.2f}',linfOrder[i])," \\ \\")

def initConditionsFlux(theta,z):
    a=4
    factor = pi/a/2
    r=np.sqrt((z-10)**2+2*(theta-pi)**2)
    if (r<=4):
        return np.cos(r*factor)**6
    else:
        return 0.0

initCondsF = np.vectorize(initConditionsFlux, otypes=[np.float])

def iota(r = 6.0):
    return np.full_like(r,0.8,dtype=float)

@pytest.mark.serial
def test_fluxAdvection_dtheta():
    dt=0.2
    zStart=16
    thetaStart=16
    npts = [thetaStart,zStart]
    
    nconvpts = 5
    
    l2 = np.ndarray(nconvpts)
    linf = np.ndarray(nconvpts)
    dts = np.ndarray(nconvpts)
    
    for i in range(nconvpts):
        dt/=2
        npts[0]*=2
        npts[1]*=2
        dts[i]=dt
        N = int(1/dt)
        print(npts,dt,N)
                
        v=0
        
        eta_vals = [np.linspace(0,1,4),np.linspace(0,2*pi,npts[0],endpoint=False),
                np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,1,4)]
        
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
        
        dz = eta_vals[2][1]-eta_vals[2][0]
        dtheta = iota()*dz/constants.R0
        
        bz = dz/np.sqrt(dz**2+dtheta**2)
        btheta = dtheta/np.sqrt(dz**2+dtheta**2)
        
        fluxAdv = FluxSurfaceAdvection(eta_vals, bsplines, iota)
        
        f_vals[:,:] = initCondsF(np.atleast_2d(eta_vals[1]).T,eta_vals[2])
        finalPts=[eta_vals[1]-c*N*dt*btheta,eta_vals[2]-c*N*dt*bz]
        final_f_vals = initCondsF(np.atleast_2d(finalPts[0]).T,finalPts[1])
        
        for n in range(N):
            print(n)
            fluxAdv.step(f_vals,dt,c)
        
        linf[i]=np.linalg.norm((f_vals-final_f_vals).flatten(),np.inf)
        l2[i]=np.sqrt(trapz(trapz((f_vals-final_f_vals)**2,eta_vals[1],axis=0),eta_vals[2]))
        
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
        print(q," & & ",z," & & ",dts[i],"    & & $",end=' ')
        print(str.format('{0:.2f}',l2[i+1]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
        print("}$ & ",str.format('{0:.2f}',l2Order[i])," & $",end=' ')
        print(str.format('{0:.2f}',linf[i+1]*10**-maginfOrder),"\\cdot 10^{", str.format('{0:n}',maginfOrder),end=' ')
        print("}$ & ",str.format('{0:.2f}',linfOrder[i])," \\\\")
        print("\\hline")

def Phi(theta,z):
    #return np.cos(z*pi*0.1) + np.sin(theta)
    return np.sin(z*pi*0.1)**2 + np.cos(theta)**2

def dPhi(theta,z,btheta,bz):
    #return -np.sin(z*pi*0.1)*pi*0.1*bz + np.cos(theta)*btheta
    return 2*np.sin(z*pi*0.1)*np.cos(z*pi*0.1)*pi*0.1*bz - 2*np.cos(theta)*np.sin(theta)*btheta

def iota(r = 6.0):
    return np.full_like(r,0.8,dtype=float)

def test_Phi_deriv_dtheta():
    nconvpts = 7
    npts = [128,8,1024]
    
    l2=np.empty(nconvpts)
    linf=np.empty(nconvpts)
    
    for i in range(nconvpts):
        breaks_theta = np.linspace(0,2*pi,npts[1]+1)
        spline_theta = spl.BSplines(spl.make_knots(breaks_theta,3,True),3,True)
        breaks_z = np.linspace(0,20,npts[2]+1)
        spline_z = spl.BSplines(spl.make_knots(breaks_z,3,True),3,True)
        
        eta_grid = [[1], spline_theta.greville, spline_z.greville]
        
        dz = eta_grid[2][1]-eta_grid[2][0]
        dtheta = iota()*dz/constants.R0
        
        bz = dz/np.sqrt(dz**2+dtheta**2)
        btheta = dtheta/np.sqrt(dz**2+dtheta**2)
        
        phiVals = np.empty([npts[1],npts[2]])
        phiVals[:] = Phi(np.atleast_2d(eta_grid[1]).T,eta_grid[2])
        
        pGrad = ParallelGradient(spline_theta,eta_grid,iota)
        
        approxGrad = pGrad.parallel_gradient(phiVals,0)
        exactGrad = dPhi(np.atleast_2d(eta_grid[1]).T,eta_grid[2],btheta,bz)
        
        print(np.linalg.norm(approxGrad-exactGrad,'fro'))
        print(np.linalg.norm((approxGrad-exactGrad).flatten(),np.inf))
        
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

def test_Phi_deriv_dz():
    nconvpts = 7
    npts = [128,1024,8]
    
    l2=np.empty(nconvpts)
    linf=np.empty(nconvpts)
    
    for i in range(nconvpts):
        breaks_theta = np.linspace(0,2*pi,npts[1]+1)
        spline_theta = spl.BSplines(spl.make_knots(breaks_theta,3,True),3,True)
        breaks_z = np.linspace(0,20,npts[2]+1)
        spline_z = spl.BSplines(spl.make_knots(breaks_z,3,True),3,True)
        
        eta_grid = [[1], spline_theta.greville, spline_z.greville]
        
        dz = eta_grid[2][1]-eta_grid[2][0]
        dtheta = iota()*dz/constants.R0
        
        bz = dz/np.sqrt(dz**2+dtheta**2)
        btheta = dtheta/np.sqrt(dz**2+dtheta**2)
        
        phiVals = np.empty([npts[1],npts[2]])
        phiVals[:] = Phi(np.atleast_2d(eta_grid[1]).T,eta_grid[2])
        
        pGrad = ParallelGradient(spline_theta,eta_grid,iota)
        
        approxGrad = pGrad.parallel_gradient(phiVals,0)
        exactGrad = dPhi(np.atleast_2d(eta_grid[1]).T,eta_grid[2],btheta,bz)
        
        print(np.linalg.norm(approxGrad-exactGrad,'fro'))
        print(np.linalg.norm((approxGrad-exactGrad).flatten(),np.inf))
        
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
