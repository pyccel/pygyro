from mpi4py                 import MPI
import pytest
import numpy                as np
import matplotlib.pyplot    as plt

from ..initialisation.setups        import setupCylindricalGrid
from ..initialisation.initialiser   import fEq
from .advection                     import *
from ..                             import splines as spl
from ..initialisation               import constants
"""
def gauss(x):
    return np.exp(-x**2/4)

@pytest.mark.serial
def test_vParallelAdvection():
    npts = 16
    
    nconvpts = 10
    
    N=100
    
    dt=0.005
    c=2.0
    
    l2 = [[npts*(2**(i+1)),0] for i in range(nconvpts)]
    linf = [[npts*(2**(i+1)),0] for i in range(nconvpts)]
    
    for j in range(nconvpts):
        print(j)
        npts*=2
        f = np.empty(npts)
        
        nkts      = npts-2
        breaks    = np.linspace( -5, 5, num=nkts )
        knots     = spl.make_knots( breaks,3,False )
        spline    = spl.BSplines( knots,3,False )
        x         = spline.greville
        
        r = 4
        fEdge = fEq(r,x[0])
        assert(fEq(r,x[0])==fEq(r,x[-1]))
        
        f = gauss(x)+fEdge - gauss(x[0])
        
        vParAdv = vParallelAdvection([0,0,0,x], spline, lambda r,v : fEdge)
        
        for i in range(N):
            vParAdv.step(f,dt,c,r)
        
        fEnd = np.empty(npts)
        
        for i in range(npts):
            if ((x[i]-c*dt*N)<x[0]):
                #fEnd[i]=fEq(r,(x[i]-c*dt*N))
                fEnd[i]=fEdge
            else:
                fEnd[i]=fEdge+gauss(x[i]-c*dt*N) - gauss(x[0])
        
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
    assert(np.abs(np.mean(l2Order)-1)<0.2)
    assert(np.abs(np.mean(linfOrder)-1)<0.2)
"""

def Phi(r,theta):
    return - 5 * r**2 + np.sin(theta)

def initConds(r,theta):
    return np.exp(-(theta-pi)**2 - (r-7)**2)

@pytest.mark.serial
def test_poloidalAdvection_constantAdv_dtheta():
    #npts = [100,8]
    npts = [16,500]
    
    nconvpts = 5
    
    l2 = np.ndarray(nconvpts)
    linf = np.ndarray(nconvpts)
    
    for i in range(nconvpts):
        npts[0]*=2
        print(npts)
        eta_vals = [np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,2*pi,npts[0],endpoint=False),
                    np.linspace(0,1,4),np.linspace(0,1,4)]
        
        N = 100
        dt=0.01
        
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
        
        polAdv = poloidalAdvection(eta_vals, bsplines[::-1])
        
        phi = Spline2D(bsplines[1],bsplines[0])
        phiVals = np.empty([npts[1],npts[0]])
        phiVals[:] = Phi(eta_vals[0],np.atleast_2d(eta_vals[1]).T)
        interp = SplineInterpolator2D(bsplines[1],bsplines[0])
        
        interp.compute_interpolant(phiVals,phi)
        
        #f_vals[:,:] = np.exp(-np.atleast_2d((eta_vals[1]-pi)**2).T - (eta_vals[0]-7)**2)# + fEq(0.1,v)
        f_vals[:,:] = initConds(eta_vals[0],np.atleast_2d(eta_vals[1]).T)
        #f_vals[:,:] = phiVals
        finalPts = ( np.ndarray([npts[1],npts[0]]), np.ndarray([npts[1],npts[0]]))
        finalPts[0][:] = np.mod(polAdv._shapedQ   -     10*dt*N/constants.B0,2*pi)
        finalPts[1][:] = np.sqrt(polAdv._points[1]**2-np.sin(polAdv._shapedQ)/5/constants.B0 \
                        + np.sin(finalPts[0])/5/constants.B0)
        final_f_vals[:,:] = initConds(finalPts[1],finalPts[0])
        #final_f_vals[:,:] = phiVals
        
        endPts = ( np.ndarray([npts[1],npts[0]]), np.ndarray([npts[1],npts[0]]))
        endPts[0][:] = polAdv._shapedQ   -     10*dt/constants.B0
        endPts[1][:] = np.sqrt(polAdv._points[1]**2-np.sin(polAdv._shapedQ)/5/constants.B0 \
                        + np.sin(endPts[0])/5/constants.B0)
        
        for n in range(N):
            polAdv.exact_step(f_vals[:,:],endPts,v)
            #polAdv.step(f_vals[:,:],dt,phi,v)
        
        #~ fig = plt.figure()
        #~ ax = plt.subplot(111, projection='polar')
        #~ colorbarax2 = fig.add_axes([0.85, 0.1, 0.03, 0.8],)
        
        #~ plotParams = {'cmap':"jet"}
        
        #~ line1 = ax.contourf(eta_vals[1],eta_vals[0],f_vals[:,:].T,20,**plotParams)
        
        #~ fig.colorbar(line1, cax = colorbarax2)
        #~ plt.show()
        
        #~ fig = plt.figure()
        #~ ax = plt.subplot(111, projection='polar')
        #~ colorbarax2 = fig.add_axes([0.85, 0.1, 0.03, 0.8],)
        
        #~ plotParams = {'cmap':"jet"}
        
        #~ line1 = ax.contourf(eta_vals[1],eta_vals[0],final_f_vals[:,:].T,20,**plotParams)
        
        #~ fig.colorbar(line1, cax = colorbarax2)
        #~ plt.show()
        
        #~ print(f_vals-final_f_vals)
        
        l2[i]=np.linalg.norm((f_vals-final_f_vals).flatten(),2)
        linf[i]=np.linalg.norm((f_vals-final_f_vals).flatten(),np.inf)
    
    print("l2:",l2)
    print("linf:",linf)
    
    print("l2 order:",l2[:-1]/l2[1:])
    print("linf order:",linf[:-1]/linf[1:])
    print("l2 order:",np.log2(l2[:-1]/l2[1:]))
    print("linf order:",np.log2(linf[:-1]/linf[1:]))

@pytest.mark.serial
def test_poloidalAdvection_constantAdv_dr():
    npts = [500,16]
    
    nconvpts = 5
    
    l2 = np.ndarray(nconvpts)
    linf = np.ndarray(nconvpts)
    
    for i in range(nconvpts):
        npts[1]*=2
        print(npts)
        eta_vals = [np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,2*pi,npts[0],endpoint=False),
                    np.linspace(0,1,4),np.linspace(0,1,4)]
        
        N = 100
        dt=0.01
        
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
        
        polAdv = poloidalAdvection(eta_vals, bsplines[::-1])
        
        phi = Spline2D(bsplines[1],bsplines[0])
        phiVals = np.empty([npts[1],npts[0]])
        phiVals[:] = Phi(eta_vals[0],np.atleast_2d(eta_vals[1]).T)
        interp = SplineInterpolator2D(bsplines[1],bsplines[0])
        
        interp.compute_interpolant(phiVals,phi)
        
        #f_vals[:,:] = np.exp(-np.atleast_2d((eta_vals[1]-pi)**2).T - (eta_vals[0]-7)**2)# + fEq(0.1,v)
        f_vals[:,:] = initConds(eta_vals[0],np.atleast_2d(eta_vals[1]).T)
        #f_vals[:,:] = phiVals
        finalPts = ( np.ndarray([npts[1],npts[0]]), np.ndarray([npts[1],npts[0]]))
        finalPts[0][:] = np.mod(polAdv._shapedQ   -     10*dt*N/constants.B0,2*pi)
        finalPts[1][:] = np.sqrt(polAdv._points[1]**2-np.sin(polAdv._shapedQ)/5/constants.B0 \
                        + np.sin(finalPts[0])/5/constants.B0)
        final_f_vals[:,:] = initConds(finalPts[1],finalPts[0])
        #final_f_vals[:,:] = phiVals
        
        endPts = ( np.ndarray([npts[1],npts[0]]), np.ndarray([npts[1],npts[0]]))
        endPts[0][:] = polAdv._shapedQ   -     10*dt/constants.B0
        endPts[1][:] = np.sqrt(polAdv._points[1]**2-np.sin(polAdv._shapedQ)/5/constants.B0 \
                        + np.sin(endPts[0])/5/constants.B0)
        
        for n in range(N):
            polAdv.exact_step(f_vals[:,:],endPts,v)
            #polAdv.step(f_vals[:,:],dt,phi,v)
        
        #~ fig = plt.figure()
        #~ ax = plt.subplot(111, projection='polar')
        #~ colorbarax2 = fig.add_axes([0.85, 0.1, 0.03, 0.8],)
        
        #~ plotParams = {'cmap':"jet"}
        
        #~ line1 = ax.contourf(eta_vals[1],eta_vals[0],f_vals[:,:].T,20,**plotParams)
        
        #~ fig.colorbar(line1, cax = colorbarax2)
        #~ plt.show()
        
        #~ print(f_vals-final_f_vals)
        
        l2[i]=np.linalg.norm((f_vals-final_f_vals).flatten(),2)
        linf[i]=np.linalg.norm((f_vals-final_f_vals).flatten(),np.inf)
    
    print("l2:",l2)
    print("linf:",linf)
    
    print("l2 order:",l2[:-1]/l2[1:])
    print("linf order:",linf[:-1]/linf[1:])
    print("l2 order:",np.log2(l2[:-1]/l2[1:]))
    print("linf order:",np.log2(linf[:-1]/linf[1:]))

@pytest.mark.serial
def test_poloidalAdvection_constantAdv():
    npts = [16,16]
    
    nconvpts = 5
    
    l2 = np.ndarray(nconvpts)
    linf = np.ndarray(nconvpts)
    
    for i in range(nconvpts):
        npts[0]*=2
        npts[1]*=2
        print(npts)
        eta_vals = [np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,2*pi,npts[0],endpoint=False),
                    np.linspace(0,1,4),np.linspace(0,1,4)]
        
        N = 100
        dt=0.01
        
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
        
        polAdv = poloidalAdvection(eta_vals, bsplines[::-1])
        
        phi = Spline2D(bsplines[1],bsplines[0])
        phiVals = np.empty([npts[1],npts[0]])
        phiVals[:] = Phi(eta_vals[0],np.atleast_2d(eta_vals[1]).T)
        interp = SplineInterpolator2D(bsplines[1],bsplines[0])
        
        interp.compute_interpolant(phiVals,phi)
        
        #f_vals[:,:] = np.exp(-np.atleast_2d((eta_vals[1]-pi)**2).T - (eta_vals[0]-7)**2)# + fEq(0.1,v)
        f_vals[:,:] = initConds(eta_vals[0],np.atleast_2d(eta_vals[1]).T)
        #f_vals[:,:] = phiVals
        finalPts = ( np.ndarray([npts[1],npts[0]]), np.ndarray([npts[1],npts[0]]))
        finalPts[0][:] = np.mod(polAdv._shapedQ   -     10*dt*N/constants.B0,2*pi)
        finalPts[1][:] = np.sqrt(polAdv._points[1]**2-np.sin(polAdv._shapedQ)/5/constants.B0 \
                        + np.sin(finalPts[0])/5/constants.B0)
        final_f_vals[:,:] = initConds(finalPts[1],finalPts[0])
        #final_f_vals[:,:] = phiVals
        
        endPts = ( np.ndarray([npts[1],npts[0]]), np.ndarray([npts[1],npts[0]]))
        endPts[0][:] = polAdv._shapedQ   -     10*dt/constants.B0
        endPts[1][:] = np.sqrt(polAdv._points[1]**2-np.sin(polAdv._shapedQ)/5/constants.B0 \
                        + np.sin(endPts[0])/5/constants.B0)
        
        for n in range(N):
            polAdv.exact_step(f_vals[:,:],endPts,v)
            #polAdv.step(f_vals[:,:],dt,phi,v)
        
        #~ fig = plt.figure()
        #~ ax = plt.subplot(111, projection='polar')
        #~ colorbarax2 = fig.add_axes([0.85, 0.1, 0.03, 0.8],)
        
        #~ plotParams = {'cmap':"jet"}
        
        #~ line1 = ax.contourf(eta_vals[1],eta_vals[0],f_vals[:,:].T,20,**plotParams)
        
        #~ fig.colorbar(line1, cax = colorbarax2)
        #~ plt.show()
        
        #~ print(f_vals-final_f_vals)
        
        l2[i]=np.linalg.norm((f_vals-final_f_vals).flatten(),2)
        linf[i]=np.linalg.norm((f_vals-final_f_vals).flatten(),np.inf)
    
    print("l2:",l2)
    print("linf:",linf)
    
    print("l2 order:",l2[:-1]/l2[1:])
    print("linf order:",linf[:-1]/linf[1:])
    print("l2 order:",np.log2(l2[:-1]/l2[1:]))
    print("linf order:",np.log2(linf[:-1]/linf[1:]))
