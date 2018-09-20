import numpy            as np
from numba              import njit
from numba.pycc         import CC
from scipy.interpolate  import splev, bisplev
from math               import pi
import sys
sys.path.insert(0,'pygyro')

from initialisation               import constants
from splines.spline_eval_funcs    import eval_spline_2d_cross, eval_spline_2d_scalar,eval_spline_1d_scalar, eval_spline_1d_vector

cc = CC('accelerated_advection_steps')

@cc.export('n0', 'f8(f8)')
@njit
def n0(r):
    return constants.CN0*np.exp(-constants.kN0*constants.deltaRN0*np.tanh((r-constants.rp)/constants.deltaRN0))

@cc.export('Ti', 'f8(f8)')
@njit
def Ti(r):
    return constants.CTi*np.exp(-constants.kTi*constants.deltaRTi*np.tanh((r-constants.rp)/constants.deltaRTi))

@cc.export('fEq', 'f8(f8,f8)')
@njit
def fEq(r,vPar):
    return n0(r)*np.exp(-0.5*vPar*vPar/Ti(r))/np.sqrt(2*pi*Ti(r))

@cc.export('PoloidalAdvectionStepExpl', '(f8[:,:],f8,f8,f8[:],f8[:],f8[:,:], \
                                          i8[:],f8[:],f8[:],f8[:,:],i4,i4, \
                                          f8[:],f8[:],f8[:,:],i4,i4,b1)')
def PoloidalAdvectionStepExpl( f: np.ndarray, dt: float, v: float,
                        rPts: np.ndarray, qPts: np.ndarray, qTPts: np.ndarray,
                        nPts: np.ndarray, kts1Phi: np.ndarray, kts2Phi: np.ndarray,
                        coeffsPhi: np.ndarray, deg1Phi: int, deg2Phi: int,
                        kts1Pol: np.ndarray, kts2Pol: np.ndarray,
                        coeffsPol: np.ndarray, deg1Pol: int, deg2Pol: int,
                        nulBound: bool = False ):
    """
    Carry out an advection step for the poloidal advection

    Parameters
    ----------
    f: array_like
        The current value of the function at the nodes.
        The result will be stored here
    
    dt: float
        Time-step
    
    phi: Spline2D
        Advection parameter d_tf + {phi,f}=0
    
    r: float
        The parallel velocity coordinate
    
    """
    
    multFactor = dt/constants.B0
    
    drPhi_0 = eval_spline_2d_cross(qPts,rPts, kts1Phi, deg1Phi, kts2Phi, deg2Phi, coeffsPhi, 0,1)/rPts
    dthetaPhi_0 = eval_spline_2d_cross(qPts,rPts, kts1Phi, deg1Phi, kts2Phi, deg2Phi, coeffsPhi, 1,0)/rPts
    
    # Step one of Heun method
    # x' = x^n + f(x^n)
    endPts_k1_q = qTPts   -     drPhi_0*multFactor
    endPts_k1_r = rPts + dthetaPhi_0*multFactor
    
    drPhi_k = np.empty_like(drPhi_0)
    dthetaPhi_k = np.empty_like(dthetaPhi_0)
    
    multFactor*=0.5
    
    for i in range(nPts[0]):
        for j in range(nPts[1]):
            # Handle theta boundary conditions
            while (endPts_k1_q[i,j]<0):
                endPts_k1_q[i,j]+=2*pi
            while (endPts_k1_q[i,j]>2*pi):
                endPts_k1_q[i,j]-=2*pi
            
            if (not (endPts_k1_r[i,j]<rPts[0] or 
                     endPts_k1_r[i,j]>rPts[-1])):
                # Add the new value of phi to the derivatives
                # x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
                #                               ^^^^^^^^^^^^^^^
                drPhi_k[i,j]     = eval_spline_2d_scalar(endPts_k1_q[i,j],endPts_k1_r[i,j],
                                                        kts1Phi, deg1Phi, kts2Phi, deg2Phi,
                                                        coeffsPhi,0,1) \
                                    / endPts_k1_r[i,j]
                dthetaPhi_k[i,j] = eval_spline_2d_scalar(endPts_k1_q[i,j],endPts_k1_r[i,j],
                                                        kts1Phi, deg1Phi, kts2Phi, deg2Phi,
                                                        coeffsPhi,1,0) \
                                    / endPts_k1_r[i,j]
            else:
                drPhi_k[i,j]     = 0.0
                dthetaPhi_k[i,j] = 0.0
    
    # Step two of Heun method
    # x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
    endPts_k2 = ( np.mod(qTPts   - (drPhi_0     + drPhi_k)*multFactor,2*pi),
                  rPts + (dthetaPhi_0 + dthetaPhi_k)*multFactor )
    
    #~ return endPts_k2
    
    # Find value at the determined point
    if (nulBound):
        for i,theta in enumerate(qPts):
            for j,r in enumerate(rPts):
                if (endPts_k2[1][i,j]<rPts[0]):
                    f[i,j]=0.0
                elif (endPts_k2[1][i,j]>rPts[-1]):
                    f[i,j]=0.0
                else:
                    while (endPts_k2[0][i,j]>2*pi):
                        endPts_k2[0][i,j]-=2*pi
                    while (endPts_k2[0][i,j]<0):
                        endPts_k2[0][i,j]+=2*pi
                    f[i,j]=eval_spline_2d_scalar(endPts_k2[0][i,j],endPts_k2[1][i,j], kts1Pol, deg1Pol, kts2Pol, deg2Pol, coeffsPol)
    else:
        for i,theta in enumerate(qPts):
            for j,r in enumerate(rPts):
                if (endPts_k2[1][i,j]<rPts[0]):
                    f[i,j]=fEq(rPts[0],v)
                elif (endPts_k2[1][i,j]>rPts[-1]):
                    f[i,j]=fEq(endPts_k2[1][i,j],v)
                else:
                    while (endPts_k2[0][i,j]>2*pi):
                        endPts_k2[0][i,j]-=2*pi
                    while (endPts_k2[0][i,j]<0):
                        endPts_k2[0][i,j]+=2*pi
                    f[i,j]=eval_spline_2d_scalar(endPts_k2[0][i,j],endPts_k2[1][i,j], kts1Pol, deg1Pol, kts2Pol, deg2Pol, coeffsPol)


@cc.export('PoloidalAdvectionStepImpl', '(f8[:,:],f8,f8,f8[:],f8[:],f8[:,:], \
                                          i8[:],f8[:],f8[:],f8[:,:],i4,i4, \
                                          f8[:],f8[:],f8[:,:],i4,i4,f8,b1)')
def PoloidalAdvectionStepImpl( f: np.ndarray, dt: float, v: float,
                        rPts: np.ndarray, qPts: np.ndarray, qTPts: np.ndarray,
                        nPts: list, kts1Phi: np.ndarray, kts2Phi: np.ndarray,
                        coeffsPhi: np.ndarray, deg1Phi: int, deg2Phi: int,
                        kts1Pol: np.ndarray, kts2Pol: np.ndarray,
                        coeffsPol: np.ndarray, deg1Pol: int, deg2Pol: int,
                        tol: float, nulBound: bool = False ):
    """
    Carry out an advection step for the poloidal advection

    Parameters
    ----------
    f: array_like
        The current value of the function at the nodes.
        The result will be stored here
    
    dt: float
        Time-step
    
    phi: Spline2D
        Advection parameter d_tf + {phi,f}=0
    
    r: float
        The parallel velocity coordinate
    
    """
    
    multFactor = dt/constants.B0
    
    drPhi_0 = eval_spline_2d_cross(qPts,rPts, kts1Phi, deg1Phi, kts2Phi, deg2Phi, coeffsPhi,0,1)/rPts
    dthetaPhi_0 = eval_spline_2d_cross(qPts,rPts, kts1Phi, deg1Phi, kts2Phi, deg2Phi, coeffsPhi,1,0)/rPts
    
    # Step one of Heun method
    # x' = x^n + f(x^n)
    endPts_k1_q = qTPts   -     drPhi_0*multFactor
    endPts_k1_r = rPts + dthetaPhi_0*multFactor
    endPts_k2_q = np.empty_like(endPts_k1_r)
    endPts_k2_r = np.empty_like(endPts_k1_r)
    
    drPhi_k = np.empty_like(drPhi_0)
    dthetaPhi_k = np.empty_like(dthetaPhi_0)
    
    multFactor*=0.5
    
    while (True):
        
        for i in range(nPts[0]):
            for j in range(nPts[1]):
                # Handle theta boundary conditions
                while (endPts_k1_q[i,j]<0):
                    endPts_k1_q[i,j]+=2*pi
                while (endPts_k1_q[i,j]>2*pi):
                    endPts_k1_q[i,j]-=2*pi
                
                if (not (endPts_k1_r[i,j]<rPts[0] or 
                         endPts_k1_r[i,j]>rPts[-1])):
                    # Add the new value of phi to the derivatives
                    # x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
                    #                               ^^^^^^^^^^^^^^^
                    drPhi_k[i,j]     = eval_spline_2d_scalar(endPts_k1_q[i,j],endPts_k1_r[i,j],
                                                            kts1Phi, deg1Phi, kts2Phi, deg2Phi,
                                                            coeffsPhi,0,1) \
                                        / endPts_k1_r[i,j]
                    dthetaPhi_k[i,j] = eval_spline_2d_scalar(endPts_k1_q[i,j],endPts_k1_r[i,j],
                                                            kts1Phi, deg1Phi, kts2Phi, deg2Phi,
                                                            coeffsPhi,1,0) \
                                        / endPts_k1_r[i,j]
                else:
                    drPhi_k[i,j]     = 0.0
                    dthetaPhi_k[i,j] = 0.0

        # Step two of Heun method
        # x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )

        # Clipping is one method of avoiding infinite loops due to boundary conditions
        # Using the splines to extrapolate is not sufficient
        endPts_k2_q[:,:] = np.mod(qTPts   - (drPhi_0     + drPhi_k)*multFactor,2*pi)
        for i in range(nPts[0]):
            for j in range(nPts[1]):
                endPts_k2_r[i,j] = rPts[j] + (dthetaPhi_0[i,j] + dthetaPhi_k[i,j])*multFactor
                if (endPts_k2_r[i,j]<rPts[0]):
                    endPts_k2_r[i,j]=rPts[0]
                elif (endPts_k2_r[i,j]>rPts[-1]):
                    endPts_k2_r[i,j]=rPts[-1]
        
        norm = max(np.linalg.norm((endPts_k2_q-endPts_k1_q).flatten(),np.inf),
                   np.linalg.norm((endPts_k2_r-endPts_k1_r).flatten(),np.inf))
        if (norm<tol):
            break
        endPts_k1_q[:,:]=endPts_k2_q[:,:]
        endPts_k1_r[:,:]=endPts_k2_r[:,:]
    
    # Find value at the determined point
    if (nulBound):
        for i,theta in enumerate(qPts):
            for j,r in enumerate(rPts):
                if (endPts_k2_r[i,j]<rPts[0]):
                    f[i,j]=0.0
                elif (endPts_k2_r[i,j]>rPts[-1]):
                    f[i,j]=0.0
                else:
                    while (endPts_k2_q[i,j]>2*pi):
                        endPts_k2_q[i,j]-=2*pi
                    while (endPts_k2_q[i,j]<0):
                        endPts_k2_q[i,j]+=2*pi
                    f[i,j]=eval_spline_2d_scalar(endPts_k2_q[i,j],endPts_k2_r[i,j], kts1Pol, deg1Pol, kts2Pol, deg2Pol, coeffsPol)
    else:
        for i,theta in enumerate(qPts):
            for j,r in enumerate(rPts):
                if (endPts_k2_r[i,j]<rPts[0]):
                    f[i,j]=fEq(rPts[0],v)
                elif (endPts_k2_r[i,j]>rPts[-1]):
                    f[i,j]=fEq(endPts_k2_r[i,j],v)
                else:
                    while (endPts_k2_q[i,j]>2*pi):
                        endPts_k2_q[i,j]-=2*pi
                    while (endPts_k2_q[i,j]<0):
                        endPts_k2_q[i,j]+=2*pi
                    f[i,j]=eval_spline_2d_scalar(endPts_k2_q[i,j],endPts_k2_r[i,j], kts1Pol, deg1Pol, kts2Pol, deg2Pol, coeffsPol)

@cc.export('VParallelAdvectionEvalStep','(f8[:],f8[:],f8,f8,f8,f8[:],i4,f8[:],b1)')
def VParallelAdvectionEvalStep( f: np.ndarray, vPts: np.ndarray, rPos: float,
                        vMin: float, vMax: float,kts: np.ndarray, deg:int,
                        coeffs: np.ndarray,nulBound: bool):
    # Find value at the determined point
    if (nulBound):
        for i, v in enumerate(vPts):
            if (v<vMin or v>vMax):
                f[i]=0.0
            else:
                f[i]=eval_spline_1d_scalar(v,kts,deg,coeffs)
    else:
        for i, v in enumerate(vPts):
            if (v<vMin or v>vMax):
                f[i]=fEq(rPos,v)
            else:
                f[i]=eval_spline_1d_scalar(v,kts,deg,coeffs)

@cc.export('get_lagrange_vals','(i4,i4,i4[:],f8[:,:,:],f8[:],f8[:],f8[:],i4,f8[:])')
def get_lagrange_vals(i,nr,shifts,vals,qVals,thetaShifts,kts,deg,coeffs):
    for j,s in enumerate(shifts):
        for k,q in enumerate(qVals):
            vals[(i-s)%nr,k,j]=eval_spline_1d_scalar(q+thetaShifts[j],kts,deg,coeffs,0)

@cc.export('flux_advection','(i4,i4,f8[:,:],f8[:],f8[:,:,:])')
def flux_advection(nq,nr,f,coeffs,vals):
        for j in range(nq):
            for i in range(nr):
                f[j,i] = coeffs[0]*vals[i,j,0]
                for k in range(1,len(coeffs)):
                    f[j,i] += coeffs[k]*vals[i,j,k]

@cc.export('parallel_gradient_wrap','(i4,i4[:],f8[:],f8[:,:],i4,f8[:],i4,f8[:],f8[:,:,:])')
def parallel_gradient_wrap(i,shifts,coeffs,der,nz,kts,deg,thetaCoeffs,thetaVals):
    for j,(s,c) in enumerate(zip(shifts,coeffs)):
        der[(i-s)%nz,:]+=c*eval_spline_1d_vector(thetaVals[:,i,j],kts,deg,thetaCoeffs,0)

@cc.export('parallel_gradient','(i4,i4[:],f8[:],f8[:,:],f8[:],i4,f8[:],f8[:,:,:])')
def parallel_gradient(i,shifts,coeffs,der,kts,deg,thetaCoeffs,thetaVals):
    for j,(s,c) in enumerate(zip(shifts,coeffs)):
        der[(i-s),:]+=c*eval_spline_1d_vector(thetaVals[:,i,j],kts,deg,thetaCoeffs,0)

if __name__ == "__main__":
    cc.compile()
