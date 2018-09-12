import numpy            as np
from numba              import jit
from scipy.interpolate  import splev, bisplev
from math               import pi

from ..initialisation.initialiser   import fEq
from ..initialisation               import constants

@jit
def eval2d( x1, x2, kts1, kts2, coeffs, deg1, deg2, der1=0, der2=0 ):

    tck = (kts1, kts2, coeffs, deg1, deg2)
    return bisplev( x1, x2, tck, der1, der2 )

@jit
def PoloidalAdvectionStepExpl( f: np.ndarray, dt: float, v: float,
                        rPts: np.ndarray, qPts: np.ndarray, qTPts: np.ndarray,
                        nPts: list, kts1Phi: np.ndarray, kts2Phi: np.ndarray,
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
    
    drPhi_0 = eval2d(qPts,rPts, kts1Phi, kts2Phi, coeffsPhi, deg1Phi, deg2Phi,0,1)/rPts
    dthetaPhi_0 = eval2d(qPts,rPts, kts1Phi, kts2Phi, coeffsPhi, deg1Phi, deg2Phi,1,0)/rPts
    
    # Step one of Heun method
    # x' = x^n + f(x^n)
    endPts_k1 = ( qTPts   -     drPhi_0*multFactor,
                 rPts + dthetaPhi_0*multFactor )
    
    drPhi_k = np.empty_like(drPhi_0)
    dthetaPhi_k = np.empty_like(dthetaPhi_0)
    
    multFactor*=0.5
    
    for i in range(nPts[0]):
        for j in range(nPts[1]):
            # Handle theta boundary conditions
            while (endPts_k1[0][i,j]<0):
                endPts_k1[0][i,j]+=2*pi
            while (endPts_k1[0][i,j]>2*pi):
                endPts_k1[0][i,j]-=2*pi
            
            if (not (endPts_k1[1][i,j]<rPts[0] or 
                     endPts_k1[1][i,j]>rPts[-1])):
                # Add the new value of phi to the derivatives
                # x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
                #                               ^^^^^^^^^^^^^^^
                drPhi_k[i,j]     = eval2d(endPts_k1[0][i,j],endPts_k1[1][i,j], kts1Phi, kts2Phi, coeffsPhi, deg1Phi, deg2Phi,0,1)/endPts_k1[1][i,j]
                dthetaPhi_k[i,j] = eval2d(endPts_k1[0][i,j],endPts_k1[1][i,j], kts1Phi, kts2Phi, coeffsPhi, deg1Phi, deg2Phi,1,0)/endPts_k1[1][i,j]
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
                    f[i,j]=eval2d(endPts_k2[0][i,j],endPts_k2[1][i,j], kts1Pol, kts2Pol, coeffsPol, deg1Pol, deg2Pol)
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
                    f[i,j]=eval2d(endPts_k2[0][i,j],endPts_k2[1][i,j], kts1Pol, kts2Pol, coeffsPol, deg1Pol, deg2Pol)



@jit
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
    
    drPhi_0 = eval2d(qPts,rPts, kts1Phi, kts2Phi, coeffsPhi, deg1Phi, deg2Phi,0,1)/rPts
    dthetaPhi_0 = eval2d(qPts,rPts, kts1Phi, kts2Phi, coeffsPhi, deg1Phi, deg2Phi,1,0)/rPts
    
    # Step one of Heun method
    # x' = x^n + f(x^n)
    endPts_k1 = ( qTPts   -     drPhi_0*multFactor,
                 rPts + dthetaPhi_0*multFactor )
    
    drPhi_k = np.empty_like(drPhi_0)
    dthetaPhi_k = np.empty_like(dthetaPhi_0)
    
    multFactor*=0.5
    
    while (True):
        
        for i in range(nPts[0]):
            for j in range(nPts[1]):
                # Handle theta boundary conditions
                while (endPts_k1[0][i,j]<0):
                    endPts_k1[0][i,j]+=2*pi
                while (endPts_k1[0][i,j]>2*pi):
                    endPts_k1[0][i,j]-=2*pi
                
                if (not (endPts_k1[1][i,j]<rPts[0] or 
                         endPts_k1[1][i,j]>rPts[-1])):
                    # Add the new value of phi to the derivatives
                    # x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
                    #                               ^^^^^^^^^^^^^^^
                    drPhi_k[i,j]     = eval2d(endPts_k1[0][i,j],endPts_k1[1][i,j], kts1Phi, kts2Phi, coeffsPhi, deg1Phi, deg2Phi,0,1)/endPts_k1[1][i,j]
                    dthetaPhi_k[i,j] = eval2d(endPts_k1[0][i,j],endPts_k1[1][i,j], kts1Phi, kts2Phi, coeffsPhi, deg1Phi, deg2Phi,1,0)/endPts_k1[1][i,j]
                else:
                    drPhi_k[i,j]     = 0.0
                    dthetaPhi_k[i,j] = 0.0

        # Step two of Heun method
        # x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )

        # Clipping is one method of avoiding infinite loops due to boundary conditions
        # Using the splines to extrapolate is not sufficient
        endPts_k2 = ( np.mod(qTPts   - (drPhi_0     + drPhi_k)*multFactor,2*pi),
                      np.clip(rPts + (dthetaPhi_0 + dthetaPhi_k)*multFactor,
                              rPts[0], rPts[-1]) )
        
        norm = max(np.linalg.norm((endPts_k2[0]-endPts_k1[0]).flatten(),np.inf),
                   np.linalg.norm((endPts_k2[1]-endPts_k1[1]).flatten(),np.inf))
        if (norm<tol):
            break
        endPts_k1=endPts_k2
    
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
                    f[i,j]=eval2d(endPts_k2[0][i,j],endPts_k2[1][i,j], kts1Pol, kts2Pol, coeffsPol, deg1Pol, deg2Pol)
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
                    f[i,j]=eval2d(endPts_k2[0][i,j],endPts_k2[1][i,j], kts1Pol, kts2Pol, coeffsPol, deg1Pol, deg2Pol)
