from pyccel.decorators  import types

#~ from ..splines.spline_eval_funcs import mod_pygyro_splines_spline_eval_funcs as spline

#~ eval_spline_2d_cross=spline.eval_spline_2d_cross
#~ eval_spline_2d_scalar=spline.eval_spline_2d_scalar

#~ from ..initialisation.mod_initialiser_funcs               import fEq

@types('double[:,:]','double','double','double[:]','double[:]','int[:]','double[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:]','double[:]','double[:,:]','int','int','double[:]','double[:]','double[:,:]','int','int','double','double','double','double','double','double','double','double','bool')
def poloidal_advection_step_expl( f, dt, v, rPts, qPts, nPts,
                        drPhi_0, dthetaPhi_0, drPhi_k, dthetaPhi_k,
                        endPts_k1_q, endPts_k1_r, endPts_k2_q, endPts_k2_r,
                        kts1Phi, kts2Phi, coeffsPhi, deg1Phi, deg2Phi,
                        kts1Pol, kts2Pol, coeffsPol, deg1Pol, deg2Pol,
                        CN0, kN0, deltaRN0, rp, CTi,
                        kTi, deltaRTi, B0, nulBound = False ):
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
    from numpy import mod, pi
    
    multFactor = dt/B0
    multFactor_half = 0.5*dt/B0
    
    eval_spline_2d_cross(qPts,rPts, kts1Phi, deg1Phi, kts2Phi, deg2Phi, coeffsPhi,drPhi_0, 0,1)
    eval_spline_2d_cross(qPts,rPts, kts1Phi, deg1Phi, kts2Phi, deg2Phi, coeffsPhi,dthetaPhi_0, 1,0)
    
    idx = nPts[1]-1
    rMax = rPts[idx]
    
    for i in range(nPts[0]):
        for j in range(nPts[1]):
            # Step one of Heun method
            # x' = x^n + f(x^n)
            drPhi_0[i,j]/=rPts[j]
            dthetaPhi_0[i,j]/=rPts[j]
            endPts_k1_q[i,j] = qPts[i] - drPhi_0[i,j]*multFactor
            endPts_k1_r[i,j] = rPts[j] + dthetaPhi_0[i,j]*multFactor
            
            # Handle theta boundary conditions
            while (endPts_k1_q[i,j]<0):
                endPts_k1_q[i,j]+=2*pi
            while (endPts_k1_q[i,j]>2*pi):
                endPts_k1_q[i,j]-=2*pi
            
            if (not (endPts_k1_r[i,j]<rPts[0] or 
                     endPts_k1_r[i,j]>rMax)):
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
            endPts_k2_q[i,j] = (qPts[i] - (drPhi_0[i,j]     + drPhi_k[i,j])*multFactor_half) % 2*pi
            endPts_k2_r[i,j] = rPts[j] + (dthetaPhi_0[i,j] + dthetaPhi_k[i,j])*multFactor_half
    
    # Find value at the determined point
    if (nulBound):
        for i,theta in enumerate(qPts):
            for j,r in enumerate(rPts):
                if (endPts_k2_r[i,j]<rPts[0]):
                    f[i,j]=0.0
                elif (endPts_k2_r[i,j]>rMax):
                    f[i,j]=0.0
                else:
                    while (endPts_k2_q[i,j]>2*pi):
                        endPts_k2_q[i,j]-=2*pi
                    while (endPts_k2_q[i,j]<0):
                        endPts_k2_q[i,j]+=2*pi
                    f[i,j]=eval_spline_2d_scalar(endPts_k2_q[i,j],endPts_k2_r[i,j],
                                                        kts1Pol, deg1Pol, kts2Pol, deg2Pol,
                                                        coeffsPol,0,0)
    else:
        for i,theta in enumerate(qPts):
            for j,r in enumerate(rPts):
                if (endPts_k2_r[i,j]<rPts[0]):
                    f[i,j]=fEq(rPts[0],v,CN0,kN0,deltaRN0,rp,CTi,
                                    kTi,deltaRTi)
                elif (endPts_k2_r[i,j]>rMax):
                    f[i,j]=fEq(endPts_k2_r[i,j],v,CN0,kN0,
                                    deltaRN0,rp,CTi,kTi,deltaRTi)
                else:
                    while (endPts_k2_q[i,j]>2*pi):
                        endPts_k2_q[i,j]-=2*pi
                    while (endPts_k2_q[i,j]<0):
                        endPts_k2_q[i,j]+=2*pi
                    f[i,j]=eval_spline_2d_scalar(endPts_k2_q[i,j],endPts_k2_r[i,j],
                                                kts1Pol, deg1Pol, kts2Pol, deg2Pol, coeffsPol,0,0)
    
