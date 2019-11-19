from pyccel.decorators  import types, pure

from ..splines  import spline_eval_funcs as SEF
from ..coordinates import coordinates as coords

if ('mod_pygyro_splines_spline_eval_funcs' in dir(SEF)):
    eval_spline_2d_cross  = lambda xVec,yVec,kts1,deg1,kts2,deg2,coeffs,z,der1,der2 : SEF.mod_pygyro_splines_spline_eval_funcs.eval_spline_2d_cross(xVec,yVec,kts1,deg1,kts2,deg2,coeffs.T,z.T,der1,der2)
    eval_spline_2d_scalar = lambda xVec,yVec,kts1,deg1,kts2,deg2,coeffs,der1,der2 : SEF.mod_pygyro_splines_spline_eval_funcs.eval_spline_2d_scalar(xVec,yVec,kts1,deg1,kts2,deg2,coeffs.T,der1,der2)
    eval_spline_1d_scalar = SEF.mod_pygyro_splines_spline_eval_funcs.eval_spline_1d_scalar
    eval_spline_2d_vector = SEF.mod_pygyro_splines_spline_eval_funcs.eval_spline_2d_vector
else:
    eval_spline_2d_cross  = SEF.eval_spline_2d_cross
    eval_spline_2d_scalar = SEF.eval_spline_2d_scalar
    eval_spline_2d_vector = SEF.eval_spline_2d_vector
    eval_spline_1d_scalar = SEF.eval_spline_1d_scalar

if ('coordinates' in dir(coords)):
    logical_to_pseudocart          = coords.coordinates.logical_to_pseudocart
    pseudocart_to_logical          = coords.coordinates.pseudocart_to_logical
    function_logical_to_pseudocart = coords.coordinates.function_logical_to_pseudocart
    function_pseudocart_to_logical = coords.coordinates.function_pseudocart_to_logical
else:
    logical_to_pseudocart          = coords.logical_to_pseudocart
    pseudocart_to_logical          = coords.pseudocart_to_logical
    function_logical_to_pseudocart = coords.function_logical_to_pseudocart
    function_pseudocart_to_logical = coords.function_pseudocart_to_logical

from ..initialisation.mod_initialiser_funcs               import fEq

@pure
@types('double','double','double[:]','int','double[:]','int','double[:,:]','double','double','double')
def poloidal_advection_pseudoDeriv( startPts_r, startPts_q, kts1Phi, deg1Phi, kts2Phi, deg2Phi,
                              coeffsPhi, dPhi_r0_dx, dPhi_r0_dy, eps):
    if (startPts_r == 0):
        df_dx = dPhi_r0_dy
        df_dy = -dPhi_r0_dx
    elif (startPts_r < eps):
        df_dx = dPhi_r0_dy
        df_dy = -dPhi_r0_dx
        """
        drPhi = eval_spline_2d_scalar(startPts_q,eps,
                                      kts1Phi, deg1Phi, kts2Phi, deg2Phi,
                                      coeffsPhi,0,1)
        drPhi /= eps
        
        dqPhi = eval_spline_2d_scalar(startPts_q,eps,
                                      kts1Phi, deg1Phi, kts2Phi, deg2Phi,
                                      coeffsPhi,1,0)
        dqPhi /= eps
        dx_eps, dy_eps = function_logical_to_pseudocart(dqPhi,-drPhi,eps, startPts_q )
        df_dx =  (1 - startPts_r/eps) * dPhi_r0_dy + startPts_r/eps * dx_eps
        df_dy = -(1 - startPts_r/eps) * dPhi_r0_dx + startPts_r/eps * dy_eps
        """
    else:
        drPhi = eval_spline_2d_scalar(startPts_q, startPts_r,
                                      kts1Phi, deg1Phi, kts2Phi, deg2Phi,
                                      coeffsPhi,0,1)
        drPhi /= startPts_r
        
        dqPhi = eval_spline_2d_scalar(startPts_q, startPts_r,
                                      kts1Phi, deg1Phi, kts2Phi, deg2Phi,
                                      coeffsPhi,1,0)
        dqPhi /= startPts_r
        #print("dr,dq:",drPhi,dqPhi)
        df_dx, df_dy = function_logical_to_pseudocart(dqPhi,-drPhi,startPts_r, startPts_q )
    return df_dx, df_dy

@types('double[:,:]','double','double','double[:]','double[:]','int[:]','double[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:]','double[:]','double[:,:]','int','int','double[:]','double[:]','double[:,:]','int','int','double','double','double','double','double','double','double','double','bool')
def poloidal_advection_step_expl( f, dt, v, rPts, qPts, nPts,
                        drPhi_0, dthetaPhi_0, drPhi_k, dthetaPhi_k,
                        endPts_k1_q, endPts_k1_r, endPts_k2_q, endPts_k2_r,
                        kts1Phi, kts2Phi, coeffsPhi, deg1Phi, deg2Phi,
                        kts1Pol, kts2Pol, coeffsPol, deg1Pol, deg2Pol,
                        CN0, kN0, deltaRN0, rp, CTi,
                        kTi, deltaRTi, B0,nulBound = False ):
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
    
    from numpy import pi, empty, sin, cos, isfinite, atleast_2d

    eps = 1e-12

    multFactor = dt/B0
    multFactor_half = 0.5*multFactor
    
    eval_spline_2d_cross(qPts,rPts, kts1Phi, deg1Phi, kts2Phi, deg2Phi, coeffsPhi,drPhi_0, 0,1)
    eval_spline_2d_cross(qPts,rPts, kts1Phi, deg1Phi, kts2Phi, deg2Phi, coeffsPhi,dthetaPhi_0, 1,0)

    r0_dr = drPhi_0[:,0]

    #TODO: Find sign error
    print(max(abs(r0_dr-3.5*cos(qPts))))

    determinant = cos(qPts[0]) * sin(qPts[1]) - cos(qPts[1]) * sin(qPts[0])
    dPhi_r0_dx = (  sin(qPts[2]) * r0_dr[0] - sin(qPts[0]) * r0_dr[2]) / determinant
    dPhi_r0_dy = (- cos(qPts[2]) * r0_dr[0] + cos(qPts[0]) * r0_dr[2]) / determinant

    rMax = rPts[nPts[1]-1]
    
    for i in range(nPts[0]):
        for j in range(nPts[1]):
            df1_dx, df1_dy = poloidal_advection_pseudoDeriv(rPts[j],qPts[i], kts1Phi, deg1Phi, kts2Phi, deg2Phi,
                              coeffsPhi, dPhi_r0_dx, dPhi_r0_dy, eps)
            # Step one of Heun method
            # x' = x^n + f(x^n)
            start_x, start_y = logical_to_pseudocart(rPts[j], qPts[i])
            endPts_k1_r[i,j], endPts_k1_q[i,j] = pseudocart_to_logical(start_x + df1_dx * multFactor,
                                                                       start_y + df1_dy * multFactor)
            if (endPts_k1_r[i,j] > rMax):
                df2_dx = 0
                df2_dy = 0
            else:
                df2_dx, df2_dy = poloidal_advection_pseudoDeriv(endPts_k1_r[i,j], endPts_k1_q[i,j], kts1Phi, deg1Phi, kts2Phi, deg2Phi,
                              coeffsPhi, dPhi_r0_dx, dPhi_r0_dy, eps)
            # Step two of Heun method
            # x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
            endPts_k2_r[i,j], endPts_k2_q[i,j] = pseudocart_to_logical(start_x + (df1_dx + df2_dx) * multFactor_half,
                                                                       start_y + (df1_dy + df2_dy) * multFactor_half)
    
    # Find value at the determined point
    if (nulBound):
        for i,theta in enumerate(qPts):
            for j,r in enumerate(rPts):
                if (endPts_k2_r[i,j]<rPts[0]):
                    f[i,j]=0.0
                elif (endPts_k2_r[i,j]>rMax):
                    f[i,j]=0.0
                else:
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
                    f[i,j]=eval_spline_2d_scalar(endPts_k2_q[i,j],endPts_k2_r[i,j],
                                                kts1Pol, deg1Pol, kts2Pol, deg2Pol, coeffsPol,0,0)
    
@types('double[:]','double[:]','double','double','double','double[:]','int','double[:]','double','double','double','double','double','double','double','int')
def v_parallel_advection_eval_step( f, vPts, rPos,vMin, vMax,kts, deg,
                        coeffs,CN0,kN0,deltaRN0,rp,CTi,kTi,deltaRTi,bound):
    # Find value at the determined point
    if (bound==0):
        for i in range(len(vPts)):
            v=vPts[i]
            if (v<vMin or v>vMax):
                f[i]=fEq(rPos,v,CN0,kN0,deltaRN0,rp,CTi,
                                    kTi,deltaRTi)
            else:
                f[i]=eval_spline_1d_scalar(v,kts,deg,coeffs,0)
    elif (bound==1):
        for i in range(len(vPts)):
            v=vPts[i]
            if (v<vMin or v>vMax):
                f[i]=0.0
            else:
                f[i]=eval_spline_1d_scalar(v,kts,deg,coeffs,0)
    elif (bound==2):
        vDiff = vMax-vMin
        for i in range(len(vPts)):
            v=vPts[i]
            while (v<vMin):
                v+=vDiff
            while (v>vMax):
                v-=vDiff
            f[i]=eval_spline_1d_scalar(v,kts,deg,coeffs,0)
        

@types('int','int','int[:]','double[:,:,:]','double[:]','double[:]','double[:]','int','double[:]')
def get_lagrange_vals(i,nr,shifts,vals,qVals,thetaShifts,kts,deg,coeffs):
    from numpy import pi
    for j,s in enumerate(shifts):
        for k,q in enumerate(qVals):
            new_q = q+thetaShifts[j]
            while (new_q<0):
                new_q+=2*pi
            while (new_q>2*pi):
                new_q-=2*pi
            vals[(i-s)%nr,k,j]=eval_spline_1d_scalar(new_q,kts,deg,coeffs,0)

@types('int','int','double[:,:]','double[:]','double[:,:,:]')
def flux_advection(nq,nr,f,coeffs,vals):
        for j in range(nq):
            for i in range(nr):
                f[j,i] = coeffs[0]*vals[i,j,0]
                for k in range(1,len(coeffs)):
                    f[j,i] += coeffs[k]*vals[i,j,k]

@types('double[:,:]','double','double','double[:]','double[:]','int[:]','double[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:]','double[:]','double[:,:]','int','int','double[:]','double[:]','double[:,:]','int','int','double','double','double','double','double','double','double','double','double','bool')
def poloidal_advection_step_impl( f, dt, v, rPts, qPts, nPts,
                        drPhi_0, dthetaPhi_0, drPhi_k, dthetaPhi_k,
                        endPts_k1_q, endPts_k1_r, endPts_k2_q, endPts_k2_r,
                        kts1Phi, kts2Phi, coeffsPhi, deg1Phi, deg2Phi,
                        kts1Pol, kts2Pol, coeffsPol, deg1Pol, deg2Pol,
                        CN0, kN0, deltaRN0, rp, CTi, kTi, deltaRTi,
                        B0, tol, nulBound = False ):
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
    from numpy import pi, abs
    
    multFactor = dt/B0
    
    eval_spline_2d_cross(qPts,rPts, kts1Phi, deg1Phi, kts2Phi, deg2Phi, coeffsPhi,drPhi_0, 0,1)
    eval_spline_2d_cross(qPts,rPts, kts1Phi, deg1Phi, kts2Phi, deg2Phi, coeffsPhi,dthetaPhi_0, 1,0)
    
    idx = nPts[1]-1
    rMax = rPts[idx]

    start_x = np.empty([nPts[1],nPts[0]])
    start_y = np.empty([nPts[1],nPts[0]])
    logical_to_pseudocart_cross(rPts[j],qPts[i], start_x, start_y)
    
    for i in range(nPts[0]):
        for j in range(nPts[1]):
            df1_dx, df1_dy = poloidal_advection_pseudoDeriv(rPts[j],qPts[i], kts1Phi, deg1Phi, kts2Phi, deg2Phi,
                              coeffsPhi, dPhi_r0_dx, dPhi_r0_dy, eps)
            # Step one of Heun method
            # x' = x^n + f(x^n)
            endPts_k1_r[i,j], endPts_k1_q[i,j] = pseudocart_to_logical(start_x[i,j] + df1_dx * multFactor,
                                                                       start_y[i,j] + df1_dy * multFactor)

    multFactor *= 0.5
    
    norm=tol+1
    while (norm>tol):
        norm=0.0
        for i in range(nPts[0]):
            for j in range(nPts[1]):
                if (endPts_k1_r[i,j] > rMax):
                    df2_dx = 0
                    df2_dy = 0
                else:
                    df2_dx, df2_dy = poloidal_advection_pseudoDeriv(endPts_k1_r[i,j], endPts_k1_q[i,j], kts1Phi, deg1Phi, kts2Phi, deg2Phi,
                                  coeffsPhi, dPhi_r0_dx, dPhi_r0_dy, eps)
                
                # Step two of Heun method
                # x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
                endPts_k2_r[i,j], endPts_k2_q[i,j] = pseudocart_to_logical(start_x[i,j] + (df1_dx + df2_dx) * multFactor_half,
                                                                           start_y[i,j] + (df1_dy + df2_dy) * multFactor_half)
                # Clipping is one method of avoiding infinite loops due to
                # boundary conditions
                # Using the splines to extrapolate is not sufficient
                if (endPts_k2_r[i,j]>rMax):
                    endPts_k2_r[i,j]=rMax
                
                diff=abs(endPts_k2_q[i,j]-endPts_k1_q[i,j])
                if (diff>norm):
                    norm=diff
                diff=abs(endPts_k2_r[i,j]-endPts_k1_r[i,j])
                if (diff>norm):
                    norm=diff
                endPts_k1_q[i,j]=endPts_k2_q[i,j]
                endPts_k1_r[i,j]=endPts_k2_r[i,j]
    
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

