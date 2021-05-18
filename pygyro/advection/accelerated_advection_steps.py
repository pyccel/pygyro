from numpy import pi, abs as np_abs

from pyccel.decorators  import types, pure, allow_negative_index
from ..splines.spline_eval_funcs import eval_spline_2d_cross, eval_spline_2d_scalar, eval_spline_1d_scalar
from ..initialisation.initialiser_funcs               import f_eq

@pure
def poloidal_advection_step_expl(
    f           : 'double[:,:]',
    dt          : 'double',
    v           : 'double',
    rPts        : 'const double[:]',
    qPts        : 'const double[:]',
    drPhi_0     : 'double[:,:]',
    dthetaPhi_0 : 'double[:,:]',
    drPhi_k     : 'double[:,:]',
    dthetaPhi_k : 'double[:,:]',
    endPts_k1_q : 'double[:,:]',
    endPts_k1_r : 'double[:,:]',
    endPts_k2_q : 'double[:,:]',
    endPts_k2_r : 'double[:,:]',
    kts1Phi     : 'const double[:]',
    kts2Phi     : 'const double[:]',
    coeffsPhi   : 'const double[:,:]',
    deg1Phi     : 'int',
    deg2Phi     : 'int',
    kts1Pol     : 'const double[:]',
    kts2Pol     : 'const double[:]',
    coeffsPol   : 'const double[:,:]',
    deg1Pol     : 'int',
    deg2Pol     : 'int',
    CN0         : 'double',
    kN0         : 'double',
    deltaRN0    : 'double',
    rp          : 'double',
    CTi         : 'double',
    kTi         : 'double',
    deltaRTi    : 'double',
    B0          : 'double',
    nulBound    : 'bool' = False ):
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

    multFactor = dt/B0
    multFactor_half = 0.5*multFactor

    eval_spline_2d_cross(qPts,rPts, kts1Phi, deg1Phi, kts2Phi, deg2Phi, coeffsPhi,drPhi_0, 0,1)
    eval_spline_2d_cross(qPts,rPts, kts1Phi, deg1Phi, kts2Phi, deg2Phi, coeffsPhi,dthetaPhi_0, 1,0)

    nPts_r = rPts.shape[0]
    nPts_q = qPts.shape[0]

    idx = nPts_r-1
    rMax = rPts[idx]

    for i in range(nPts_q):
        for j in range(nPts_r):
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
                                                        coeffsPhi,0,1)
                drPhi_k[i,j]     /= endPts_k1_r[i,j]

                dthetaPhi_k[i,j] = eval_spline_2d_scalar(endPts_k1_q[i,j],endPts_k1_r[i,j],
                                                        kts1Phi, deg1Phi, kts2Phi, deg2Phi,
                                                        coeffsPhi,1,0)
                dthetaPhi_k[i,j] /= endPts_k1_r[i,j]
            else:
                drPhi_k[i,j]     = 0.0
                dthetaPhi_k[i,j] = 0.0

            # Step two of Heun method
            # x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
            endPts_k2_q[i,j] = (qPts[i] - (drPhi_0[i,j]     + drPhi_k[i,j])*multFactor_half) % (2*pi)
            if endPts_k2_q[i,j]<0: # Needed for C due to pyccel issue #854
                endPts_k2_q[i,j] += 2*pi
            endPts_k2_r[i,j] = rPts[j] + (dthetaPhi_0[i,j] + dthetaPhi_k[i,j])*multFactor_half

    # Find value at the determined point
    if (nulBound):
        for i in range(nPts_q): # theta
            for j in range(nPts_r): # r
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
        for i in range(nPts_q): # theta
            for j in range(nPts_r): # r
                if (endPts_k2_r[i,j]<rPts[0]):
                    f[i,j]=f_eq(rPts[0],v,CN0,kN0,deltaRN0,rp,CTi,
                                    kTi,deltaRTi)
                elif (endPts_k2_r[i,j]>rMax):
                    f[i,j]=f_eq(endPts_k2_r[i,j],v,CN0,kN0,
                                    deltaRN0,rp,CTi,kTi,deltaRTi)
                else:
                    while (endPts_k2_q[i,j]>2*pi):
                        endPts_k2_q[i,j]-=2*pi
                    while (endPts_k2_q[i,j]<0):
                        endPts_k2_q[i,j]+=2*pi
                    f[i,j]=eval_spline_2d_scalar(endPts_k2_q[i,j],endPts_k2_r[i,j],
                                                kts1Pol, deg1Pol, kts2Pol, deg2Pol, coeffsPol,0,0)

@pure
def v_parallel_advection_eval_step(
        f        : 'double[:]',
        vPts     : 'const double[:]',
        rPos     : 'double',
        vMin     : 'double',
        vMax     : 'double',
        kts      : 'const double[:]',
        deg      : 'int',
        coeffs   : 'double[:]',
        CN0      : 'double',
        kN0      : 'double',
        deltaRN0 : 'double',
        rp       : 'double',
        CTi      : 'double',
        kTi      : 'double',
        deltaRTi : 'double',
        bound    : 'int'):
    # Find value at the determined point
    if (bound==0):
        for i,v in enumerate(vPts):
            if (v<vMin or v>vMax):
                f[i]=f_eq(rPos,v,CN0,kN0,deltaRN0,rp,CTi,
                                    kTi,deltaRTi)
            else:
                f[i]=eval_spline_1d_scalar(v,kts,deg,coeffs,0)
    elif (bound==1):
        for i,v in enumerate(vPts):
            if (v<vMin or v>vMax):
                f[i]=0.0
            else:
                f[i]=eval_spline_1d_scalar(v,kts,deg,coeffs,0)
    elif (bound==2):
        vDiff = vMax-vMin
        for i,v in enumerate(vPts):
            while (v<vMin):
                v+=vDiff
            while (v>vMax):
                v-=vDiff
            f[i]=eval_spline_1d_scalar(v,kts,deg,coeffs,0)


@pure
@allow_negative_index('vals') # Needed for C due to pyccel issue #854
def get_lagrange_vals(
        i           : 'int',
        shifts      : 'const int[:]',
        vals        : 'double[:,:,:]',
        qVals       : 'double[:]',
        thetaShifts : 'double[:]',
        kts         : 'double[:]',
        deg         : 'int',
        coeffs      : 'double[:]'):
    nz = vals.shape[0]
    for j,s in enumerate(shifts):
        for k,q in enumerate(qVals):
            new_q = q+thetaShifts[j]
            while (new_q<0):
                new_q+=2*pi
            while (new_q>2*pi):
                new_q-=2*pi
            vals[(i-s)%nz,k,j]=eval_spline_1d_scalar(new_q,kts,deg,coeffs,0)

@pure
def flux_advection(
        nq     : 'int',
        nr     : 'int',
        f      : 'double[:,:]',
        coeffs : 'const double[:]',
        vals   : 'const double[:,:,:]'):
        for j in range(nq):
            for i in range(nr):
                f[j,i] = coeffs[0]*vals[i,j,0]
                for k in range(1,len(coeffs)):
                    f[j,i] += coeffs[k]*vals[i,j,k]

@pure
def poloidal_advection_step_impl(
    f           : 'double[:,:]',
    dt          : 'double',
    v           : 'double',
    rPts        : 'const double[:]',
    qPts        : 'const double[:]',
    drPhi_0     : 'double[:,:]',
    dthetaPhi_0 : 'double[:,:]',
    drPhi_k     : 'double[:,:]',
    dthetaPhi_k : 'double[:,:]',
    endPts_k1_q : 'double[:,:]',
    endPts_k1_r : 'double[:,:]',
    endPts_k2_q : 'double[:,:]',
    endPts_k2_r : 'double[:,:]',
    kts1Phi     : 'const double[:]',
    kts2Phi     : 'const double[:]',
    coeffsPhi   : 'const double[:,:]',
    deg1Phi     : 'int',
    deg2Phi     : 'int',
    kts1Pol     : 'const double[:]',
    kts2Pol     : 'const double[:]',
    coeffsPol   : 'const double[:,:]',
    deg1Pol     : 'int',
    deg2Pol     : 'int',
    CN0         : 'double',
    kN0         : 'double',
    deltaRN0    : 'double',
    rp          : 'double',
    CTi         : 'double',
    kTi         : 'double',
    deltaRTi    : 'double',
    B0          : 'double',
    tol         : 'double',
    nulBound    : 'bool' = False ):
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

    multFactor = dt/B0

    eval_spline_2d_cross(qPts,rPts, kts1Phi, deg1Phi, kts2Phi, deg2Phi, coeffsPhi,drPhi_0, 0,1)
    eval_spline_2d_cross(qPts,rPts, kts1Phi, deg1Phi, kts2Phi, deg2Phi, coeffsPhi,dthetaPhi_0, 1,0)

    nPts_r = rPts.shape[0]
    nPts_q = qPts.shape[0]

    idx = nPts_r-1
    rMax = rPts[idx]

    for i in range(nPts_q):
        for j in range(nPts_r):
            # Step one of Heun method
            # x' = x^n + f(x^n)
            drPhi_0[i,j]/=rPts[j]
            dthetaPhi_0[i,j]/=rPts[j]
            endPts_k1_q[i,j] = qPts[i] - drPhi_0[i,j]*multFactor
            endPts_k1_r[i,j] = rPts[j] + dthetaPhi_0[i,j]*multFactor

    multFactor *= 0.5

    norm=tol+1
    while (norm>tol):
        norm=0.0
        for i in range(nPts_q):
            for j in range(nPts_r):
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
                                                        coeffsPhi,0,1)
                    drPhi_k[i,j]     /= endPts_k1_r[i,j]
                    dthetaPhi_k[i,j] = eval_spline_2d_scalar(endPts_k1_q[i,j],endPts_k1_r[i,j],
                                                        kts1Phi, deg1Phi, kts2Phi, deg2Phi,
                                                        coeffsPhi,1,0)
                    dthetaPhi_k[i,j] /= endPts_k1_r[i,j]
                else:
                    drPhi_k[i,j]     = 0.0
                    dthetaPhi_k[i,j] = 0.0

                # Step two of Heun method
                # x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
                # Clipping is one method of avoiding infinite loops due to
                # boundary conditions
                # Using the splines to extrapolate is not sufficient
                endPts_k2_q[i,j] = (qPts[i] - (drPhi_0[i,j]     + drPhi_k[i,j])*multFactor) % (2*pi)
                if endPts_k2_q[i,j]<0: # Needed for C due to pyccel issue #854
                    endPts_k2_q[i,j] += 2*pi
                endPts_k2_r[i,j] = rPts[j] + (dthetaPhi_0[i,j] + dthetaPhi_k[i,j])*multFactor

                if (endPts_k2_r[i,j]<rPts[0]):
                    endPts_k2_r[i,j]=rPts[0]
                elif (endPts_k2_r[i,j]>rMax):
                    endPts_k2_r[i,j]=rMax

                diff=np_abs(endPts_k2_q[i,j]-endPts_k1_q[i,j])
                if (diff>norm):
                    norm=diff
                diff=np_abs(endPts_k2_r[i,j]-endPts_k1_r[i,j])
                if (diff>norm):
                    norm=diff
                endPts_k1_q[i,j]=endPts_k2_q[i,j]
                endPts_k1_r[i,j]=endPts_k2_r[i,j]

    # Find value at the determined point
    if (nulBound):
        for i in range(nPts_q):
            for j in range(nPts_r):
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
        for i in range(nPts_q):
            for j in range(nPts_r):
                if (endPts_k2_r[i,j]<rPts[0]):
                    f[i,j]=f_eq(rPts[0],v,CN0,kN0,deltaRN0,rp,CTi,
                                    kTi,deltaRTi)
                elif (endPts_k2_r[i,j]>rMax):
                    f[i,j]=f_eq(endPts_k2_r[i,j],v,CN0,kN0,
                                    deltaRN0,rp,CTi,kTi,deltaRTi)
                else:
                    while (endPts_k2_q[i,j]>2*pi):
                        endPts_k2_q[i,j]-=2*pi
                    while (endPts_k2_q[i,j]<0):
                        endPts_k2_q[i,j]+=2*pi
                    f[i,j]=eval_spline_2d_scalar(endPts_k2_q[i,j],endPts_k2_r[i,j],
                                                kts1Pol, deg1Pol, kts2Pol, deg2Pol, coeffsPol,0,0)

