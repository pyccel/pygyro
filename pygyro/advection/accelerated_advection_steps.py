from pyccel.decorators import pure
from ..splines.spline_eval_funcs import eval_spline_2d_cross, eval_spline_2d_scalar, eval_spline_1d_scalar
from ..initialisation.initialiser_funcs import f_eq


@pure
def poloidal_advection_step_expl(f: 'float[:,:]',
                                 dt: 'float', v: 'float',
                                 rPts: 'float[:]', qPts: 'float[:]',
                                 drPhi_0: 'float[:,:]', dthetaPhi_0: 'float[:,:]',
                                 drPhi_k: 'float[:,:]', dthetaPhi_k: 'float[:,:]',
                                 endPts_k1_q: 'float[:,:]', endPts_k1_r: 'float[:,:]', endPts_k2_q: 'float[:,:]', endPts_k2_r: 'float[:,:]',
                                 kts1Phi: 'float[:]', kts2Phi: 'float[:]', coeffsPhi: 'float[:,:]', deg1Phi: 'int', deg2Phi: 'int',
                                 kts1Pol: 'float[:]', kts2Pol: 'float[:]', coeffsPol: 'float[:,:]', deg1Pol: 'int', deg2Pol: 'int',
                                 CN0: 'float', kN0: 'float', deltaRN0: 'float', rp: 'float', CTi: 'float',
                                 kTi: 'float', deltaRTi: 'float', B0: 'float', nulBound: 'bool' = False):
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

    from numpy import pi

    multFactor = dt / B0
    multFactor_half = 0.5 * multFactor
    eval_spline_2d_cross(qPts, rPts, kts1Phi, deg1Phi,
                         kts2Phi, deg2Phi, coeffsPhi, drPhi_0, 0, 1)
    eval_spline_2d_cross(qPts, rPts, kts1Phi, deg1Phi,
                         kts2Phi, deg2Phi, coeffsPhi, dthetaPhi_0, 1, 0)

    nPts_r = rPts.shape[0]
    nPts_q = qPts.shape[0]

    idx = nPts_r-1
    rMax = rPts[idx]

    for i in range(nPts_q):
        for j in range(nPts_r):
            # Step one of Heun method
            # x' = x^n + f(x^n)
            drPhi_0[i, j] /= rPts[j]
            dthetaPhi_0[i, j] /= rPts[j]
            endPts_k1_q[i, j] = qPts[i] - drPhi_0[i, j] * multFactor
            endPts_k1_r[i, j] = rPts[j] + dthetaPhi_0[i, j] * multFactor

            # Handle theta boundary conditions
            endPts_k1_q[i, j] = endPts_k1_q[i, j] % 2*pi

            if (not (endPts_k1_r[i, j] < rPts[0] or
                     endPts_k1_r[i, j] > rMax)):
                # Add the new value of phi to the derivatives
                # x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
                #                               ^^^^^^^^^^^^^^^
                drPhi_k[i, j] = eval_spline_2d_scalar(endPts_k1_q[i, j], endPts_k1_r[i, j],
                                                      kts1Phi, deg1Phi, kts2Phi, deg2Phi,
                                                      coeffsPhi, 0, 1)
                drPhi_k[i, j] /= endPts_k1_r[i, j]

                dthetaPhi_k[i, j] = eval_spline_2d_scalar(endPts_k1_q[i, j], endPts_k1_r[i, j],
                                                          kts1Phi, deg1Phi, kts2Phi, deg2Phi,
                                                          coeffsPhi, 1, 0)
                dthetaPhi_k[i, j] /= endPts_k1_r[i, j]
            else:
                drPhi_k[i, j] = 0.0
                dthetaPhi_k[i, j] = 0.0

            # Step two of Heun method
            # x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
            endPts_k2_q[i, j] = (
                qPts[i] - (drPhi_0[i, j] + drPhi_k[i, j]) * multFactor_half) % (2*pi)

            endPts_k2_r[i, j] = rPts[j] + \
                (dthetaPhi_0[i, j] + dthetaPhi_k[i, j]) * multFactor_half

    # Find value at the determined point
    if (nulBound):
        for i in range(nPts_q):  # theta
            for j in range(nPts_r):  # r
                if (endPts_k2_r[i, j] < rPts[0]):
                    f[i, j] = 0.0
                elif (endPts_k2_r[i, j] > rMax):
                    f[i, j] = 0.0
                else:
                    endPts_k2_q[i, j] = endPts_k2_q[i, j] % 2*pi
                    f[i, j] = eval_spline_2d_scalar(endPts_k2_q[i, j], endPts_k2_r[i, j],
                                                    kts1Pol, deg1Pol, kts2Pol, deg2Pol,
                                                    coeffsPol, 0, 0)
    else:
        for i in range(nPts_q):  # theta
            for j in range(nPts_r):  # r
                if (endPts_k2_r[i, j] < rPts[0]):
                    f[i, j] = f_eq(rPts[0], v, CN0, kN0, deltaRN0, rp, CTi,
                                   kTi, deltaRTi)
                elif (endPts_k2_r[i, j] > rMax):
                    f[i, j] = f_eq(endPts_k2_r[i, j], v, CN0, kN0,
                                   deltaRN0, rp, CTi, kTi, deltaRTi)
                else:
                    endPts_k2_q[i, j] = endPts_k2_q[i, j] % 2*pi
                    f[i, j] = eval_spline_2d_scalar(endPts_k2_q[i, j], endPts_k2_r[i, j],
                                                    kts1Pol, deg1Pol, kts2Pol, deg2Pol, coeffsPol, 0, 0)


@pure
def v_parallel_advection_eval_step(f: 'float[:]', vPts: 'float[:]',
                                   rPos: 'float', vMin: 'float', vMax: 'float',
                                   kts: 'float[:]', deg: 'int',
                                   coeffs: 'float[:]',
                                   CN0: 'float', kN0: 'float', deltaRN0: 'float', rp: 'float',
                                   CTi: 'float', kTi: 'float', deltaRTi: 'float', bound: 'int'):
    """
    TODO
    """
    # Find value at the determined point
    if (bound == 0):
        for i, v in enumerate(vPts):
            if (v < vMin or v > vMax):
                f[i] = f_eq(rPos, v, CN0, kN0, deltaRN0, rp, CTi,
                            kTi, deltaRTi)
            else:
                f[i] = eval_spline_1d_scalar(v, kts, deg, coeffs, 0)
    elif (bound == 1):
        for i, v in enumerate(vPts):
            if (v < vMin or v > vMax):
                f[i] = 0.0
            else:
                f[i] = eval_spline_1d_scalar(v, kts, deg, coeffs, 0)
    elif (bound == 2):
        vDiff = vMax - vMin
        for i, v in enumerate(vPts):
            while (v < vMin):
                v += vDiff
            while (v > vMax):
                v -= vDiff
            f[i] = eval_spline_1d_scalar(v, kts, deg, coeffs, 0)


@pure
def get_lagrange_vals(i: 'int', shifts: 'int[:]',
                      vals: 'float[:,:,:]', qVals: 'float[:]',
                      thetaShifts: 'float[:]', kts: 'float[:]',
                      deg: 'int', coeffs: 'float[:]'):
    """
    TODO
    """
    from numpy import pi
    nz = vals.shape[0]

    for j, s in enumerate(shifts):
        for k, q in enumerate(qVals):
            new_q = q + thetaShifts[j]
            new_q = new_q % 2*pi

            # the first index can be negative, but allow_negative_index will take the necessary modulo
            vals[(i - s) % nz, k, j] = eval_spline_1d_scalar(new_q,
                                                             kts, deg, coeffs, 0)


@pure
def flux_advection(nq: 'int', nr: 'int',
                   f: 'float[:,:]', coeffs: 'float[:]', vals: 'float[:,:,:]'):
    """
    TODO
    """
    for j in range(nq):
        for i in range(nr):
            f[j, i] = coeffs[0]*vals[i, j, 0]
            for k in range(1, len(coeffs)):
                f[j, i] += coeffs[k]*vals[i, j, k]


@pure
def poloidal_advection_step_impl(f: 'float[:,:]', dt: 'float', v: 'float', rPts: 'float[:]', qPts: 'float[:]',
                                 drPhi_0: 'float[:,:]', dthetaPhi_0: 'float[:,:]', drPhi_k: 'float[:,:]', dthetaPhi_k: 'float[:,:]',
                                 endPts_k1_q: 'float[:,:]', endPts_k1_r: 'float[:,:]', endPts_k2_q: 'float[:,:]', endPts_k2_r: 'float[:,:]',
                                 kts1Phi: 'float[:]', kts2Phi: 'float[:]', coeffsPhi: 'float[:,:]', deg1Phi: 'int', deg2Phi: 'int',
                                 kts1Pol: 'float[:]', kts2Pol: 'float[:]', coeffsPol: 'float[:,:]', deg1Pol: 'int', deg2Pol: 'int',
                                 CN0: 'float', kN0: 'float', deltaRN0: 'float', rp: 'float', CTi: 'float', kTi: 'float', deltaRTi: 'float',
                                 B0: 'float', tol: 'float', nulBound: 'bool' = False):
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

    eval_spline_2d_cross(qPts, rPts, kts1Phi, deg1Phi,
                         kts2Phi, deg2Phi, coeffsPhi, drPhi_0, 0, 1)
    eval_spline_2d_cross(qPts, rPts, kts1Phi, deg1Phi,
                         kts2Phi, deg2Phi, coeffsPhi, dthetaPhi_0, 1, 0)

    nPts_r = rPts.shape[0]
    nPts_q = qPts.shape[0]

    idx = nPts_r-1
    rMax = rPts[idx]

    for i in range(nPts_q):
        for j in range(nPts_r):
            # Step one of Heun method
            # x' = x^n + f(x^n)
            drPhi_0[i, j] /= rPts[j]
            dthetaPhi_0[i, j] /= rPts[j]
            endPts_k1_q[i, j] = qPts[i] - drPhi_0[i, j]*multFactor
            endPts_k1_r[i, j] = rPts[j] + dthetaPhi_0[i, j]*multFactor

    multFactor *= 0.5

    norm = tol+1
    while (norm > tol):
        norm = 0.0
        for i in range(nPts_q):
            for j in range(nPts_r):
                # Handle theta boundary conditions
                endPts_k1_q[i, j] = endPts_k1_q[i, j] % 2*pi

                if (not (endPts_k1_r[i, j] < rPts[0] or
                         endPts_k1_r[i, j] > rMax)):
                    # Add the new value of phi to the derivatives
                    # x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
                    #                               ^^^^^^^^^^^^^^^
                    drPhi_k[i, j] = eval_spline_2d_scalar(endPts_k1_q[i, j], endPts_k1_r[i, j],
                                                          kts1Phi, deg1Phi, kts2Phi, deg2Phi,
                                                          coeffsPhi, 0, 1)
                    drPhi_k[i, j] /= endPts_k1_r[i, j]
                    dthetaPhi_k[i, j] = eval_spline_2d_scalar(endPts_k1_q[i, j], endPts_k1_r[i, j],
                                                              kts1Phi, deg1Phi, kts2Phi, deg2Phi,
                                                              coeffsPhi, 1, 0)
                    dthetaPhi_k[i, j] /= endPts_k1_r[i, j]
                else:
                    drPhi_k[i, j] = 0.0
                    dthetaPhi_k[i, j] = 0.0

                # Step two of Heun method
                # x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
                # Clipping is one method of avoiding infinite loops due to
                # boundary conditions
                # Using the splines to extrapolate is not sufficient
                endPts_k2_q[i, j] = (
                    qPts[i] - (drPhi_0[i, j] + drPhi_k[i, j]) * multFactor) % (2*pi)

                endPts_k2_r[i, j] = rPts[j] + \
                    (dthetaPhi_0[i, j] + dthetaPhi_k[i, j]) * multFactor

                if (endPts_k2_r[i, j] < rPts[0]):
                    endPts_k2_r[i, j] = rPts[0]
                elif (endPts_k2_r[i, j] > rMax):
                    endPts_k2_r[i, j] = rMax

                diff = abs(endPts_k2_q[i, j]-endPts_k1_q[i, j])
                if (diff > norm):
                    norm = diff
                diff = abs(endPts_k2_r[i, j]-endPts_k1_r[i, j])
                if (diff > norm):
                    norm = diff
                endPts_k1_q[i, j] = endPts_k2_q[i, j]
                endPts_k1_r[i, j] = endPts_k2_r[i, j]

    # Find value at the determined point
    if (nulBound):
        for i in range(nPts_q):
            for j in range(nPts_r):
                if (endPts_k2_r[i, j] < rPts[0]):
                    f[i, j] = 0.0
                elif (endPts_k2_r[i, j] > rMax):
                    f[i, j] = 0.0
                else:
                    endPts_k2_q[i, j] = endPts_k2_q[i, j] % 2*pi
                    f[i, j] = eval_spline_2d_scalar(endPts_k2_q[i, j], endPts_k2_r[i, j],
                                                    kts1Pol, deg1Pol, kts2Pol, deg2Pol,
                                                    coeffsPol, 0, 0)
    else:
        for i in range(nPts_q):
            for j in range(nPts_r):
                if (endPts_k2_r[i, j] < rPts[0]):
                    f[i, j] = f_eq(rPts[0], v, CN0, kN0, deltaRN0, rp, CTi,
                                   kTi, deltaRTi)
                elif (endPts_k2_r[i, j] > rMax):
                    f[i, j] = f_eq(endPts_k2_r[i, j], v, CN0, kN0,
                                   deltaRN0, rp, CTi, kTi, deltaRTi)
                else:
                    endPts_k2_q[i, j] = endPts_k2_q[i, j] % 2*pi
                    f[i, j] = eval_spline_2d_scalar(endPts_k2_q[i, j], endPts_k2_r[i, j],
                                                    kts1Pol, deg1Pol, kts2Pol, deg2Pol, coeffsPol, 0, 0)
