import numpy as np
from typing import Final
from pyccel.decorators import pure
from ..splines.splines import Spline1D, Spline2D
from ..splines.accelerated_spline_interpolators import solve_system_nonperiodic, solve_system_periodic, solve_2d_system
from ..splines.sll_m_spline_matrix_periodic_banded import PeriodicBandedMatrix
from ..initialisation.initialiser_funcs import f_eq


def poloidal_advection_step_expl(f: 'float[:,:]',
                                         dt: 'float', v: 'float',
                                         rPts: 'Final[float[:]]', qPts: 'Final[float[:]]',
                                         drPhi_0: 'float[:,:]', dthetaPhi_0: 'float[:,:]',
                                         drPhi_k: 'float[:,:]', dthetaPhi_k: 'float[:,:]',
                                         endPts_k1_q: 'float[:,:]', endPts_k1_r: 'float[:,:]', endPts_k2_q: 'float[:,:]', endPts_k2_r: 'float[:,:]',
                                         phi_spline : Spline2D, pol_spline : Spline2D,
                                         CN0: 'float', kN0: 'float', deltaRN0: 'float', rp: 'float', CTi: 'float',
                                         kTi: 'float', deltaRTi: 'float', B0: 'float', nulBound: 'bool'):
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
    phi_spline.eval_vector(qPts, rPts, drPhi_0, 0, 1)
    phi_spline.eval_vector(qPts, rPts, dthetaPhi_0, 1, 0)

    nPts_r = rPts.shape[0]
    nPts_q = qPts.shape[0]

    idx = nPts_r-1
    rMax = rPts[idx]

    #$ omp parallel for collapse(2)
    for i in range(nPts_q):
        for j in range(nPts_r):
            # Step one of Heun method
            # x' = x^n + f(x^n)
            drPhi_0[i, j] /= rPts[j]
            dthetaPhi_0[i, j] /= rPts[j]
            endPts_k1_q[i, j] = qPts[i] - drPhi_0[i, j] * multFactor
            endPts_k1_r[i, j] = rPts[j] + dthetaPhi_0[i, j] * multFactor

            # Handle theta boundary conditions
            endPts_k1_q[i, j] = endPts_k1_q[i, j] % (2*pi)

            if (not (endPts_k1_r[i, j] < rPts[0] or
                     endPts_k1_r[i, j] > rMax)):
                # Add the new value of phi to the derivatives
                # x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
                #                               ^^^^^^^^^^^^^^^
                drPhi_k[i, j] = phi_spline.eval(endPts_k1_q[i, j], endPts_k1_r[i, j],
                                                      0, 1)
                drPhi_k[i, j] /= endPts_k1_r[i, j]

                dthetaPhi_k[i, j] = phi_spline.eval(endPts_k1_q[i, j], endPts_k1_r[i, j],
                                                          1, 0)
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
        #$ omp parallel for collapse(2)
        for i in range(nPts_q):  # theta
            for j in range(nPts_r):  # r
                if (endPts_k2_r[i, j] < rPts[0]):
                    f[i, j] = 0.0
                elif (endPts_k2_r[i, j] > rMax):
                    f[i, j] = 0.0
                else:
                    endPts_k2_q[i, j] = endPts_k2_q[i, j] % (2*pi)
                    f[i, j] = pol_spline.eval(endPts_k2_q[i, j], endPts_k2_r[i, j])
    else:
        #$ omp parallel for collapse(2)
        for i in range(nPts_q):  # theta
            for j in range(nPts_r):  # r
                if (endPts_k2_r[i, j] < rPts[0]):
                    f[i, j] = f_eq(rPts[0], v, CN0, kN0, deltaRN0, rp, CTi,
                                   kTi, deltaRTi)
                elif (endPts_k2_r[i, j] > rMax):
                    f[i, j] = f_eq(endPts_k2_r[i, j], v, CN0, kN0,
                                   deltaRN0, rp, CTi, kTi, deltaRTi)
                else:
                    endPts_k2_q[i, j] = endPts_k2_q[i, j] % (2*pi)
                    f[i, j] = pol_spline.eval(endPts_k2_q[i, j], endPts_k2_r[i, j])


def v_parallel_advection_eval_step(f: 'float[:]', vPts: 'float[:]',
                                           rPos: 'float', vMin: 'float', vMax: 'float',
                                           spl : Spline1D,
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
                f[i] = spl.eval(v)
    elif (bound == 1):
        for i, v in enumerate(vPts):
            if (v < vMin or v > vMax):
                f[i] = 0.0
            else:
                f[i] = spl.eval(v)
    elif (bound == 2):
        vDiff = vMax - vMin
        for i, v in enumerate(vPts):
            while (v < vMin):
                v += vDiff
            while (v > vMax):
                v -= vDiff
            f[i] = spl.eval(v)

def v_parallel_advection_eval_step_loop(f: 'float[:,:,:]', vPts: 'float[:]',
                                           rPos: 'float', vMin: 'float', vMax: 'float',
                                        spl : Spline1D, bmat : 'float[:,:](order=F)', l : np.int32, u : np.int32, ipiv : 'int32[:]',
                                        parGradVals : 'float[:,:,:]', dt : float, i : int,
                                           CN0: 'float', kN0: 'float', deltaRN0: 'float', rp: 'float',
                                           CTi: 'float', kTi: 'float', deltaRTi: 'float', bound: 'int'):
    n1, n2, _ = f.shape
    #$ omp parallel for collapse(2) firstprivate(spl, vPts) private(coeffs)
    for j in range(n1):  # z
        for k in range(n2):  # q
            coeffs = spl.coeffs
            solve_system_nonperiodic(f[j,k,:], coeffs, bmat, l, u, ipiv)

            vPts -= parGradVals[i, j, k]*dt

            v_parallel_advection_eval_step(f[j,k,:], vPts, rPos, vMin, vMax, spl,
                                           CN0, kN0, deltaRN0, rp, CTi, kTi, deltaRTi, bound)


def get_lagrange_vals(i: 'int', shifts: 'int[:]',
                              vals: 'float[:,:,:]', qVals: 'float[:]',
                              thetaShifts: 'float[:]', spl : Spline1D):
    """
    TODO
    """
    from numpy import pi, empty_like
    nz = vals.shape[0]
    new_q = empty_like(qVals)

    for j, s in enumerate(shifts):
        idx = (i - s) % nz
        new_q[:] = (qVals + thetaShifts[j]) % (2*pi)
        spl.eval_vector(new_q, vals[idx, :, j])


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

def flux_advection_loop(f : 'float[:,:,:,:]', thetaSpline : Spline1D, theta_offset : int, theta_splu : PeriodicBandedMatrix,
                        shifts : 'int[:,:,:]', lagrange_vals: 'float[:,:,:]', qVals: 'float[:]',
                        thetaShifts: 'float[:,:,:]', lagrange_coeffs : 'float[:,:,:]'):
    nr, nv, nq, nz = f.shape

    #$ omp parallel for collapse(2) firstprivate(thetaSpline)
    for rIdx in range(nr):  # r
        for cIdx in range(nv):  # v
            # find the values of the function at each required point
            for k in range(nz):
                solve_system_periodic(f[rIdx, cIdx, :, k], thetaSpline, theta_offset, theta_splu)

                get_lagrange_vals(k, shifts[rIdx, cIdx],
                                  lagrange_vals, qVals,
                                  thetaShifts[rIdx, cIdx],
                                  thetaSpline)

            flux_advection(nq, nz, f[rIdx, cIdx],
                           lagrange_coeffs[rIdx, cIdx],
                           lagrange_vals)


def poloidal_advection_step_impl(f: 'float[:,:]', dt: 'float', v: 'float', rPts: 'float[:]', qPts: 'float[:]',
                                         drPhi_0: 'float[:,:]', dthetaPhi_0: 'float[:,:]', drPhi_k: 'float[:,:]', dthetaPhi_k: 'float[:,:]',
                                         endPts_k1_q: 'float[:,:]', endPts_k1_r: 'float[:,:]', endPts_k2_q: 'float[:,:]', endPts_k2_r: 'float[:,:]',
                                         phi_spline : Spline2D, pol_spline : Spline2D,
                                         CN0: 'float', kN0: 'float', deltaRN0: 'float', rp: 'float', CTi: 'float', kTi: 'float', deltaRTi: 'float',
                                         B0: 'float', tol: 'float', nulBound: 'bool'):
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

    phi_spline.eval_vector(qPts, rPts, drPhi_0, 0, 1)
    phi_spline.eval_vector(qPts, rPts, dthetaPhi_0, 1, 0)

    nPts_r = rPts.shape[0]
    nPts_q = qPts.shape[0]

    idx = nPts_r-1
    rMax = rPts[idx]

    #$ omp parallel for collapse(2)
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
        #$ omp parallel for collapse(2)
        for i in range(nPts_q):
            for j in range(nPts_r):
                # Handle theta boundary conditions
                endPts_k1_q[i, j] = endPts_k1_q[i, j] % (2*pi)

                if (not (endPts_k1_r[i, j] < rPts[0] or
                         endPts_k1_r[i, j] > rMax)):
                    # Add the new value of phi to the derivatives
                    # x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
                    #                               ^^^^^^^^^^^^^^^
                    drPhi_k[i, j] = phi_spline.eval(endPts_k1_q[i, j], endPts_k1_r[i, j], 0, 1)
                    drPhi_k[i, j] /= endPts_k1_r[i, j]
                    dthetaPhi_k[i, j] = phi_spline.eval(endPts_k1_q[i, j], endPts_k1_r[i, j], 1, 0)
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
                if diff > pi:
                    diff = 2*pi - diff
                if (diff > norm):
                    norm = diff
                diff = abs(endPts_k2_r[i, j]-endPts_k1_r[i, j])
                if (diff > norm):
                    norm = diff
                endPts_k1_q[i, j] = endPts_k2_q[i, j]
                endPts_k1_r[i, j] = endPts_k2_r[i, j]

    # Find value at the determined point
    if (nulBound):
        #$ omp parallel for collapse(2)
        for i in range(nPts_q):
            for j in range(nPts_r):
                if (endPts_k2_r[i, j] < rPts[0]):
                    f[i, j] = 0.0
                elif (endPts_k2_r[i, j] > rMax):
                    f[i, j] = 0.0
                else:
                    endPts_k2_q[i, j] = endPts_k2_q[i, j] % (2*pi)
                    f[i, j] = pol_spline.eval(endPts_k2_q[i, j], endPts_k2_r[i, j])
    else:
        #$ omp parallel for collapse(2)
        for i in range(nPts_q):
            for j in range(nPts_r):
                if (endPts_k2_r[i, j] < rPts[0]):
                    f[i, j] = f_eq(rPts[0], v, CN0, kN0, deltaRN0, rp, CTi,
                                   kTi, deltaRTi)
                elif (endPts_k2_r[i, j] > rMax):
                    f[i, j] = f_eq(endPts_k2_r[i, j], v, CN0, kN0,
                                   deltaRN0, rp, CTi, kTi, deltaRTi)
                else:
                    endPts_k2_q[i, j] = endPts_k2_q[i, j] % (2*pi)
                    f[i, j] = pol_spline.eval(endPts_k2_q[i, j], endPts_k2_r[i, j])

def poloidal_advection_loop(f: 'float[:,:,:,:]', phi: 'float[:,:,:]', dt: 'float',
                            is_explicit : 'bool', rPts: 'float[:]', qPts: 'float[:]',
                            vPts: 'float[:]', interp_wt : 'float[:,:]',
                            r_bmat : 'float[:,:](order=F)', r_l : np.int32, r_u : np.int32,
                            r_ipiv : 'int32[:]', theta_offset : int, theta_splu : PeriodicBandedMatrix,
                            drPhi_0: 'float[:,:]', dthetaPhi_0: 'float[:,:]', drPhi_k: 'float[:,:]', dthetaPhi_k: 'float[:,:]',
                            endPts_k1_q: 'float[:,:]', endPts_k1_r: 'float[:,:]', endPts_k2_q: 'float[:,:]', endPts_k2_r: 'float[:,:]',
                            phi_spline : Spline2D, pol_spline : Spline2D,
                            CN0: 'float', kN0: 'float', deltaRN0: 'float', rp: 'float', CTi: 'float', kTi: 'float', deltaRTi: 'float',
                            B0: 'float', tol: 'float', nulBound: 'bool'):
    nv, nz, _, _ = f.shape
    ##$ omp parallel for firstprivate(phi_spline, pol_spline) private(interp_wt, drPhi_0, dthetaPhi_0, drPhi_k, dthetaPhi_k, endPts_k1_q, endPts_k1_r, endPts_k2_q, endPts_k2_r)
    for j in range(nz):
        solve_2d_system(phi[j], phi_spline, interp_wt, r_bmat, r_l, r_u,
                        r_ipiv, theta_offset, theta_splu)

        # Do step
        for i, v in enumerate(vPts):
            solve_2d_system(f[i,j], pol_spline, interp_wt, r_bmat, r_l, r_u,
                        r_ipiv, theta_offset, theta_splu)

            if is_explicit:
                poloidal_advection_step_expl(f[i,j], float(dt), v, rPts, qPts,
                                             drPhi_0, dthetaPhi_0, drPhi_k, dthetaPhi_k,
                                             endPts_k1_q, endPts_k1_r, endPts_k2_q, endPts_k2_r,
                                             phi_spline, pol_spline, CN0, kN0, deltaRN0, rp, CTi,
                                             kTi, deltaRTi, B0, nulBound)
            else:
                poloidal_advection_step_impl(f[i,j], float(dt), v, rPts, qPts,
                                             drPhi_0, dthetaPhi_0, drPhi_k, dthetaPhi_k,
                                             endPts_k1_q, endPts_k1_r, endPts_k2_q, endPts_k2_r,
                                             phi_spline, pol_spline, CN0, kN0, deltaRN0, rp, CTi,
                                             kTi, deltaRTi, B0, tol, nulBound)
