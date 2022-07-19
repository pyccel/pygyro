from numba import njit
from numpy import exp, tanh, cos, sqrt, pi


@njit
def n0(r, CN0, kN0, deltaRN0, rp):
    """
    TODO
    """
    return CN0 * exp(- kN0 * deltaRN0 * tanh((r - rp) / deltaRN0))


@njit
def Ti(r, Cti, kti, deltaRti, rp):
    """
    TODO
    """
    return Cti * exp(- kti * deltaRti * tanh((r - rp) / deltaRti))


@njit
def perturbation(r, theta, z, m, n, rp, deltaR, R0):
    """
    TODO
    """
    return exp(-(r - rp)**2 / deltaR) * cos(m * theta + n * z / R0)


@njit
def f_eq(r, vPar, CN0, kN0, deltaRN0, rp, Cti, kti, deltaRti):
    """
    TODO
    """
    return n0(r, CN0, kN0, deltaRN0, rp) * exp(-0.5 * vPar * vPar / Ti(r, Cti, kti, deltaRti, rp)) \
        / sqrt(2.0 * pi * Ti(r, Cti, kti, deltaRti, rp))


@njit
def n0deriv_normalised(r, kN0, rp, deltaRN0):
    """
    TODO
    """
    return -kN0 * (1 - tanh((r - rp) / deltaRN0)**2)


@njit
def Te(r, Cte, kte, deltaRte, rp):
    """
    TODO
    """
    return Cte * exp(-kte * deltaRte * tanh((r - rp) / deltaRte))
