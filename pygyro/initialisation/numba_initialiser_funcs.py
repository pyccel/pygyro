from numba              import njit
from numpy              import exp, tanh, cos, sqrt, pi

@njit
def n0(r,CN0,kN0,deltaRN0,rp):
    return CN0*exp(-kN0*deltaRN0*tanh((r-rp)/deltaRN0))

@njit
def ti(r,Cti,kti,deltaRti,rp):
    return Cti*exp(-kti*deltaRti*tanh((r-rp)/deltaRti))

@njit
def perturbation(r,theta,z,m,n,rp,deltaR,R0):
    return exp(-(r-rp)**2/deltaR)*cos(m*theta+n*z/R0)

@njit
def f_eq(r,vPar,CN0,kN0,deltaRN0,rp,Cti,kti,deltaRti):
    return n0(r,CN0,kN0,deltaRN0,rp)*exp(-0.5*vPar*vPar/ti(r,Cti,kti,deltaRti,rp))/sqrt(2.0*pi*ti(r,Cti,kti,deltaRti,rp))

@njit
def n0deriv_normalised(r,kN0,rp,deltaRN0):
    return -kN0 * (1 - tanh( (r - rp ) / deltaRN0 )**2)

@njit
def te(r,Cte,kte,deltaRte,rp):
    return Cte*exp(-kte*deltaRte*tanh((r-rp)/deltaRte))

