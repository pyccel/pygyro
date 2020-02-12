from numba              import njit
from numpy              import exp, tanh, cos, sqrt, pi

@njit
def n0(r,CN0,kN0,deltaRN0,rp):
    return CN0*exp(-kN0*deltaRN0*tanh((r-rp)/deltaRN0))

@njit
def Ti(r,CTi,kTi,deltaRTi,rp):
    return CTi*exp(-kTi*deltaRTi*tanh((r-rp)/deltaRTi))

@njit
def perturbation(r,theta,z,m,n,rp,deltaR,R0):
    return exp(-(r-rp)**2/deltaR)*cos(m*theta+n*z/R0)

@njit
def fEq(r,vPar,CN0,kN0,deltaRN0,rp,CTi,kTi,deltaRTi):
    return n0(r,CN0,kN0,deltaRN0,rp)*exp(-0.5*vPar*vPar/Ti(r,CTi,kTi,deltaRTi,rp))/sqrt(2.0*pi*Ti(r,CTi,kTi,deltaRTi,rp))

@njit
def n0derivNormalised(r,kN0,rp,deltaRN0):
    return -kN0 * (1 - tanh( (r - rp ) / deltaRN0 )**2)

@njit
def Te(r,CTe,kTe,deltaRTe,rp):
    return CTe*exp(-kTe*deltaRTe*tanh((r-rp)/deltaRTe))

