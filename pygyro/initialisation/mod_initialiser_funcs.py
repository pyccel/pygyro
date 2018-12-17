from pyccel.decorators import types

@types('double','double','double','double','double')
def n0(r,CN0,kN0,deltaRN0,rp):
    from numpy import exp, tanh
    return CN0*exp(-kN0*deltaRN0*tanh((r-rp)/deltaRN0))

@types('double','double','double','double','double')
def Ti(r,CTi,kTi,deltaRTi,rp):
    from numpy import exp, tanh
    return CTi*exp(-kTi*deltaRTi*tanh((r-rp)/deltaRTi))

@types('double','double','double','int','int','double','double','double')
def perturbation(r,theta,z,m,n,rp,deltaR,R0):
    from numpy import exp, cos
    return exp(-(r-rp)**2/deltaR)*cos(m*theta+n*z/R0)

@types('double','double','double','double','double','double','double','double','double')
def fEq(r,vPar,CN0,kN0,deltaRN0,rp,CTi,kTi,deltaRTi):
    from numpy import exp, sqrt, pi
    return n0(r,CN0,kN0,deltaRN0,rp)*exp(-0.5*vPar*vPar/Ti(r,CTi,kTi,deltaRTi,rp))/sqrt(2.0*pi*Ti(r,CTi,kTi,deltaRTi,rp))

@types('double','double','double','double')
def n0derivNormalised(r,kN0,rp,deltaRN0):
    from numpy import tanh
    return -kN0 * (1 - tanh( (r - rp ) / deltaRN0 )**2)

@types('double','double','double','double','double')
def Te(r,CTe,kTe,deltaRTe,rp):
    from numpy import exp, tanh
    return CTe*exp(-kTe*deltaRTe*tanh((r-rp)/deltaRTe))

