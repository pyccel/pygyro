#~ from pyccel.decorators import types

def types(*args):
    def id(f):
        return f
    return id

@types('float','float','float','float','float')
def n0(r,CN0,kN0,deltaRN0,rp):
    from numpy import exp, tanh
    return CN0*exp(-kN0*deltaRN0*tanh((r-rp)/deltaRN0))

@types('float','float','float','float','float')
def Ti(r,CTi,kTi,deltaRTi,rp):
    from numpy import exp, tanh
    return CTi*exp(-kTi*deltaRTi*tanh((r-rp)/deltaRTi))

@types('float','float','float','int','int','float','float','float')
def perturbation(r,theta,z,m,n,rp,deltaR,R0):
    from numpy import exp, cos
    return exp(-(r-rp)**2/deltaR)*cos(m*theta+n*z/R0)

@types('float','float','float','float','float','float','float','float','float')
def fEq(r,vPar,CN0,kN0,deltaRN0,rp,CTi,kTi,deltaRTi):
    from numpy import exp,sqrt, pi
    return n0(r,CN0,kN0,deltaRN0,rp)*exp(-0.5*vPar*vPar/Ti(r,CTi,kTi,deltaRTi,rp))/sqrt(2*pi*Ti(r,CTi,kTi,deltaRTi,rp))

@types('float','float','float','float')
def n0derivNormalised(r,kN0,rp,deltaRN0):
    from numpy import tanh
    return -kN0 * (1 - tanh( (r - rp ) / deltaRN0 )**2)

@types('float','float','float','float','float')
def Te(r,CTe,kTe,deltaRTe,rp):
    from numpy import exp, tanh
    return CTe*exp(-kTe*deltaRTe*tanh((r-rp)/deltaRTe))

