from scipy import integrate
from math import exp, tanh

B0 = 1.0
R0 = 239.8081535
rMin = 0.1
rMax = 14.5
rp = 0.5*(rMin + rMax)
eps = 1e-6
kN0 = 0.055
kTi = 0.27586
kTe = float(kTi)
deltaRTi = 1.45
deltaRTe = float(deltaRTi)
deltaRN0 = 2.0*deltaRTe
deltaR = 4.0*deltaRN0/deltaRTi
CTi = 1.0
CTe = float(CTi)
m = 15
n = -11


def normalisingFunc(r):
    return exp(-kN0*deltaRN0*tanh((r-rp)/deltaRN0))

CN0 = (rMax-rMin)/integrate.quad(normalisingFunc,rMin,rMax)[0]
