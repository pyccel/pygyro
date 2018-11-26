import numpy as np
from scipy import integrate
from math import exp, tanh, pi

B0 = 1.0
R0 = 239.8081535
rMin = 0.1
rMax = 14.5
zMin = 0.0
zMax = R0*2*pi
vMax = 7.32
vMin = -vMax
rp = 0.5*(rMin + rMax)
eps = 1e-6
eps0 = 8.854187817e-12
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
n = 1
iotaVal = 0.0

def iota(r = rp):
    return np.full_like(r,iotaVal,dtype=float)

def normalisingFunc(r):
    return exp(-kN0*deltaRN0*tanh((r-rp)/deltaRN0))

CN0 = (rMax-rMin)/integrate.quad(normalisingFunc,rMin,rMax)[0]
#~ CN0 = 0.14711120412124
