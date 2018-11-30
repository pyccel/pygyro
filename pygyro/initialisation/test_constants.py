import numpy as np
from scipy          import integrate
from math           import exp, tanh, pi

from .constants     import get_constants

def test_constants():
    constants = get_constants('testSetups/iota0.json')
    
    assert(constants.B0 == 1.0)
    assert(constants.R0 == 239.8081535)
    assert(constants.rMin == 0.1)
    assert(constants.rMax == 14.5)
    assert(constants.zMin == 0.0)
    assert(constants.zMax == constants.R0*2*pi)
    assert(constants.vMax == 7.32)
    assert(constants.vMin == -constants.vMax)
    assert(constants.rp == 0.5*(constants.rMin + constants.rMax))
    assert(constants.eps == 1e-6)
    assert(constants.eps0 == 8.854187817e-12)
    assert(constants.kN0 == 0.055)
    assert(constants.kTi == 0.27586)
    assert(constants.kTe == float(constants.kTi))
    assert(constants.deltaRTi == 1.45)
    assert(constants.deltaRTe == float(constants.deltaRTi))
    assert(constants.deltaRN0 == 2.0*constants.deltaRTe)
    assert(constants.deltaR == 4.0*constants.deltaRN0/constants.deltaRTi)
    assert(constants.CTi == 1.0)
    assert(constants.CTe == float(constants.CTi))
    assert(constants.m == 15)
    assert(constants.n == 1)
    assert(constants.iotaVal == 0.0)

    def normalisingFunc(r):
        return exp(-constants.kN0*constants.deltaRN0*tanh((r-constants.rp)/constants.deltaRN0))

    assert(constants.CN0 == (constants.rMax-constants.rMin)/integrate.quad(normalisingFunc,constants.rMin,constants.rMax)[0])
