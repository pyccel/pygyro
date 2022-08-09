from scipy import integrate
from math import exp, tanh, pi
import os
import pytest

from .constants import get_constants, Constants


@pytest.mark.serial
def test_constants_file():
    """
    TODO
    """
    f = open("constants_test_file.json", "w")
    f.write("{\n\"B0\": 1.0,\n\"R0\":239.8081535,\n\"rMin\":0.1,\n\"rMax\":14.5,\n\
        \"zMin\":0.0,\n\"zMax\":\"R0*2*pi\",\n\"vMax\":7.32,\n\"vMin\":\"-vMax\",\n\
        \"eps\":1e-6,\n\"eps0\":8.854187817e-12,\n\"kN0\":0.055,\n\"kTi\":0.27586,\n\
        \"kTe\":\"kTi\",\n\"deltaRTi\":1.45,\n\"deltaRTe\":\"deltaRTi\",\n\
        \"deltaRN0\":\"2.0*deltaRTe\",\n\"deltaR\":\"4.0*deltaRN0/deltaRTi\",\n\
        \"CTi\":1.0,\n\"CTe\":\"CTi\",\n\"m\":15,\n\"n\":1,\n\"iotaVal\":0.0,\n\
        \"npts\":[256,512,32,128],\n\"splineDegrees\":[3,3,3,3],\n\"dt\":2\n}\n")
    f.close()

    constants = get_constants("constants_test_file.json")
    os.remove("constants_test_file.json")

    assert (constants.B0 == 1.0)
    assert (constants.R0 == 239.8081535)
    assert (constants.rMin == 0.1)
    assert (constants.rMax == 14.5)
    assert (constants.zMin == 0.0)
    assert (constants.zMax == constants.R0*2*pi)
    assert (constants.vMax == 7.32)
    assert (constants.vMin == -constants.vMax)
    assert (constants.rp == 0.5*(constants.rMin + constants.rMax))
    assert (constants.eps == 1e-6)
    assert (constants.eps0 == 8.854187817e-12)
    assert (constants.kN0 == 0.055)
    assert (constants.kTi == 0.27586)
    assert (constants.kTe == float(constants.kTi))
    assert (constants.deltaRTi == 1.45)
    assert (constants.deltaRTe == float(constants.deltaRTi))
    assert (constants.deltaRN0 == 2.0*constants.deltaRTe)
    assert (constants.deltaR == 4.0*constants.deltaRN0/constants.deltaRTi)
    assert (constants.CTi == 1.0)
    assert (constants.CTe == float(constants.CTi))
    assert (constants.m == 15)
    assert (constants.n == 1)
    assert (constants.iotaVal == 0.0)
    assert (constants.npts == [256, 512, 32, 128])
    assert (constants.splineDegrees == [3, 3, 3, 3])
    assert (constants.dt == 2)

    def normalisingFunc(r):
        return exp(-constants.kN0*constants.deltaRN0*tanh((r-constants.rp)/constants.deltaRN0))

    assert (constants.CN0 == (constants.rMax-constants.rMin) /
            integrate.quad(normalisingFunc, constants.rMin, constants.rMax)[0])


@pytest.mark.serial
def test_constants_defaults():
    """
    TODO
    """
    constants = Constants()

    assert (constants.B0 == 1.0)
    assert (constants.R0 == 239.8081535)
    assert (constants.rMin == 0.1)
    assert (constants.rMax == 14.5)
    assert (constants.zMin == 0.0)
    assert (constants.zMax == constants.R0*2*pi)
    assert (constants.vMax == 7.32)
    assert (constants.vMin == -constants.vMax)
    assert (constants.rp == 0.5*(constants.rMin + constants.rMax))
    assert (constants.eps == 1e-6)
    assert (constants.eps0 == 8.854187817e-12)
    assert (constants.kN0 == 0.055)
    assert (constants.kTi == 0.27586)
    assert (constants.kTe == float(constants.kTi))
    assert (constants.deltaRTi == 1.45)
    assert (constants.deltaRTe == float(constants.deltaRTi))
    assert (constants.deltaRN0 == 2.0*constants.deltaRTe)
    assert (constants.deltaR == 4.0*constants.deltaRN0/constants.deltaRTi)
    assert (constants.CTi == 1.0)
    assert (constants.CTe == float(constants.CTi))
    assert (constants.m == 15)
    assert (constants.n == 1)
    assert (constants.iotaVal == 0.0)
    assert (constants.npts == [256, 512, 32, 128])
    assert (constants.splineDegrees == [3, 3, 3, 3])
    assert (constants.dt == 2)

    def normalisingFunc(r):
        """
        TODO
        """
        return exp(-constants.kN0*constants.deltaRN0*tanh((r-constants.rp)/constants.deltaRN0))

    assert (constants.CN0 == (constants.rMax-constants.rMin) /
            integrate.quad(normalisingFunc, constants.rMin, constants.rMax)[0])
