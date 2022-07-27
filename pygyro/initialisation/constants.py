import numpy as np
from scipy import integrate
import math
import json
import re

from .default_constants import defaults


class Constants:
    """
    TODO
    """
    B0 = None
    R0 = None
    _rMin = None
    _rMax = None
    zMin = None
    zMax = None
    vMax = None
    vMin = None
    rp = None
    eps = None
    eps0 = None
    kN0 = None
    kTi = None
    kTe = None
    deltaRTi = None
    deltaRTe = None
    deltaRN0 = None
    deltaR = None
    CTi = None
    CTe = None
    m = None
    n = None
    iotaVal = None
    CN0 = None
    _splineDegrees = None
    _npts = None
    dt = None

    def __init__(self, setup=True):
        if (setup):
            self.set_defaults()
            if (self.CN0 is None):
                self.getCN0()

    @property
    def rMin(self):
        return self._rMin

    @rMin.setter
    def rMin(self, x):
        self._rMin = x
        if (self._rMax is not None):
            self.rp = 0.5*(self._rMin + self._rMax)

    @property
    def rMax(self):
        return self._rMax

    @rMax.setter
    def rMax(self, x):
        self._rMax = x
        if (self._rMin is not None):
            self.rp = 0.5*(self._rMin + self._rMax)

    @property
    def npts(self):
        return self._npts

    @npts.setter
    def npts(self, x):
        self._npts = x
        if (self._splineDegrees is not None):
            assert(len(self._npts) == len(self._splineDegrees))

    @property
    def splineDegrees(self):
        return self._splineDegrees

    @splineDegrees.setter
    def splineDegrees(self, x):
        self._splineDegrees = x
        if (self._npts is not None):
            assert(len(self._npts) == len(self._splineDegrees))

    def iota(self, r=rp):
        return np.full_like(r, self.iotaVal, dtype=float)

    def getCN0(self):
        self.CN0 = (self.rMax-self.rMin) / \
            integrate.quad(self.normalisingFunc, self.rMin, self.rMax)[0]

    def normalisingFunc(self, r):
        return math.exp(-self.kN0*self.deltaRN0*math.tanh((r-self.rp)/self.deltaRN0))

    def set_defaults(self):
        for key, val in defaults.items():
            if (getattr(self, key) is None):
                setattr(self, key, val)

    def __str__(self):
        s = "{\n"
        for obj in dir(self):
            val = getattr(self, obj)
            if not callable(val) and obj[0] != '_':
                s += "\""+obj+"\":"+"{}".format(val)+",\n"
        s = s[:-2]+"\n}"
        return s


def eval_expr(mystr, constants):
    f = re.split('([+*/\\-\\(\\)])', mystr.replace(" ", ""))
    for i, el in enumerate(f):
        if (hasattr(constants, el)):
            val = getattr(constants, el)
            if (val is not None):
                f[i] = str(val)
            else:
                return None
        elif (el not in '([+*/\\-\\(\\)])'):
            try:
                float(el)
            except ValueError:
                f[i] = str(getattr(math, el))
    return eval(''.join(f))


def get_constants(filename):
    constants = Constants(False)
    with open(filename) as f:
        data = json.load(f)
    unmatched = {}
    n = len(data)
    while (len(data) > 0):
        while (len(data) > 0):
            item = data.popitem()
            if (not isinstance(item[1], str)):
                setattr(constants, item[0], item[1])
            else:
                res = eval_expr(item[1], constants)
                if (res is None):
                    assert(len(data) > 0 or len(unmatched))
                    unmatched[item[0]] = item[1]
                else:
                    setattr(constants, item[0], res)
        data, unmatched = unmatched, data
        assert(len(data) < n)
        n = len(data)
    constants.set_defaults()
    if (constants.CN0 is None):
        constants.getCN0()
    return constants

# ~ CN0 = 0.14711120412124
