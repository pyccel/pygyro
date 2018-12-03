import numpy as np
from scipy import integrate
from math import exp, tanh, pi
import json
import re

from .default_constants import defaults

class Constants:
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
    
    def __init__(self,setup = True):
        if (setup):
            self.set_defaults()
            if (self.CN0==None):
                self.getCN0()
    
    @property
    def rMin(self):
        return self._rMin

    @rMin.setter
    def rMin(self, x):
        self._rMin = x
        if (self._rMax!=None):
            self.rp = 0.5*(self._rMin + self._rMax)
    
    @property
    def rMax(self):
        return self._rMax

    @rMax.setter
    def rMax(self, x):
        self._rMax = x
        if (self._rMin!=None):
            self.rp = 0.5*(self._rMin + self._rMax)

    def iota(self,r = rp):
        return np.full_like(r,self.iotaVal,dtype=float)
    
    def getCN0(self):
        self.CN0 = (self.rMax-self.rMin)/integrate.quad(self.normalisingFunc,self.rMin,self.rMax)[0]
    
    def normalisingFunc(self,r):
        return exp(-self.kN0*self.deltaRN0*tanh((r-self.rp)/self.deltaRN0))
    
    def set_defaults(self):
        for key,val in defaults.items():
            if (getattr(self,key)==None):
                setattr(self,key,val)

def eval_expr(mystr,constants):
    f=re.split('([+*/\\-\\(\\)])', mystr.replace(" ", ""))
    for i,el in enumerate(f):
        if (hasattr(constants,el)):
            val=getattr(constants,el)
            if (val!=None):
                f[i]=str(val)
            else:
                return None
    return eval(''.join(f))
    

def get_constants(filename):
    constants=Constants(False)
    with open(filename) as f:
        data = json.load(f)
    unmatched={}
    n = len(data)
    while (len(data)>0):
        while (len(data)>0):
            item = data.popitem()
            if (type(item[1])!=str):
                setattr(constants,item[0],item[1])
            else:
                res=eval_expr(item[1],constants)
                if (res==None):
                    assert(len(data)>0 or len(unmatched))
                    unmatched[item[0]]=item[1]
                else:
                    setattr(constants,item[0],res)
        data,unmatched=unmatched,data
        assert(len(data)<n)
        n = len(data)
    constants.set_defaults()
    if (constants.CN0==None):
        constants.getCN0()
    return constants

#~ CN0 = 0.14711120412124
