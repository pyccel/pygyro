import numpy as np
from scipy import integrate
from math import exp, tanh, pi
import json
import re

class Constants:
    B0 = None
    R0 = None
    rMin = None
    rMax = None
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

    def iota(self,r = rp):
        return np.full_like(r,self.iotaVal,dtype=float)
    
    def getCN0(self):
        self.CN0 = (self.rMax-self.rMin)/integrate.quad(self.normalisingFunc,self.rMin,self.rMax)[0]
    
    def normalisingFunc(self,r):
        return exp(-self.kN0*self.deltaRN0*tanh((r-self.rp)/self.deltaRN0))

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
    constants=Constants()
    with open(filename) as f:
        data = json.load(f)
    unmatched={}
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
    if (constants.CN0==None):
        constants.getCN0()
    return constants

#~ CN0 = 0.14711120412124
