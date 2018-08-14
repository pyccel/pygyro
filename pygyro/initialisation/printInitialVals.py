import inspect
import numpy        as np

from .          import constants

def getParameters(file,rDegree,qDegree,zDegree,vDegree,npts):
    constStr = inspect.getsource(constants).split('\n')[4:]
    constGroup = file.create_group('constants')
    for s in constStr:
        n=s.split(" = ")
        try:
            if (len(n)<2):
                continue
            val = float(n[1])
            constGroup.attrs[n[0]] = val
        except ValueError:
            pass
    degs = file.create_group('degrees')
    degs.attrs.create('r',rDegree,dtype=int)
    degs.attrs.create('theta',qDegree,dtype=int)
    degs.attrs.create('z',zDegree,dtype=int)
    degs.attrs.create('v',vDegree,dtype=int)
    file.attrs['npts'] = npts
