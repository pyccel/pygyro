from mpi4py import MPI
import os
import inspect
import numpy        as np
import h5py

from ..initialisation          import constants

def setupSave(rDegree,qDegree,zDegree,vDegree,npts,dt,foldername: str = None):
    if (foldername==None):
        i=0
        foldername="simulation_{0}".format(i)
        while(os.path.isdir(foldername)):
            i+=1
            foldername="simulation_{0}".format(i)
        os.mkdir(foldername)
    else:
        if (not os.path.isdir(foldername)):
            os.mkdir(foldername)
    filename = "{0}/initParams.h5".format(foldername)
    saveFile = h5py.File(filename,'w',driver='mpio',comm=MPI.COMM_WORLD)
    printParameters(saveFile,rDegree,qDegree,zDegree,vDegree,npts,dt)
    saveFile.close()
    return foldername

def printParameters(file,rDegree,qDegree,zDegree,vDegree,npts,dt):
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
    file.attrs['dt'] = dt
