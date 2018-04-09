from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum, IntEnum

class Layout(Enum):
    RADIAL = 1
    BLOCK = 2

class Grid(object):
    class Dimension(IntEnum):
        R = 2
        THETA = 1
        Z = 3
        V = 0
    
    def __init__(self,r,rSpline,theta,thetaSpline,z,zSpline,v,vSpline,rStarts,zStarts,f,layout):
        if (type(layout)==str):
            if (layout=="radial"):
                self.layout=Layout.RADIAL
            elif(layout=="block"):
                self.layout=Layout.BLOCK
            else:
                raise NotImplementedError("%s is not an implemented layout" % layout)
        else:
            self.layout=layout
        self.rVals=r
        self.rSpline=rSpline
        self.thetaVals=theta
        self.thetaSpline=thetaSpline
        self.vVals=v
        self.vSpline=vSpline
        self.zVals=z
        self.zSpline=zSpline
        self.f=f
        self.rStarts=rStarts
        self.zStarts=zStarts
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.mpi_size = MPI.COMM_WORLD.Get_size()
    
    def size(self):
        return self.f.size
    
    def swapLayout(self):
        if (self.layout==Layout.RADIAL):
            self.f = np.concatenate(MPI.COMM_WORLD.alltoall(np.split(self.f,self.zStarts[1:self.mpi_size],axis=3)),axis=2)
            self.layout=Layout.BLOCK
        elif (self.layout==Layout.BLOCK):
            self.f = np.concatenate(MPI.COMM_WORLD.alltoall(np.split(self.f,self.rStarts[1:self.mpi_size],axis=2)),axis=3)
            self.layout=Layout.RADIAL
        else:
            raise NotImplementedError("%s is not an implemented layout" % self.layout)
    
    def printGrid(self):
        if (self.layout==Layout.RADIAL):
            print("to do")
        elif (self.layout==Layout.BLOCK):
            print("to do")
        else:
            raise NotImplementedError("%s is not an implemented layout" % self.layout)
    
    
    def getSliceFromDict(self,d : dict):
        r = d.get(self.Dimension.R,None)
        q = d.get(self.Dimension.THETA,None)
        v = d.get(self.Dimension.V,None)
        z = d.get(self.Dimension.Z,None)
        return self.getSlice(r,q,z,v)
    
    def getSlice(self,r = None,theta = None,z = None,v = None):
        finalSize=1
        if (r==None):
            finalSize=finalSize*len(self.rVals)
            if (self.layout==Layout.BLOCK):
                rVal=slice(0,len(self.rVals))
            elif (self.layout==Layout.RADIAL):
                rVal=slice(0,self.rStarts[self.rank+1]-self.rStarts[self.rank])
            else:
                raise NotImplementedError("%s is not an implemented layout" % self.layout)
        else:
            if (self.layout==Layout.BLOCK):
                rVal = (np.abs(r-self.rVals)).argmin()
            elif (self.layout==Layout.RADIAL):
                rVal=None
                for i in range(self.rStarts[self.rank],self.rStarts[self.rank+1]):
                    if (self.rVals[i] == r):
                        rVal=i-self.rStarts[self.rank]
                        break
            else:
                raise NotImplementedError("%s is not an implemented layout" % self.layout)
        if (theta==None):
            finalSize=finalSize*len(self.thetaVals)
            thetaVal=slice(0,len(self.thetaVals))
        else:
            thetaVal=(np.abs(theta-self.thetaVals)).argmin()
        if (z==None):
            finalSize=finalSize*len(self.zVals)
            if (self.layout==Layout.RADIAL):
                zVal=slice(0,len(self.zVals))
            elif(self.layout==Layout.BLOCK):
                zVal=slice(0,self.zStarts[self.rank+1]-self.zStarts[self.rank])
            else:
                raise NotImplementedError("%s is not an implemented layout" % self.layout)
        else:
            if (self.layout==Layout.RADIAL):
                zVal = (np.abs(z-self.zVals)).argmin()
            elif (self.layout==Layout.BLOCK):
                zVal=None
                for i in range(self.zStarts[self.rank],self.zStarts[self.rank+1]):
                    if (self.zVals[i] == z):
                        zVal=i-self.zStarts[self.rank]
                        break
            else:
                raise NotImplementedError("%s is not an implemented layout" % self.layout)
        if (v==None):
            finalSize=finalSize*len(self.vVals)
            vVal=slice(0,len(self.vVals))
        else:
            vVal = (np.abs(v-self.vVals)).argmin()
        
        if (self.layout==Layout.BLOCK):
            if (zVal==None):
                sendSize=0
                toSend = np.ndarray(0)
            else:
                toSend = self.f[vVal,thetaVal,rVal,zVal]
                sendSize=toSend.size
            
            sizes=MPI.COMM_WORLD.gather(sendSize,root=0)
            mySlice = None
            if (self.rank==0):
                starts=np.zeros(len(sizes))
                starts[1:]=sizes[:self.mpi_size-1]
                starts=starts.cumsum().astype(int)
                mySlice = np.empty(finalSize, dtype=float)
                MPI.COMM_WORLD.Gatherv(toSend.reshape(sendSize),(mySlice, sizes, starts, MPI.DOUBLE), 0)
                if (v==None and theta==None):
                    mySlice=mySlice.reshape(len(self.vVals),len(self.thetaVals))
                    return np.append(mySlice,mySlice[:,0,None],axis=1)
                elif (v==None and r==None):
                    return mySlice.reshape(len(self.vVals),len(self.rVals))
                elif (v==None and z==None):
                    mySlice=np.split(mySlice,starts[1:])
                    vLen=len(self.vVals)
                    for i in range(0,self.mpi_size):
                        mySlice[i]=mySlice[i].reshape(vLen,sizes[i]//vLen)
                    return np.concatenate(mySlice,axis=1)
                elif (theta==None and r==None):
                    mySlice=mySlice.reshape(len(self.thetaVals),len(self.rVals))
                    return np.append(mySlice,mySlice[None,0,:],axis=0)
                elif (theta==None and z==None):
                    mySlice=np.split(mySlice,starts[1:])
                    qLen=len(self.thetaVals)
                    for i in range(0,self.mpi_size):
                        mySlice[i]=mySlice[i].reshape(qLen,sizes[i]//qLen)
                    mySlice=np.concatenate(mySlice,axis=1)
                    return np.append(mySlice,mySlice[None,0,:],axis=0)
                elif (r==None and z==None):
                    mySlice=np.split(mySlice,starts[1:])
                    rLen=len(self.rVals)
                    for i in range(0,self.mpi_size):
                        mySlice[i]=mySlice[i].reshape(rLen,sizes[i]//rLen)
                    return np.concatenate(mySlice,axis=1)
                else:
                    print("r = ",r)
                    print("theta = ",theta)
                    print("z = ",z)
                    print("v = ",v)
                    raise NotImplementedError()
            else:
                MPI.COMM_WORLD.Gatherv(toSend.reshape(sendSize),toSend,0)
                return mySlice
        elif (self.layout ==Layout.RADIAL):
            if (rVal==None):
                sendSize=0
                toSend = np.ndarray(0)
            else:
                toSend = self.f[vVal,thetaVal,rVal,zVal]
                sendSize = toSend.size
            
            mySlice = None
            sizes=MPI.COMM_WORLD.gather(sendSize,root=0)
            if (self.rank==0):
                starts=np.zeros(len(sizes))
                starts[1:]=sizes[:self.mpi_size-1]
                starts=starts.cumsum().astype(int)
                mySlice = np.empty(finalSize, dtype=float)
                MPI.COMM_WORLD.Gatherv(toSend.reshape(sendSize),(mySlice, sizes, starts, MPI.DOUBLE), 0)
                if (v==None and theta==None):
                    mySlice=mySlice.reshape(len(self.vVals),len(self.thetaVals))
                    return np.append(mySlice,mySlice[:,0,None],axis=1)
                elif (v==None and r==None):
                    mySlice=np.split(mySlice,starts[1:])
                    vLen=len(self.vVals)
                    for i in range(0,self.mpi_size):
                        mySlice[i]=mySlice[i].reshape(vLen,sizes[i]//vLen)
                    return np.concatenate(mySlice,axis=1)
                elif (v==None and z==None):
                    return mySlice.reshape(len(self.vVals),len(self.zVals))
                elif (theta==None and r==None):
                    mySlice=np.split(mySlice,starts[1:])
                    qLen=len(self.thetaVals)
                    for i in range(0,self.mpi_size):
                        mySlice[i]=mySlice[i].reshape(qLen,sizes[i]//qLen)
                    mySlice=np.concatenate(mySlice,axis=1)
                    return np.append(mySlice,mySlice[None,0,:],axis=0)
                elif (theta==None and z==None):
                    mySlice=mySlice.reshape(len(self.thetaVals),len(self.zVals))
                    return np.append(mySlice,mySlice[None,0,:],axis=0)
                elif (r==None and z==None):
                    mySlice=np.split(mySlice,starts[1:])
                    zLen=len(self.zVals)
                    for i in range(0,self.mpi_size):
                        mySlice[i]=mySlice[i].reshape(sizes[i]//zLen,zLen)
                    return np.concatenate(mySlice,axis=0)
                else:
                    print("r = ",r)
                    print("theta = ",theta)
                    print("z = ",z)
                    print("v = ",v)
                    raise NotImplementedError()
            else:
                MPI.COMM_WORLD.Gatherv(toSend.reshape(sendSize),toSend,0)
            return mySlice
        else:
            raise NotImplementedError("%s is not an implemented layout" % self.layout)

    def getMin(self):
        return MPI.COMM_WORLD.reduce(np.amin(self.f),op=MPI.MIN,root=0)
    
    def getMax(self):
        return MPI.COMM_WORLD.reduce(np.amax(self.f),op=MPI.MAX,root=0)
