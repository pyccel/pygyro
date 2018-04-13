from mpi4py import MPI
import numpy as np
from enum import Enum, IntEnum
from math import pi

from ..                 import splines as spl
from ..initialisation   import constants

class Layout(Enum):
    FIELD_ALIGNED = 1
    V_PARALLEL = 2
    POLOIDAL = 3

class Grid(object):
    class Dimension(IntEnum):
        R = 1
        THETA = 0
        Z = 2
        V = 3
    
    def __init__(self,r,rSpline,theta,thetaSpline,z,zSpline,v,vSpline,layout: Layout,
                    *,nProcR : int = 1,nProcZ : int = 1,nProcV = 1):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        self.layout=layout
        
        if (self.layout==Layout.FIELD_ALIGNED):
            self.sizeRV=nProcR
            self.sizeVZ=nProcV
        elif (self.layout==Layout.V_PARALLEL):
            self.sizeRV=nProcR
            self.sizeVZ=nProcZ
        elif (self.layout==Layout.POLOIDAL):
            self.sizeRV=nProcV
            self.sizeVZ=nProcZ
        else:
            raise NotImplementedError("Layout not implemented")
        
        if (size>1):
            assert(self.sizeRV*self.sizeVZ==size)
            topology = comm.Create_cart([self.sizeRV,self.sizeVZ], periods=[False, False])
            self.commRV = topology.Sub([True, False])
            self.commVZ = topology.Sub([False, True])
        else:
            self.commRV = MPI.COMM_WORLD
            self.commVZ = MPI.COMM_WORLD
        
        self.rankRV=self.commRV.Get_rank()
        self.rankVZ=self.commVZ.Get_rank()
        
        self.ranksRV=np.arange(0,self.sizeRV)
        self.ranksVZ=np.arange(0,self.sizeVZ)
        
        self.Vals = [theta, r, z, v]
        self.nr=len(r)
        self.nq=len(theta)
        self.nz=len(z)
        self.nv=len(v)
        
        self.redefineShape()
        
        # ordering chosen to increase step size to improve cache-coherency
        self.f = np.empty((self.nq,len(self.Vals[self.Dimension.R][self.rStart:self.rEnd]),
                len(self.Vals[self.Dimension.Z][self.zStart:self.zEnd]),
                len(self.Vals[self.Dimension.V][self.vStart:self.vEnd])),float,order='F')
        
        #self.rank = MPI.COMM_WORLD.Get_rank()
        #self.mpi_size = MPI.COMM_WORLD.Get_size()
        
        
        
    
        #initialise(f,rVals[rStarts[rank]:rEnd],qVals,zVals,vVals,m,n)
    
    def redefineShape(self):
        if (self.layout==Layout.FIELD_ALIGNED):
            nrOverflow=self.nr%self.sizeRV
            nvOverflow=self.nv%self.sizeVZ
            
            rStarts=self.nr//self.sizeRV*self.ranksRV + np.minimum(self.ranksRV,nrOverflow)
            vStarts=self.nv//self.sizeVZ*self.ranksVZ + np.minimum(self.ranksVZ,nvOverflow)
            rStarts=np.append(rStarts,self.nr)
            vStarts=np.append(vStarts,self.nv)
            self.rStart=rStarts[self.rankRV]
            self.zStart=0
            self.vStart=vStarts[self.rankVZ]
            self.rEnd=rStarts[self.rankRV+1]
            self.zEnd=len(self.Vals[self.Dimension.Z])
            self.vEnd=vStarts[self.rankVZ+1]
        elif (self.layout==Layout.V_PARALLEL):
            nrOverflow=self.nr%self.sizeRV
            nzOverflow=self.nz%self.sizeVZ
            
            rStarts=self.nr//self.sizeRV*self.ranksRV + np.minimum(self.ranksRV,nrOverflow)
            zStarts=self.nz//self.sizeVZ*self.ranksVZ + np.minimum(self.ranksVZ,nzOverflow)
            rStarts=np.append(rStarts,self.nr)
            zStarts=np.append(zStarts,self.nz)
            self.rStart=rStarts[self.rankRV]
            self.zStart=zStarts[self.rankVZ]
            self.vStart=0
            self.rEnd=rStarts[self.rankRV+1]
            self.zEnd=zStarts[self.rankVZ+1]
            self.vEnd=len(self.Vals[self.Dimension.V])
        elif (self.layout==Layout.POLOIDAL):
            nvOverflow=self.nv%self.sizeRV
            nzOverflow=self.nz%self.sizeVZ
            
            vStarts=self.nv//self.sizeRV*self.ranksRV + np.minimum(self.ranksRV,nvOverflow)
            zStarts=self.nz//self.sizeVZ*self.ranksVZ + np.minimum(self.ranksVZ,nzOverflow)
            vStarts=np.append(vStarts,self.nv)
            zStarts=np.append(zStarts,self.nz)
            self.vStart=vStarts[self.rankRV]
            self.zStart=zStarts[self.rankVZ]
            self.rStart=0
            self.vEnd=vStarts[self.rankRV+1]
            self.zEnd=zStarts[self.rankVZ+1]
            self.rEnd=len(self.Vals[self.Dimension.R])
    
    def size(self):
        return self.f.size
    
    def getRCoords(self):
        return enumerate(self.Vals[self.Dimension.R][self.rStart:self.rEnd])
    
    @property
    def rVals(self):
        return self.Vals[self.Dimension.R][self.rStart:self.rEnd]
    
    def getThetaCoords(self):
        return enumerate(self.Vals[self.Dimension.THETA])
    
    @property
    def thetaVals(self):
        return self.Vals[self.Dimension.THETA]
    
    def getZCoords(self):
        return enumerate(self.Vals[self.Dimension.Z][self.zStart:self.zEnd])
    
    @property
    def zVals(self):
        return self.Vals[self.Dimension.Z][self.zStart:self.zEnd]
    
    def getVCoords(self):
        return enumerate(self.Vals[self.Dimension.V][self.vStart:self.vEnd])
    
    @property
    def vVals(self):
        return self.Vals[self.Dimension.V][self.vStart:self.vEnd]
    
    def setLayout(self,new_layout: Layout):
        if (new_layout==self.layout):
            return
        if (self.layout==Layout.FIELD_ALIGNED):
            if (new_layout==Layout.V_PARALLEL):
                nzOverflow=self.nz%self.sizeVZ
                zStarts=self.nz//self.sizeVZ*self.ranksVZ + np.minimum(self.ranksVZ,nzOverflow)
                
                self.f = np.concatenate(
                            self.commVZ.alltoall(
                                np.split(self.f,zStarts[1:],axis=self.Dimension.Z)
                            )
                        ,axis=self.Dimension.V)
                self.layout = Layout.V_PARALLEL
                zStarts=np.append(zStarts,self.nz)
                self.zStart=zStarts[self.rankVZ]
                self.vStart=0
                self.zEnd=zStarts[self.rankVZ+1]
                self.vEnd=len(self.Vals[self.Dimension.V])
            elif (new_layout==Layout.POLOIDAL):
                self.setLayout(Layout.V_PARALLEL)
                self.setLayout(Layout.POLOIDAL)
                raise RuntimeWarning("Changing from Field Aligned layout to Poloidal layout requires two steps")
        elif (self.layout==Layout.POLOIDAL):
            if (new_layout==Layout.V_PARALLEL):
                nrOverflow=self.nr%self.sizeRV
                rStarts=self.nr//self.sizeRV*self.ranksRV + np.minimum(self.ranksRV,nrOverflow)
                
                self.f = np.concatenate(
                            self.commRV.alltoall(
                                np.split(self.f,rStarts[1:],axis=self.Dimension.R)
                            )
                        ,axis=self.Dimension.V)
                self.layout = Layout.V_PARALLEL
                rStarts=np.append(rStarts,self.nr)
                self.rStart=rStarts[self.rankRV]
                self.vStart=0
                self.rEnd=rStarts[self.rankRV+1]
                self.vEnd=len(self.Vals[self.Dimension.V])
            elif (new_layout==Layout.FIELD_ALIGNED):
                self.setLayout(Layout.V_PARALLEL)
                self.setLayout(Layout.FIELD_ALIGNED)
                raise RuntimeWarning("Changing from Poloidal layout to Field Aligned layout requires two steps")
        elif (self.layout==Layout.V_PARALLEL):
            if (new_layout==Layout.FIELD_ALIGNED):
                nvOverflow=self.nv%self.sizeVZ
                vStarts=self.nv//self.sizeVZ*self.ranksVZ + np.minimum(self.ranksVZ,nvOverflow)
                
                self.f = np.concatenate(
                            self.commVZ.alltoall(
                                np.split(self.f,vStarts[1:],axis=self.Dimension.V)
                            )
                        ,axis=self.Dimension.Z)
                self.layout = Layout.FIELD_ALIGNED
                vStarts=np.append(vStarts,self.nv)
                self.zStart=0
                self.vStart=vStarts[self.rankVZ]
                self.zEnd=len(self.Vals[self.Dimension.Z])
                self.vEnd=vStarts[self.rankVZ+1]
            elif (new_layout==Layout.POLOIDAL):
                nvOverflow=self.nv%self.sizeRV
                vStarts=self.nv//self.sizeRV*self.ranksRV + np.minimum(self.ranksRV,nvOverflow)
                
                self.f = np.concatenate(
                            self.commRV.alltoall(
                                np.split(self.f,vStarts[1:],axis=self.Dimension.V)
                            )
                        ,axis=self.Dimension.R)
                self.layout = Layout.POLOIDAL
                vStarts=np.append(vStarts,self.nv)
                self.vStart=vStarts[self.rankRV]
                self.rStart=0
                self.vEnd=vStarts[self.rankRV+1]
                self.rEnd=len(self.Vals[self.Dimension.R])
        
    
    def swapLayout(self):
        if (self.layout==Layout.RADIAL):
            self.f = np.concatenate(MPI.COMM_WORLD.alltoall(np.split(self.f,self.zStarts[1:self.mpi_size],axis=2)),axis=1)
            self.layout=Layout.BLOCK
        elif (self.layout==Layout.BLOCK):
            self.f = np.concatenate(MPI.COMM_WORLD.alltoall(np.split(self.f,self.rStarts[1:self.mpi_size],axis=1)),axis=2)
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
        return self.getSliceForFig(r,q,z,v)
    
    def getSliceForFig(self,r = None,theta = None,z = None,v = None):
        finalSize=1
        if (r==None):
            finalSize=finalSize*len(self.Vals[self.Dimension.R])
            if (self.layout==Layout.BLOCK):
                rVal=slice(0,len(self.Vals[self.Dimension.R]))
            elif (self.layout==Layout.RADIAL):
                rVal=slice(0,self.rStarts[self.rank+1]-self.rStarts[self.rank])
            else:
                raise NotImplementedError("%s is not an implemented layout" % self.layout)
        else:
            if (self.layout==Layout.BLOCK):
                rVal = r
            elif (self.layout==Layout.RADIAL):
                rVal=None
                if (r>=self.rStarts[self.rank] and r<self.rStarts[self.rank+1]):
                    rVal=r-self.rStarts[self.rank]
            else:
                raise NotImplementedError("%s is not an implemented layout" % self.layout)
        if (theta==None):
            finalSize=finalSize*len(self.Vals[self.Dimension.THETA])
            thetaVal=slice(0,len(self.Vals[self.Dimension.THETA]))
        else:
            thetaVal=theta
        if (z==None):
            finalSize=finalSize*len(self.Vals[self.Dimension.Z])
            if (self.layout==Layout.RADIAL):
                zVal=slice(0,len(self.Vals[self.Dimension.Z]))
            elif(self.layout==Layout.BLOCK):
                zVal=slice(0,self.zStarts[self.rank+1]-self.zStarts[self.rank])
            else:
                raise NotImplementedError("%s is not an implemented layout" % self.layout)
        else:
            if (self.layout==Layout.RADIAL):
                zVal = z
            elif (self.layout==Layout.BLOCK):
                zVal=None
                if (z>=self.zStarts[self.rank] and z<self.zStarts[self.rank+1]):
                    zVal=z-self.zStarts[self.rank]
            else:
                raise NotImplementedError("%s is not an implemented layout" % self.layout)
        if (v==None):
            finalSize=finalSize*len(self.Vals[self.Dimension.V])
            vVal=slice(0,len(self.Vals[self.Dimension.V]))
        else:
            vVal = v
        
        if (self.layout==Layout.BLOCK):
            if (zVal==None):
                sendSize=0
                toSend = np.ndarray(0)
            else:
                toSend = self.f[thetaVal,rVal,zVal,vVal].flatten()
                sendSize=toSend.size
            
            sizes=MPI.COMM_WORLD.gather(sendSize,root=0)
            mySlice = None
            if (self.rank==0):
                starts=np.zeros(len(sizes))
                starts[1:]=sizes[:self.mpi_size-1]
                starts=starts.cumsum().astype(int)
                mySlice = np.empty(finalSize, dtype=float)
                MPI.COMM_WORLD.Gatherv(toSend,(mySlice, sizes, starts, MPI.DOUBLE), 0)
                if (v==None and theta==None):
                    mySlice=mySlice.reshape(len(self.Vals[self.Dimension.THETA]),len(self.Vals[self.Dimension.V]))
                    return np.append(mySlice,mySlice[None,0,:],axis=0)
                elif (v==None and r==None):
                    return mySlice.reshape(len(self.Vals[self.Dimension.R]),len(self.Vals[self.Dimension.V]))
                elif (v==None and z==None):
                    mySlice=np.split(mySlice,starts[1:])
                    vLen=len(self.Vals[self.Dimension.V])
                    for i in range(0,self.mpi_size):
                        mySlice[i]=mySlice[i].reshape(sizes[i]//vLen,vLen)
                    return np.concatenate(mySlice,axis=0)
                elif (theta==None and r==None):
                    mySlice=mySlice.reshape(len(self.Vals[self.Dimension.THETA]),len(self.Vals[self.Dimension.R]))
                    return np.append(mySlice,mySlice[None,0,:],axis=0)
                elif (theta==None and z==None):
                    mySlice=np.split(mySlice,starts[1:])
                    qLen=len(self.Vals[self.Dimension.THETA])
                    for i in range(0,self.mpi_size):
                        mySlice[i]=mySlice[i].reshape(qLen,sizes[i]//qLen)
                    mySlice=np.concatenate(mySlice,axis=1)
                    return np.append(mySlice,mySlice[None,0,:],axis=0)
                elif (r==None and z==None):
                    mySlice=np.split(mySlice,starts[1:])
                    rLen=len(self.Vals[self.Dimension.R])
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
                MPI.COMM_WORLD.Gatherv(toSend,toSend,0)
                return mySlice
        elif (self.layout ==Layout.RADIAL):
            if (rVal==None):
                sendSize=0
                toSend = np.ndarray(0)
            else:
                toSend = self.f[thetaVal,rVal,zVal,vVal].flatten()
                sendSize = toSend.size
            
            mySlice = None
            sizes=MPI.COMM_WORLD.gather(sendSize,root=0)
            if (self.rank==0):
                starts=np.zeros(len(sizes))
                starts[1:]=sizes[:self.mpi_size-1]
                starts=starts.cumsum().astype(int)
                mySlice = np.empty(finalSize, dtype=float)
                MPI.COMM_WORLD.Gatherv(toSend,(mySlice, sizes, starts, MPI.DOUBLE), 0)
                if (v==None and theta==None):
                    mySlice=mySlice.reshape(len(self.Vals[self.Dimension.THETA]),len(self.Vals[self.Dimension.V]))
                    return np.append(mySlice,mySlice[:,0,None],axis=1)
                elif (v==None and r==None):
                    mySlice=np.split(mySlice,starts[1:])
                    vLen=len(self.Vals[self.Dimension.V])
                    for i in range(0,self.mpi_size):
                        mySlice[i]=mySlice[i].reshape(sizes[i]//vLen,vLen)
                    return np.concatenate(mySlice,axis=0)
                elif (v==None and z==None):
                    return mySlice.reshape(len(self.Vals[self.Dimension.Z]),len(self.Vals[self.Dimension.V]))
                elif (theta==None and r==None):
                    mySlice=np.split(mySlice,starts[1:])
                    qLen=len(self.Vals[self.Dimension.THETA])
                    for i in range(0,self.mpi_size):
                        mySlice[i]=mySlice[i].reshape(qLen,sizes[i]//qLen)
                    mySlice=np.concatenate(mySlice,axis=1)
                    return np.append(mySlice,mySlice[None,0,:],axis=0)
                elif (theta==None and z==None):
                    mySlice=mySlice.reshape(len(self.Vals[self.Dimension.THETA]),len(self.Vals[self.Dimension.Z]))
                    return np.append(mySlice,mySlice[None,0,:],axis=0)
                elif (r==None and z==None):
                    mySlice=np.split(mySlice,starts[1:])
                    zLen=len(self.Vals[self.Dimension.Z])
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
                MPI.COMM_WORLD.Gatherv(toSend,toSend,0)
            return mySlice
        else:
            raise NotImplementedError("%s is not an implemented layout" % self.layout)
    
    def getMin(self,axis = None,fixValue = None):
        if (axis==None and fixValue==None):
            return MPI.COMM_WORLD.reduce(np.amin(self.f),op=MPI.MIN,root=0)
        if (self.layout==Layout.BLOCK):
            if (axis==self.Dimension.Z):
                if (fixValue>=self.zStarts[self.rank] and fixValue<self.zStarts[self.rank+1]):
                    idx = (np.s_[:],) * axis + (fixValue-self.zStarts[self.rank],)
                    return MPI.COMM_WORLD.reduce(np.amin(self.f[idx]),op=MPI.MIN,root=0)
                else:
                    return MPI.COMM_WORLD.reduce(1,op=MPI.MIN,root=0)
            else:
                idx = (np.s_[:],) * axis + (fixValue,)
                return MPI.COMM_WORLD.reduce(np.amin(self.f[idx]),op=MPI.MIN,root=0)
        elif (self.layout==Layout.RADIAL):
            if (axis==self.Dimension.R):
                if (fixValue>=self.rStarts[self.rank] and fixValue<self.rStarts[self.rank+1]):
                    idx = (np.s_[:],) * axis + (fixValue-self.rStarts[self.rank],)
                    return MPI.COMM_WORLD.reduce(np.amin(self.f[idx]),op=MPI.MIN,root=0)
                else:
                    return MPI.COMM_WORLD.reduce(1,op=MPI.MIN,root=0)
            else:
                idx = (np.s_[:],) * axis + (fixValue,)
                return MPI.COMM_WORLD.reduce(np.amin(self.f[idx]),op=MPI.MIN,root=0)
    
    def getMax(self,axis = None,fixValue = None):
        if (axis==None and fixValue==None):
            return MPI.COMM_WORLD.reduce(np.amax(self.f),op=MPI.MAX,root=0)
        if (self.layout==Layout.BLOCK):
            if (axis==self.Dimension.Z):
                if (fixValue>=self.zStarts[self.rank] and fixValue<self.zStarts[self.rank+1]):
                    idx = (np.s_[:],) * axis + (fixValue-self.zStarts[self.rank],)
                    return MPI.COMM_WORLD.reduce(np.amax(self.f[idx]),op=MPI.MAX,root=0)
                else:
                    return MPI.COMM_WORLD.reduce(0,op=MPI.MAX,root=0)
            else:
                idx = (np.s_[:],) * axis + (fixValue,)
                return MPI.COMM_WORLD.reduce(np.amax(self.f[idx]),op=MPI.MAX,root=0)
        elif (self.layout==Layout.RADIAL):
            if (axis==self.Dimension.R):
                if (fixValue>=self.rStarts[self.rank] and fixValue<self.rStarts[self.rank+1]):
                    idx = (np.s_[:],) * axis + (fixValue-self.rStarts[self.rank],)
                    return MPI.COMM_WORLD.reduce(np.amax(self.f[idx]),op=MPI.MAX,root=0)
                else:
                    return MPI.COMM_WORLD.reduce(0,op=MPI.MAX,root=0)
            else:
                idx = (np.s_[:],) * axis + (fixValue,)
                return MPI.COMM_WORLD.reduce(np.amax(self.f[idx]),op=MPI.MAX,root=0)
