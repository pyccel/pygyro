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
        self.rank = comm.Get_rank()
        self.mpi_size = comm.Get_size()
        
        self.layout=layout
        
        if (self.layout==Layout.FIELD_ALIGNED):
            self.sizeRV=nProcR
            self.sizeVZ=nProcV
            if(nProcZ!=1):
                raise ValueError("The data should not be distributed in z for a field-aligned layout")
        elif (self.layout==Layout.V_PARALLEL):
            self.sizeRV=nProcR
            self.sizeVZ=nProcZ
            if(nProcV!=1):
                raise ValueError("The data should not be distributed in v for a v parallel layout")
        elif (self.layout==Layout.POLOIDAL):
            self.sizeRV=nProcV
            self.sizeVZ=nProcZ
            if(nProcR!=1):
                raise ValueError("The data should not be distributed in r for a poloidal layout")
        else:
            raise NotImplementedError("Layout not implemented")
        
        if (self.mpi_size>1):
            assert(self.sizeRV*self.sizeVZ==self.mpi_size)
            topology = comm.Create_cart([self.sizeRV,self.sizeVZ], periods=[False, False])
            self.commRV = topology.Sub([True, False])
            self.commVZ = topology.Sub([False, True])
        else:
            self.commRV = MPI.COMM_WORLD
            self.commVZ = MPI.COMM_WORLD
        
        self.rankRV=self.commRV.Get_rank()
        self.rankVZ=self.commVZ.Get_rank()
        
        if (self.rank==0):
            assert(self.rankRV==0)
            assert(self.rankVZ==0)
        
        self.ranksRV=np.arange(0,self.sizeRV)
        self.ranksVZ=np.arange(0,self.sizeVZ)
        
        self.Vals = [theta, r, z, v]
        self.nr=len(r)
        self.nq=len(theta)
        self.nz=len(z)
        self.nv=len(v)
        
        self.defineShape()
        
        # ordering chosen to increase step size to improve cache-coherency
        self.f = np.empty((self.nq,len(self.Vals[self.Dimension.R][self.rStart:self.rEnd]),
                len(self.Vals[self.Dimension.Z][self.zStart:self.zEnd]),
                len(self.Vals[self.Dimension.V][self.vStart:self.vEnd])),float,order='F')
    
    def defineShape(self):
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
    
    def getVParSlice(self, theta: int, r: int, z: int):
        assert(self.layout==Layout.V_PARALLEL)
        return self.f[theta,r,z,:]
    
    def setVParSlice(self, theta: int, r: int, z: int, f):
        assert(self.layout==Layout.V_PARALLEL)
        self.f[theta,r,z,:]=f
    
    def getFieldAlignedSlice(self, r: int, vPar: int):
        assert(self.layout==Layout.FIELD_ALIGNED)
        return self.f[:,r,:,vPar]
    
    def setFieldAlignedSlice(self, r: int, vPar: int,f):
        assert(self.layout==Layout.FIELD_ALIGNED)
        self.f[:,r,:,vPar]=f
    
    def getPoloidalSlice(self, z: int, vPar: int):
        assert(self.layout==Layout.POLOIDAL)
        return self.f[:,:,z,vPar]
    
    def setPoloidalSlice(self, z: int, vPar: int,f):
        assert(self.layout==Layout.POLOIDAL)
        self.f[:,:,z,vPar]=f
    
    def getSliceFromDict(self,d : dict):
        r = d.get(self.Dimension.R,None)
        q = d.get(self.Dimension.THETA,None)
        v = d.get(self.Dimension.V,None)
        z = d.get(self.Dimension.Z,None)
        return self.getSliceForFig(r,q,z,v)
    
    def getSliceForFig(self,r = None,theta = None,z = None,v = None):
        finalSize=1
        dims=[]
        dimIdx=[]
        # get slices for each dimension
        # if value is none then all values along that dimension should be returned
        # if not then only values at that index should be returned
        if (theta==None):
            finalSize=finalSize*self.nq
            thetaVal=slice(0,self.nq)
            dims.append(self.nq)
            dimIdx.append(self.Dimension.THETA)
        else:
            thetaVal=theta
        if (r==None):
            finalSize=finalSize*self.nr
            rVal=slice(0,self.rEnd-self.rStart)
            dims.append(self.nr)
            dimIdx.append(self.Dimension.R)
        else:
            rVal=None
            if (r>=self.rStart and r<self.rEnd):
                rVal=r-self.rStart
        if (z==None):
            finalSize=finalSize*self.nz
            zVal=slice(0,self.zEnd-self.zStart)
            dims.append(self.nz)
            dimIdx.append(self.Dimension.Z)
        else:
            zVal=None
            if (z>=self.zStart and z<self.zEnd):
                zVal=z-self.zStart
        if (v==None):
            finalSize=finalSize*self.nv
            vVal=slice(0,self.vEnd-self.vStart)
            dims.append(self.nv)
            dimIdx.append(self.Dimension.V)
        else:
            vVal=None
            if (v>=self.vStart and v<self.vEnd):
                vVal=v-self.vStart
        
        # set sendSize and data to be sent
        # if the data is not on this processor then at least one of the slices is equal to None
        # in this case send something of size 0
        if (None in [thetaVal,rVal,zVal,vVal]):
            sendSize=0
            toSend = np.ndarray(0,order='F')
        else:
            toSend = self.f[thetaVal,rVal,zVal,vVal].flatten()
            sendSize=toSend.size
        # the dimensions must be concatenated one at a time to properly shape the output.
        # collect the send sizes to properly formulate the gatherv
        sizes=self.commRV.gather(sendSize,root=0)
        mySlice = None
        if (self.rankRV==0):
            starts=np.zeros(len(sizes))
            starts[1:]=sizes[:self.sizeRV-1]
            starts=starts.cumsum().astype(int)
            mySlice = np.empty(np.sum(sizes), dtype=float)
            self.commRV.Gatherv(toSend,(mySlice, sizes, starts, MPI.DOUBLE), 0)
            
            if (self.layout==Layout.FIELD_ALIGNED):
                splitDims=[self.Dimension.R, self.Dimension.V]
            elif (self.layout==Layout.POLOIDAL):
                splitDims=[self.Dimension.V, self.Dimension.Z]
            elif (self.layout==Layout.V_PARALLEL):
                splitDims=[self.Dimension.R, self.Dimension.Z]
            
            if (splitDims[0]==dimIdx[0]):
                mySlice=np.split(mySlice,starts[1:])
                for i in range(0,self.sizeRV):
                    mySlice[i]=mySlice[i].reshape(sizes[i]//dims[1],dims[1])
                mySlice=np.concatenate(mySlice,axis=0)
            elif(splitDims[0]==dimIdx[1]):
                mySlice=np.split(mySlice,starts[1:])
                for i in range(0,self.sizeRV):
                    mySlice[i]=mySlice[i].reshape(dims[0],sizes[i]//dims[0])
                mySlice=np.concatenate(mySlice,axis=1)
            
            sizes=self.commVZ.gather(mySlice.size,root=0)
            if (self.rankVZ==0):
                starts=np.zeros(len(sizes))
                starts[1:]=sizes[:self.sizeVZ-1]
                starts=starts.cumsum().astype(int)
                toSend = np.empty(np.sum(sizes), dtype=float)
                self.commVZ.Gatherv(mySlice,(toSend, sizes, starts, MPI.DOUBLE), 0)
                
                if (splitDims[1]==dimIdx[0]):
                    mySlice=np.split(toSend,starts[1:])
                    for i in range(0,self.sizeVZ):
                        mySlice[i]=mySlice[i].reshape(sizes[i]//dims[1],dims[1])
                    return np.concatenate(mySlice,axis=0)
                elif(splitDims[1]==dimIdx[1]):
                    mySlice=np.split(toSend,starts[1:])
                    for i in range(0,self.sizeVZ):
                        mySlice[i]=mySlice[i].reshape(dims[0],sizes[i]//dims[0])
                    return np.concatenate(mySlice,axis=1)
                else:
                    mySlice=toSend.reshape(dims[0],dims[1])
                return mySlice
            else:
                self.commVZ.Gatherv(mySlice,mySlice, 0)
                return
        else:
            self.commRV.Gatherv(toSend,toSend, 0)
            return
    
    def getMin(self,axis = None,fixValue = None):
        if (axis==None and fixValue==None):
            return MPI.COMM_WORLD.reduce(np.amin(self.f),op=MPI.MIN,root=0)
        if (axis==self.Dimension.Z and fixValue>=self.zStart and fixValue<self.zStart):
            idx = (np.s_[:],) * axis + (fixValue-self.zStart,)
            return MPI.COMM_WORLD.reduce(np.amin(self.f[idx]),op=MPI.MIN,root=0)
        elif (axis==self.Dimension.R and fixValue>=self.rStart and fixValue<self.rStart):
            idx = (np.s_[:],) * axis + (fixValue-self.rStart,)
            return MPI.COMM_WORLD.reduce(np.amin(self.f[idx]),op=MPI.MIN,root=0)
        elif (axis==self.Dimension.V and fixValue>=self.vStart and fixValue<self.vStart):
            idx = (np.s_[:],) * axis + (fixValue-self.vStart,)
            return MPI.COMM_WORLD.reduce(np.amin(self.f[idx]),op=MPI.MIN,root=0)
        elif (axis==self.Dimension.THETA):
            idx = (np.s_[:],) * axis + (fixValue,)
            return MPI.COMM_WORLD.reduce(np.amin(self.f[idx]),op=MPI.MIN,root=0)
        else:
            return MPI.COMM_WORLD.reduce(0,op=MPI.MIN,root=0)
    
    def getMax(self,axis = None,fixValue = None):
        if (axis==None and fixValue==None):
            return MPI.COMM_WORLD.reduce(np.amax(self.f),op=MPI.MAX,root=0)
        if (axis==self.Dimension.Z and fixValue>=self.zStart and fixValue<self.zStart):
            idx = (np.s_[:],) * axis + (fixValue-self.zStart,)
            return MPI.COMM_WORLD.reduce(np.amax(self.f[idx]),op=MPI.MAX,root=0)
        elif (axis==self.Dimension.R and fixValue>=self.rStart and fixValue<self.rStart):
            idx = (np.s_[:],) * axis + (fixValue-self.rStart,)
            return MPI.COMM_WORLD.reduce(np.amax(self.f[idx]),op=MPI.MAX,root=0)
        elif (axis==self.Dimension.V and fixValue>=self.vStart and fixValue<self.vStart):
            idx = (np.s_[:],) * axis + (fixValue-self.vStart,)
            return MPI.COMM_WORLD.reduce(np.amax(self.f[idx]),op=MPI.MAX,root=0)
        elif (axis==self.Dimension.THETA):
            idx = (np.s_[:],) * axis + (fixValue,)
            return MPI.COMM_WORLD.reduce(np.amax(self.f[idx]),op=MPI.MAX,root=0)
        else:
            return MPI.COMM_WORLD.reduce(0,op=MPI.MAX,root=0)
