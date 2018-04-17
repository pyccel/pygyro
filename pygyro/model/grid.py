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
    
    def __init__(self,r,theta,z,v,layout: Layout,
                    *,nProcR : int = 1,nProcZ : int = 1,nProcV = 1):
        # get MPI values
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.mpi_size = comm.Get_size()
        
        # remember layout
        self.layout=layout
        
        # ensure that the combination of processors agrees with the layout
        # (i.e. there is only 1 processor along the non-distributed direction
        # save the grid shape
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
            # ensure that we are not distributing more than makes sense within MPI
            assert(self.sizeRV*self.sizeVZ==self.mpi_size)
            # create the communicators
            # reorder = false to help with the figures (then 0,0 == 0 a.k.a the return node)
            topology = comm.Create_cart([self.sizeRV,self.sizeVZ], periods=[False, False], reorder=False)
            self.commRV = topology.Sub([True, False])
            self.commVZ = topology.Sub([False, True])
        else:
            # if the code is run in serial then the values should be assigned
            # but all directions contain all processors
            self.commRV = MPI.COMM_WORLD
            self.commVZ = MPI.COMM_WORLD
        
        # get ranks for the different communicators
        self.rankRV=self.commRV.Get_rank()
        self.rankVZ=self.commVZ.Get_rank()
        
        # save coordinate information
        # saving in list allows simpler reordering of coordinates
        self.Vals = [theta, r, z, v]
        self.nr=len(r)
        self.nq=len(theta)
        self.nz=len(z)
        self.nv=len(v)
        
        #get start and end points for each processor
        self.defineShape()
        
        # ordering chosen to increase step size to improve cache-coherency
        self.f = np.empty((self.nq,len(self.Vals[self.Dimension.R][self.rStart:self.rEnd]),
                len(self.Vals[self.Dimension.Z][self.zStart:self.zEnd]),
                len(self.Vals[self.Dimension.V][self.vStart:self.vEnd])),float,order='F')
    
    def defineShape(self):
        # get common variables
        ranksRV=np.arange(0,self.sizeRV)
        ranksVZ=np.arange(0,self.sizeVZ)
        
        # variables depend on setup
        if (self.layout==Layout.FIELD_ALIGNED):
            # get overflows to better distribute data that is not divisible by
            # the number of processors
            nrOverflow=self.nr%self.sizeRV
            nvOverflow=self.nv%self.sizeVZ
            
            # get start indices for all processors
            rStarts=self.nr//self.sizeRV*ranksRV + np.minimum(ranksRV,nrOverflow)
            vStarts=self.nv//self.sizeVZ*ranksVZ + np.minimum(ranksVZ,nvOverflow)
            # append end index
            rStarts=np.append(rStarts,self.nr)
            vStarts=np.append(vStarts,self.nv)
            
            # save start indices from list using cartesian ranks
            self.rStart=rStarts[self.rankRV]
            self.vStart=vStarts[self.rankVZ]
            # z is not distributed so the start index is 0
            self.zStart=0
            
            # save end points from list using cartesian ranks
            self.rEnd=rStarts[self.rankRV+1]
            self.vEnd=vStarts[self.rankVZ+1]
            # z is not distributed so the end index is its length
            self.zEnd=self.nz
        elif (self.layout==Layout.V_PARALLEL):
            # get overflows to better distribute data that is not divisible by
            # the number of processors
            nrOverflow=self.nr%self.sizeRV
            nzOverflow=self.nz%self.sizeVZ
            
            # get start indices for all processors
            rStarts=self.nr//self.sizeRV*ranksRV + np.minimum(ranksRV,nrOverflow)
            zStarts=self.nz//self.sizeVZ*ranksVZ + np.minimum(ranksVZ,nzOverflow)
            # append end index
            rStarts=np.append(rStarts,self.nr)
            zStarts=np.append(zStarts,self.nz)
            
            # save start indices from list using cartesian ranks
            self.rStart=rStarts[self.rankRV]
            self.zStart=zStarts[self.rankVZ]
            # v is not distributed so the start index is 0
            self.vStart=0
            
            # save end points from list using cartesian ranks
            self.rEnd=rStarts[self.rankRV+1]
            self.zEnd=zStarts[self.rankVZ+1]
            # v is not distributed so the end index is its length
            self.vEnd=self.nv
        elif (self.layout==Layout.POLOIDAL):
            # get overflows to better distribute data that is not divisible by
            # the number of processors
            nvOverflow=self.nv%self.sizeRV
            nzOverflow=self.nz%self.sizeVZ
            
            # get start indices for all processors
            vStarts=self.nv//self.sizeRV*ranksRV + np.minimum(ranksRV,nvOverflow)
            zStarts=self.nz//self.sizeVZ*ranksVZ + np.minimum(ranksVZ,nzOverflow)
            # append end index
            vStarts=np.append(vStarts,self.nv)
            zStarts=np.append(zStarts,self.nz)
            
            # save start indices from list using cartesian ranks
            self.vStart=vStarts[self.rankRV]
            self.zStart=zStarts[self.rankVZ]
            # r is not distributed so the start index is 0
            self.rStart=0
            
            # save end points from list using cartesian ranks
            self.vEnd=vStarts[self.rankRV+1]
            self.zEnd=zStarts[self.rankVZ+1]
            # r is not distributed so the end index is its length
            self.rEnd=self.nr
    
    @property
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
    
    def setLayout(self,new_layout: Layout):
        # if layout has not changed then do nothing
        if (new_layout==self.layout):
            return
        if (self.layout==Layout.FIELD_ALIGNED):
            if (new_layout==Layout.V_PARALLEL):
                # if Field_aligned -> v_parallel
                # recalculate start indices
                nzOverflow=self.nz%self.sizeVZ
                ranksVZ=np.arange(0,self.sizeVZ)
                zStarts=self.nz//self.sizeVZ*ranksVZ + np.minimum(ranksVZ,nzOverflow)
                
                # redistribute data
                self.f = np.concatenate(
                            self.commVZ.alltoall(
                                # break data on this processor into chunks using start indices
                                np.split(self.f,zStarts[1:],axis=self.Dimension.Z)
                            ) # use all to all to pass chunks to correct processors
                        ,axis=self.Dimension.V) # use concatenate to join the data back together in the right shape
                
                # save new layout
                self.layout = Layout.V_PARALLEL
                # add end index
                zStarts=np.append(zStarts,self.nz)
                # save new start and end indices
                self.zStart=zStarts[self.rankVZ]
                self.vStart=0
                self.zEnd=zStarts[self.rankVZ+1]
                self.vEnd=self.nv
            elif (new_layout==Layout.POLOIDAL):
                # if Field_aligned -> poloidal
                # 2 steps are required
                self.setLayout(Layout.V_PARALLEL)
                self.setLayout(Layout.POLOIDAL)
                raise RuntimeWarning("Changing from Field Aligned layout to Poloidal layout requires two steps")
        elif (self.layout==Layout.POLOIDAL):
            if (new_layout==Layout.V_PARALLEL):
                # if poloidal -> v_parallel
                # recalculate start indices
                nrOverflow=self.nr%self.sizeRV
                ranksRV=np.arange(0,self.sizeRV)
                rStarts=self.nr//self.sizeRV*ranksRV + np.minimum(ranksRV,nrOverflow)
                
                # redistribute data
                self.f = np.concatenate(
                            self.commRV.alltoall(
                                # break data on this processor into chunks using start indices
                                np.split(self.f,rStarts[1:],axis=self.Dimension.R)
                            ) # use all to all to pass chunks to correct processors
                        ,axis=self.Dimension.V) # use concatenate to join the data back together in the right shape
                
                # save new layout
                self.layout = Layout.V_PARALLEL
                rStarts=np.append(rStarts,self.nr)
                # save new start and end indices
                self.rStart=rStarts[self.rankRV]
                self.vStart=0
                self.rEnd=rStarts[self.rankRV+1]
                self.vEnd=len(self.Vals[self.Dimension.V])
            elif (new_layout==Layout.FIELD_ALIGNED):
                # if poloidal -> Field_aligned
                # 2 steps are required
                self.setLayout(Layout.V_PARALLEL)
                self.setLayout(Layout.FIELD_ALIGNED)
                raise RuntimeWarning("Changing from Poloidal layout to Field Aligned layout requires two steps")
        elif (self.layout==Layout.V_PARALLEL):
            if (new_layout==Layout.FIELD_ALIGNED):
                # if v_parallel -> Field_aligned
                # recalculate start indices
                nvOverflow=self.nv%self.sizeVZ
                ranksVZ=np.arange(0,self.sizeVZ)
                vStarts=self.nv//self.sizeVZ*ranksVZ + np.minimum(ranksVZ,nvOverflow)
                
                # redistribute data
                self.f = np.concatenate(
                            self.commVZ.alltoall(
                                # break data on this processor into chunks using start indices
                                np.split(self.f,vStarts[1:],axis=self.Dimension.V)
                            ) # use all to all to pass chunks to correct processors
                        ,axis=self.Dimension.Z) # use concatenate to join the data back together in the right shape
                
                # save new layout
                self.layout = Layout.FIELD_ALIGNED
                # add end index
                vStarts=np.append(vStarts,self.nv)
                # save new start and end indices
                self.zStart=0
                self.vStart=vStarts[self.rankVZ]
                self.zEnd=len(self.Vals[self.Dimension.Z])
                self.vEnd=vStarts[self.rankVZ+1]
            elif (new_layout==Layout.POLOIDAL):
                # if v_parallel -> poloidal
                # recalculate start indices
                nvOverflow=self.nv%self.sizeRV
                ranksRV=np.arange(0,self.sizeRV)
                vStarts=self.nv//self.sizeRV*ranksRV + np.minimum(ranksRV,nvOverflow)
                
                # redistribute data
                self.f = np.concatenate(
                            self.commRV.alltoall(
                                # break data on this processor into chunks using start indices
                                np.split(self.f,vStarts[1:],axis=self.Dimension.V)
                            ) # use all to all to pass chunks to correct processors
                        ,axis=self.Dimension.R) # use concatenate to join the data back together in the right shape
                
                # save new layout
                self.layout = Layout.POLOIDAL
                # add end index
                vStarts=np.append(vStarts,self.nv)
                # save new start and end indices
                self.vStart=vStarts[self.rankRV]
                self.rStart=0
                self.vEnd=vStarts[self.rankRV+1]
                self.rEnd=len(self.Vals[self.Dimension.R])
    
    ####################################################################
    ####                   Functions for figures                    ####
    ####################################################################
    
    
    def getSliceFromDict(self,d : dict):
        """
        Utility function to access getSliceForFig without using 6 if statements
        The function takes a dictionary which plots the fixed dimensions to the
        index at which they are fixed
        
        >>> getSliceFromDict({self.Dimension.Z: 5, self.Dimension.V : 2})
        """
        r = d.get(self.Dimension.R,None)
        q = d.get(self.Dimension.THETA,None)
        v = d.get(self.Dimension.V,None)
        z = d.get(self.Dimension.Z,None)
        return self.getSliceForFig(r,q,z,v)
    
    def getSliceForFig(self,r = None,theta = None,z = None,v = None):
        """
        Class to retrieve a 2D slice. Any values set to None will vary.
        Any dimensions not set to None will be fixed at the global index provided
        """
        
        # helper variables to correctly reshape the slice after gathering
        dimSize=[]
        dimIdx=[]
        
        # Get slices for each dimension (thetaVal, rVal, zVal, vVal)
        # If value is None then all values along that dimension should be returned
        # this means that the size and dimension index will be stored
        # If value is not None then only values at that index should be returned
        # if that index cannot be found on the current processor then None will
        # be stored
        if (theta==None):
            thetaVal=slice(0,self.nq)
            dimSize.append(self.nq)
            dimIdx.append(self.Dimension.THETA)
        else:
            thetaVal=theta
        if (r==None):
            rVal=slice(0,self.rEnd-self.rStart)
            dimSize.append(self.nr)
            dimIdx.append(self.Dimension.R)
        else:
            rVal=None
            if (r>=self.rStart and r<self.rEnd):
                rVal=r-self.rStart
        if (z==None):
            zVal=slice(0,self.zEnd-self.zStart)
            dimSize.append(self.nz)
            dimIdx.append(self.Dimension.Z)
        else:
            zVal=None
            if (z>=self.zStart and z<self.zEnd):
                zVal=z-self.zStart
        if (v==None):
            vVal=slice(0,self.vEnd-self.vStart)
            dimSize.append(self.nv)
            dimIdx.append(self.Dimension.V)
        else:
            vVal=None
            if (v>=self.vStart and v<self.vEnd):
                vVal=v-self.vStart
        
        # if the data is not on this processor then at least one of the slices is equal to None
        # in this case send something of size 0
        if (None in [thetaVal,rVal,zVal,vVal]):
            sendSize=0
            toSend = np.ndarray(0,order='F')
        else:
            # set sendSize and data to be sent
            toSend = self.f[thetaVal,rVal,zVal,vVal].flatten()
            sendSize=toSend.size
        
        # collect the send sizes to properly formulate the gatherv
        sizes=self.commRV.gather(sendSize,root=0)
        
        # the dimensions must be concatenated one at a time to properly shape the output.
        if (self.rankRV==0):
            # use sizes to get start points
            starts=np.zeros(len(sizes),int)
            starts[1:]=np.cumsum(sizes[:self.sizeRV-1])
            
            # save memory for gatherv to fill
            mySlice = np.empty(np.sum(sizes), dtype=float)
            
            # Gather information from all ranks to rank 0 in direction R or V (dependant upon layout)
            self.commRV.Gatherv(toSend,(mySlice, sizes, starts, MPI.DOUBLE), 0)
            
            # get dimensions along which split occurs
            if (self.layout==Layout.FIELD_ALIGNED):
                splitDims=[self.Dimension.R, self.Dimension.V]
            elif (self.layout==Layout.POLOIDAL):
                splitDims=[self.Dimension.V, self.Dimension.Z]
            elif (self.layout==Layout.V_PARALLEL):
                splitDims=[self.Dimension.R, self.Dimension.Z]
            
            # if the first dimension of the slice is distributed
            if (splitDims[0]==dimIdx[0]):
                # break received data into constituent parts
                mySlice=np.split(mySlice,starts[1:])
                for i in range(0,self.sizeRV):
                    # return them to original shape
                    mySlice[i]=mySlice[i].reshape(sizes[i]//dimSize[1],dimSize[1])
                # stitch back together along broken axis
                mySlice=np.concatenate(mySlice,axis=0)
            # if the second dimension of the slice is distributed
            elif(splitDims[0]==dimIdx[1]):
                # break received data into constituent parts
                mySlice=np.split(mySlice,starts[1:])
                for i in range(0,self.sizeRV):
                    # return them to original shape
                    mySlice[i]=mySlice[i].reshape(dimSize[0],sizes[i]//dimSize[0])
                # stitch back together along broken axis
                mySlice=np.concatenate(mySlice,axis=1)
            # else the data will be sent flattened anyway
            
            sizes=self.commVZ.gather(mySlice.size,root=0)
            if (self.rankVZ==0):
                # use sizes to get new start points
                starts=np.zeros(len(sizes),int)
                starts[1:]=np.cumsum(sizes[:self.sizeVZ-1])
                
                # ensure toSend has correct size in memory for gatherv to fill
                toSend = np.empty(np.sum(sizes), dtype=float)
                
                # collect data sent up first column on first cell
                self.commVZ.Gatherv(mySlice,(toSend, sizes, starts, MPI.DOUBLE), 0)
                
                # if the first dimension of the slice is distributed
                if (splitDims[1]==dimIdx[0]):
                    # break received data into constituent parts
                    mySlice=np.split(toSend,starts[1:])
                    for i in range(0,self.sizeVZ):
                        # return them to original shape
                        mySlice[i]=mySlice[i].reshape(sizes[i]//dimSize[1],dimSize[1])
                    # stitch back together along broken axis
                    return np.concatenate(mySlice,axis=0)
                # if the second dimension of the slice is distributed
                elif(splitDims[1]==dimIdx[1]):
                    # break received data into constituent parts
                    mySlice=np.split(toSend,starts[1:])
                    for i in range(0,self.sizeVZ):
                        # return them to original shape
                        mySlice[i]=mySlice[i].reshape(dimSize[0],sizes[i]//dimSize[0])
                    # stitch back together along broken axis
                    return np.concatenate(mySlice,axis=1)
                # if neither dimension is distributed
                else:
                    # ensure that the data has the right return shape
                    mySlice=toSend.reshape(dimSize[0],dimSize[1])
                return mySlice
            else:
                # send data collected to first column up to first cell
                self.commVZ.Gatherv(mySlice,mySlice, 0)
        else:
            # Gather information from all ranks
            self.commRV.Gatherv(toSend,toSend, 0)
    
    def getMin(self,axis = None,fixValue = None):
        
        # if we want the total of all points on the grid
        if (axis==None and fixValue==None):
            # return the min of the min found on each processor
            return MPI.COMM_WORLD.reduce(np.amin(self.f),op=MPI.MIN,root=0)
        
        # if we want the total of all points on a 3D slice where the value of z is fixed
        # ensure that the required index is covered by this processor
        if (axis==self.Dimension.Z and fixValue>=self.zStart and fixValue<self.zEnd):
            # get the indices of the required slice
            idx = (np.s_[:],) * axis + (fixValue-self.zStart,)
            # return the min of the min found on each processor's slice
            return MPI.COMM_WORLD.reduce(np.amin(self.f[idx]),op=MPI.MIN,root=0)
        
        # if we want the total of all points on a 3D slice where the value of r is fixed
        # ensure that the required index is covered by this processor
        elif (axis==self.Dimension.R and fixValue>=self.rStart and fixValue<self.rEnd):
            # get the indices of the required slice
            idx = (np.s_[:],) * axis + (fixValue-self.rStart,)
            # return the min of the min found on each processor's slice
            return MPI.COMM_WORLD.reduce(np.amin(self.f[idx]),op=MPI.MIN,root=0)
        
        # if we want the total of all points on a 3D slice where the value of v is fixed
        # ensure that the required index is covered by this processor
        elif (axis==self.Dimension.V and fixValue>=self.vStart and fixValue<self.vEnd):
            # get the indices of the required slice
            idx = (np.s_[:],) * axis + (fixValue-self.vStart,)
            # return the min of the min found on each processor's slice
            return MPI.COMM_WORLD.reduce(np.amin(self.f[idx]),op=MPI.MIN,root=0)
        
        # if we want the total of all points on a 3D slice where the value of theta is fixed
        # ensure that the required index is covered by this processor
        elif (axis==self.Dimension.THETA):
            # get the indices of the required slice
            idx = (np.s_[:],) * axis + (fixValue,)
            # return the min of the min found on each processor's slice
            return MPI.COMM_WORLD.reduce(np.amin(self.f[idx]),op=MPI.MIN,root=0)
        
        # if the data is not on this processor then send the largest possible value of f
        # this way min will always choose an alternative
        else:
            return MPI.COMM_WORLD.reduce(1,op=MPI.MIN,root=0)
    
    
    def getMax(self,axis = None,fixValue = None):
        
        # if we want the total of all points on the grid
        if (axis==None and fixValue==None):
            # return the max of the max found on each processor
            return MPI.COMM_WORLD.reduce(np.amax(self.f),op=MPI.MAX,root=0)
        
        # if we want the total of all points on a 3D slice where the value of z is fixed
        # ensure that the required index is covered by this processor
        if (axis==self.Dimension.Z and fixValue>=self.zStart and fixValue<self.zEnd):
            # get the indices of the required slice
            idx = (np.s_[:],) * axis + (fixValue-self.zStart,)
            # return the max of the max found on each processor's slice
            return MPI.COMM_WORLD.reduce(np.amax(self.f[idx]),op=MPI.MAX,root=0)
        
        # if we want the total of all points on a 3D slice where the value of r is fixed
        # ensure that the required index is covered by this processor
        elif (axis==self.Dimension.R and fixValue>=self.rStart and fixValue<self.rEnd):
            # get the indices of the required slice
            idx = (np.s_[:],) * axis + (fixValue-self.rStart,)
            # return the max of the max found on each processor's slice
            return MPI.COMM_WORLD.reduce(np.amax(self.f[idx]),op=MPI.MAX,root=0)
        
        # if we want the total of all points on a 3D slice where the value of v is fixed
        # ensure that the required index is covered by this processor
        elif (axis==self.Dimension.V and fixValue>=self.vStart and fixValue<self.vEnd):
            # get the indices of the required slice
            idx = (np.s_[:],) * axis + (fixValue-self.vStart,)
            # return the max of the max found on each processor's slice
            return MPI.COMM_WORLD.reduce(np.amax(self.f[idx]),op=MPI.MAX,root=0)
        
        # if we want the total of all points on a 3D slice where the value of theta is fixed
        # ensure that the required index is covered by this processor
        elif (axis==self.Dimension.THETA):
            # get the indices of the required slice
            idx = (np.s_[:],) * axis + (fixValue,)
            # return the max of the max found on each processor's slice
            return MPI.COMM_WORLD.reduce(np.amax(self.f[idx]),op=MPI.MAX,root=0)
        
        # if the data is not on this processor then send the smallest possible value of f
        # this way max will always choose an alternative
        else:
            return MPI.COMM_WORLD.reduce(0,op=MPI.MAX,root=0)
