from mpi4py import MPI
import numpy as np
from enum import Enum, IntEnum
from math import pi

from ..                 import splines as spl

class Layout(Enum):
    FIELD_ALIGNED = 1
    V_PARALLEL = 2
    POLOIDAL = 3

class Grid(object):
    class Dimension(IntEnum):
        ETA1 = 1
        ETA2 = 0
        ETA3 = 2
        ETA4 = 3
    
    def __init__(self,eta1,eta2,eta3,eta4,layout: Layout,
                    *,nProcEta1 : int = 1,nProcEta3 : int = 1,nProcEta4 = 1):
        # get MPI values
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.mpi_size = comm.Get_size()
        
        # remember layout
        self.layout=layout
        
        # ensure that the combination of processes agrees with the layout
        # (i.e. there is only 1 process along the non-distributed direction
        # save the grid shape
        if (self.layout==Layout.FIELD_ALIGNED):
            self.sizeEta14=nProcEta1
            self.sizeEta34=nProcEta4
            if(nProcEta3!=1):
                raise ValueError("The data should not be distributed in z for a field-aligned layout")
        elif (self.layout==Layout.V_PARALLEL):
            self.sizeEta14=nProcEta1
            self.sizeEta34=nProcEta3
            if(nProcEta4!=1):
                raise ValueError("The data should not be distributed in v for a v parallel layout")
        elif (self.layout==Layout.POLOIDAL):
            self.sizeEta14=nProcEta4
            self.sizeEta34=nProcEta3
            if(nProcEta1!=1):
                raise ValueError("The data should not be distributed in r for a poloidal layout")
        else:
            raise NotImplementedError("Layout not implemented")
        
        if (self.mpi_size>1):
            # ensure that we are not distributing more than makes sense within MPI
            assert(self.sizeEta14*self.sizeEta34==self.mpi_size)
            # create the communicators
            # reorder = false to help with the figures (then 0,0 == 0 a.k.a the return node)
            topology = comm.Create_cart([self.sizeEta14,self.sizeEta34], periods=[False, False], reorder=False)
            self.commEta14 = topology.Sub([True, False])
            self.commEta34 = topology.Sub([False, True])
        else:
            # if the code is run in serial then the values should be assigned
            # but all directions contain all processes
            self.commEta14 = MPI.COMM_WORLD
            self.commEta34 = MPI.COMM_WORLD
        
        # get ranks for the different communicators
        self.rankEta14=self.commEta14.Get_rank()
        self.rankEta34=self.commEta34.Get_rank()
        
        # save coordinate information
        # saving in list allows simpler reordering of coordinates
        self.Vals = [eta2, eta1, eta3, eta4]
        self.nEta1=len(eta1)
        self.nEta2=len(eta2)
        self.nEta3=len(eta3)
        self.nEta4=len(eta4)
        
        #get start and end points for each process
        self.defineShape()
        
        # ordering chosen to increase step size to improve cache-coherency
        self.f = np.empty((self.nEta2,len(self.Vals[self.Dimension.ETA1][self.eta1_start:self.eta1_end]),
                len(self.Vals[self.Dimension.ETA3][self.eta3_start:self.eta3_end]),
                len(self.Vals[self.Dimension.ETA4][self.eta4_start:self.eta4_end])),float,order='F')
    
    def defineShape(self):
        # get common variables
        ranksEta14=np.arange(0,self.sizeEta14)
        ranksEta34=np.arange(0,self.sizeEta34)
        
        # variables depend on setup
        if (self.layout==Layout.FIELD_ALIGNED):
            # get overflows to better distribute data that is not divisible by
            # the number of processes
            nEta1_Overflow=self.nEta1%self.sizeEta14
            nEta4_Overflow=self.nEta4%self.sizeEta34
            
            # get start indices for all processes
            eta1_Starts=self.nEta1//self.sizeEta14*ranksEta14 + np.minimum(ranksEta14,nEta1_Overflow)
            eta4_Starts=self.nEta4//self.sizeEta34*ranksEta34 + np.minimum(ranksEta34,nEta4_Overflow)
            # append end index
            eta1_Starts=np.append(eta1_Starts,self.nEta1)
            eta4_Starts=np.append(eta4_Starts,self.nEta4)
            
            # save start indices from list using cartesian ranks
            self.eta1_start=eta1_Starts[self.rankEta14]
            self.eta4_start=eta4_Starts[self.rankEta34]
            # eta3 is not distributed so the start index is 0
            self.eta3_start=0
            
            # save end points from list using cartesian ranks
            self.eta1_end=eta1_Starts[self.rankEta14+1]
            self.eta4_end=eta4_Starts[self.rankEta34+1]
            # eta3 is not distributed so the end index is its length
            self.eta3_end=self.nEta3
        elif (self.layout==Layout.V_PARALLEL):
            # get overflows to better distribute data that is not divisible by
            # the number of processes
            nEta1_Overflow=self.nEta1%self.sizeEta14
            nEta3_Overflow=self.nEta3%self.sizeEta34
            
            # get start indices for all processes
            eta1_Starts=self.nEta1//self.sizeEta14*ranksEta14 + np.minimum(ranksEta14,nEta1_Overflow)
            self.eta3_starts=self.nEta3//self.sizeEta34*ranksEta34 + np.minimum(ranksEta34,nEta3_Overflow)
            # append end index
            eta1_Starts=np.append(eta1_Starts,self.nEta1)
            self.eta3_starts=np.append(self.eta3_starts,self.nEta3)
            
            # save start indices from list using cartesian ranks
            self.eta1_start=eta1_Starts[self.rankEta14]
            self.eta3_start=self.eta3_starts[self.rankEta34]
            # eta4 is not distributed so the start index is 0
            self.eta4_start=0
            
            # save end points from list using cartesian ranks
            self.eta1_end=eta1_Starts[self.rankEta14+1]
            self.eta3_end=self.eta3_starts[self.rankEta34+1]
            # eta4 is not distributed so the end index is its length
            self.eta4_end=self.nEta4
        elif (self.layout==Layout.POLOIDAL):
            # get overflows to better distribute data that is not divisible by
            # the number of processes
            nEta4_Overflow=self.nEta4%self.sizeEta14
            nEta3_Overflow=self.nEta3%self.sizeEta34
            
            # get start indices for all processes
            eta4_Starts=self.nEta4//self.sizeEta14*ranksEta14 + np.minimum(ranksEta14,nEta4_Overflow)
            self.eta3_starts=self.nEta3//self.sizeEta34*ranksEta34 + np.minimum(ranksEta34,nEta3_Overflow)
            # append end index
            eta4_Starts=np.append(eta4_Starts,self.nEta4)
            self.eta3_starts=np.append(self.eta3_starts,self.nEta3)
            
            # save start indices from list using cartesian ranks
            self.eta4_start=eta4_Starts[self.rankEta14]
            self.eta3_start=self.eta3_starts[self.rankEta34]
            # eta1 is not distributed so the start index is 0
            self.eta1_start=0
            
            # save end points from list using cartesian ranks
            self.eta4_end=eta4_Starts[self.rankEta14+1]
            self.eta3_end=self.eta3_starts[self.rankEta34+1]
            # eta1 is not distributed so the end index is its length
            self.eta1_end=self.nEta1
    
    @property
    def size(self):
        return self.f.size
    
    def getEta1Coords(self):
        return enumerate(self.Vals[self.Dimension.ETA1][self.eta1_start:self.eta1_end])
    
    @property
    def eta1_Vals(self):
        return self.Vals[self.Dimension.ETA1][self.eta1_start:self.eta1_end]
    
    def getEta2Coords(self):
        return enumerate(self.Vals[self.Dimension.ETA2])
    
    @property
    def eta2_Vals(self):
        return self.Vals[self.Dimension.ETA2]
    
    def getEta3Coords(self):
        return enumerate(self.Vals[self.Dimension.ETA3][self.eta3_start:self.eta3_end])
    
    @property
    def eta3_Vals(self):
        return self.Vals[self.Dimension.ETA3][self.eta3_start:self.eta3_end]
    
    def getEta4Coords(self):
        return enumerate(self.Vals[self.Dimension.ETA4][self.eta4_start:self.eta4_end])
    
    @property
    def eta4_Vals(self):
        return self.Vals[self.Dimension.ETA4][self.eta4_start:self.eta4_end]
    
    def getEta4_Slice(self, eta2: int, eta1: int, eta3: int):
        assert(self.layout==Layout.V_PARALLEL)
        return self.f[eta2,eta1,eta3,:]
    
    def setEta4_Slice(self, eta2: int, eta1: int, eta3: int, f):
        assert(self.layout==Layout.V_PARALLEL)
        self.f[eta2,eta1,eta3,:]=f
    
    def getFieldAlignedSlice(self, eta1: int, eta4: int):
        assert(self.layout==Layout.FIELD_ALIGNED)
        return self.f[:,eta1,:,eta4]
    
    def setFieldAlignedSlice(self, eta1: int, eta4: int,f):
        assert(self.layout==Layout.FIELD_ALIGNED)
        self.f[:,eta1,:,eta4]=f
    
    def getPoloidalSlice(self, eta3: int, eta4: int):
        assert(self.layout==Layout.POLOIDAL)
        return self.f[:,:,eta3,eta4]
    
    def setPoloidalSlice(self, eta3: int, eta4: int,f):
        assert(self.layout==Layout.POLOIDAL)
        self.f[:,:,eta3,eta4]=f
    
    def setLayout(self,new_layout: Layout):
        # if layout has not changed then do nothing
        if (new_layout==self.layout):
            return
        if (self.layout==Layout.FIELD_ALIGNED):
            if (new_layout==Layout.V_PARALLEL):
                # if Field_aligned -> v_parallel
                # recalculate start indices
                nEta3_Overflow=self.nEta3%self.sizeEta34
                ranksEta34=np.arange(0,self.sizeEta34)
                self.eta3_starts=self.nEta3//self.sizeEta34*ranksEta34 + np.minimum(ranksEta34,nEta3_Overflow)
                
                # redistribute data
                self.f = np.concatenate(
                            self.commEta34.alltoall(
                                # break data on this process into chunks using start indices
                                np.split(self.f,self.eta3_starts[1:],axis=self.Dimension.ETA3)
                            ) # use all to all to pass chunks to correct processes
                        ,axis=self.Dimension.ETA4) # use concatenate to join the data back together in the right shape
                
                # save new layout
                self.layout = Layout.V_PARALLEL
                # add end index
                self.eta3_starts=np.append(self.eta3_starts,self.nEta3)
                # save new start and end indices
                self.eta3_start=self.eta3_starts[self.rankEta34]
                self.eta4_start=0
                self.eta3_end=self.eta3_starts[self.rankEta34+1]
                self.eta4_end=self.nEta4
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
                nEta1_Overflow=self.nEta1%self.sizeEta14
                ranksEta14=np.arange(0,self.sizeEta14)
                eta1_Starts=self.nEta1//self.sizeEta14*ranksEta14 + np.minimum(ranksEta14,nEta1_Overflow)
                
                # redistribute data
                self.f = np.concatenate(
                            self.commEta14.alltoall(
                                # break data on this process into chunks using start indices
                                np.split(self.f,eta1_Starts[1:],axis=self.Dimension.ETA1)
                            ) # use all to all to pass chunks to correct processes
                        ,axis=self.Dimension.ETA4) # use concatenate to join the data back together in the right shape
                
                # save new layout
                self.layout = Layout.V_PARALLEL
                eta1_Starts=np.append(eta1_Starts,self.nEta1)
                # save new start and end indices
                self.eta1_start=eta1_Starts[self.rankEta14]
                self.eta4_start=0
                self.eta1_end=eta1_Starts[self.rankEta14+1]
                self.eta4_end=len(self.Vals[self.Dimension.ETA4])
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
                nEta4_Overflow=self.nEta4%self.sizeEta34
                ranksEta34=np.arange(0,self.sizeEta34)
                eta4_Starts=self.nEta4//self.sizeEta34*ranksEta34 + np.minimum(ranksEta34,nEta4_Overflow)
                
                # redistribute data
                self.f = np.concatenate(
                            self.commEta34.alltoall(
                                # break data on this process into chunks using start indices
                                np.split(self.f,eta4_Starts[1:],axis=self.Dimension.ETA4)
                            ) # use all to all to pass chunks to correct processes
                        ,axis=self.Dimension.ETA3) # use concatenate to join the data back together in the right shape
                
                # save new layout
                self.layout = Layout.FIELD_ALIGNED
                # add end index
                eta4_Starts=np.append(eta4_Starts,self.nEta4)
                # save new start and end indices
                self.eta3_start=0
                self.eta4_start=eta4_Starts[self.rankEta34]
                self.eta3_end=len(self.Vals[self.Dimension.ETA3])
                self.eta4_end=eta4_Starts[self.rankEta34+1]
            elif (new_layout==Layout.POLOIDAL):
                # if v_parallel -> poloidal
                # recalculate start indices
                nEta4_Overflow=self.nEta4%self.sizeEta14
                ranksEta14=np.arange(0,self.sizeEta14)
                eta4_Starts=self.nEta4//self.sizeEta14*ranksEta14 + np.minimum(ranksEta14,nEta4_Overflow)
                
                # redistribute data
                self.f = np.concatenate(
                            self.commEta14.alltoall(
                                # break data on this process into chunks using start indices
                                np.split(self.f,eta4_Starts[1:],axis=self.Dimension.ETA4)
                            ) # use all to all to pass chunks to correct processes
                        ,axis=self.Dimension.ETA1) # use concatenate to join the data back together in the right shape
                
                # save new layout
                self.layout = Layout.POLOIDAL
                # add end index
                eta4_Starts=np.append(eta4_Starts,self.nEta4)
                # save new start and end indices
                self.eta4_start=eta4_Starts[self.rankEta14]
                self.eta1_start=0
                self.eta4_end=eta4_Starts[self.rankEta14+1]
                self.eta1_end=len(self.Vals[self.Dimension.ETA1])
    
    ####################################################################
    ####                   Functions for figures                    ####
    ####################################################################
    
    
    def getSliceFromDict(self,d : dict):
        """
        Utility function to access getSliceForFig without using 6 if statements
        The function takes a dictionary which plots the fixed dimensions to the
        index at which they are fixed
        
        >>> getSliceFromDict({self.Dimension.ETA3: 5, self.Dimension.ETA4 : 2})
        """
        eta1 = d.get(self.Dimension.ETA1,None)
        eta2 = d.get(self.Dimension.ETA2,None)
        eta3 = d.get(self.Dimension.ETA3,None)
        eta4 = d.get(self.Dimension.ETA4,None)
        return self.getSliceForFig(eta1,eta2,eta3,eta4)
    
    def getSliceForFig(self,eta1 = None,eta2 = None,eta3 = None,eta4 = None):
        """
        Class to retrieve a 2D slice. Any values set to None will vary.
        Any dimensions not set to None will be fixed at the global index provided
        """
        
        # helper variables to correctly reshape the slice after gathering
        dimSize=[]
        dimIdx=[]
        
        # Get slices for each dimension (eta2_slice, eta1_slice, eta3_slice, eta4_slice)
        # If value is None then all values along that dimension should be returned
        # this means that the size and dimension index will be stored
        # If value is not None then only values at that index should be returned
        # if that index cannot be found on the current process then None will
        # be stored
        if (eta2==None):
            eta2_slice=slice(0,self.nEta2)
            dimSize.append(self.nEta2)
            dimIdx.append(self.Dimension.ETA2)
        else:
            eta2_slice=theta
        if (eta1==None):
            eta1_slice=slice(0,self.eta1_end-self.eta1_start)
            dimSize.append(self.nEta1)
            dimIdx.append(self.Dimension.ETA1)
        else:
            eta1_slice=None
            if (eta1>=self.eta1_start and eta1<self.eta1_end):
                eta1_slice=r-self.eta1_start
        if (eta3==None):
            eta3_slice=slice(0,self.eta3_end-self.eta3_start)
            dimSize.append(self.nEta3)
            dimIdx.append(self.Dimension.ETA3)
        else:
            eta3_slice=None
            if (eta3>=self.eta3_start and eta3<self.eta3_end):
                eta3_slice=z-self.eta3_start
        if (eta4==None):
            eta4_slice=slice(0,self.eta4_end-self.eta4_start)
            dimSize.append(self.nEta4)
            dimIdx.append(self.Dimension.ETA4)
        else:
            eta4_slice=None
            if (eta4>=self.eta4_start and eta4<self.eta4_end):
                eta4_slice=v-self.eta4_start
        
        # if the data is not on this process then at least one of the slices is equal to None
        # in this case send something of size 0
        if (None in [eta2_slice,eta1_slice,eta3_slice,eta4_slice]):
            sendSize=0
            toSend = np.ndarray(0,order='F')
        else:
            # set sendSize and data to be sent
            toSend = self.f[eta2_slice,eta1_slice,eta3_slice,eta4_slice].flatten()
            sendSize=toSend.size
        
        # collect the send sizes to properly formulate the gatherv
        sizes=self.commEta14.gather(sendSize,root=0)
        
        # the dimensions must be concatenated one at a time to properly shape the output.
        if (self.rankEta14==0):
            # use sizes to get start points
            starts=np.zeros(len(sizes),int)
            starts[1:]=np.cumsum(sizes[:self.sizeEta14-1])
            
            # save memory for gatherv to fill
            mySlice = np.empty(np.sum(sizes), dtype=float)
            
            # Gather information from all ranks to rank 0 in direction eta1 or eta4 (dependant upon layout)
            self.commEta14.Gatherv(toSend,(mySlice, sizes, starts, MPI.DOUBLE), 0)
            
            # get dimensions along which split occurs
            if (self.layout==Layout.FIELD_ALIGNED):
                splitDims=[self.Dimension.ETA1, self.Dimension.ETA4]
            elif (self.layout==Layout.POLOIDAL):
                splitDims=[self.Dimension.ETA4, self.Dimension.ETA3]
            elif (self.layout==Layout.V_PARALLEL):
                splitDims=[self.Dimension.ETA1, self.Dimension.ETA3]
            
            # if the first dimension of the slice is distributed
            if (splitDims[0]==dimIdx[0]):
                # break received data into constituent parts
                mySlice=np.split(mySlice,starts[1:])
                for i in range(0,self.sizeEta14):
                    # return them to original shape
                    mySlice[i]=mySlice[i].reshape(sizes[i]//dimSize[1],dimSize[1])
                # stitch back together along broken axis
                mySlice=np.concatenate(mySlice,axis=0)
            # if the second dimension of the slice is distributed
            elif(splitDims[0]==dimIdx[1]):
                # break received data into constituent parts
                mySlice=np.split(mySlice,starts[1:])
                for i in range(0,self.sizeEta14):
                    # return them to original shape
                    mySlice[i]=mySlice[i].reshape(dimSize[0],sizes[i]//dimSize[0])
                # stitch back together along broken axis
                mySlice=np.concatenate(mySlice,axis=1)
            # else the data will be sent flattened anyway
            
            sizes=self.commEta34.gather(mySlice.size,root=0)
            if (self.rankEta34==0):
                # use sizes to get new start points
                starts=np.zeros(len(sizes),int)
                starts[1:]=np.cumsum(sizes[:self.sizeEta34-1])
                
                # ensure toSend has correct size in memory for gatherv to fill
                toSend = np.empty(np.sum(sizes), dtype=float)
                
                # collect data sent up first column on first cell
                self.commEta34.Gatherv(mySlice,(toSend, sizes, starts, MPI.DOUBLE), 0)
                
                # if the first dimension of the slice is distributed
                if (splitDims[1]==dimIdx[0]):
                    # break received data into constituent parts
                    mySlice=np.split(toSend,starts[1:])
                    for i in range(0,self.sizeEta34):
                        # return them to original shape
                        mySlice[i]=mySlice[i].reshape(sizes[i]//dimSize[1],dimSize[1])
                    # stitch back together along broken axis
                    return np.concatenate(mySlice,axis=0)
                # if the second dimension of the slice is distributed
                elif(splitDims[1]==dimIdx[1]):
                    # break received data into constituent parts
                    mySlice=np.split(toSend,starts[1:])
                    for i in range(0,self.sizeEta34):
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
                self.commEta34.Gatherv(mySlice,mySlice, 0)
        else:
            # Gather information from all ranks
            self.commEta14.Gatherv(toSend,toSend, 0)
    
    def getMin(self,axis = None,fixValue = None):
        
        # if we want the total of all points on the grid
        if (axis==None and fixValue==None):
            # return the min of the min found on each process
            return MPI.COMM_WORLD.reduce(np.amin(self.f),op=MPI.MIN,root=0)
        
        # if we want the total of all points on a 3D slice where the value of eta3 is fixed
        # ensure that the required index is covered by this process
        if (axis==self.Dimension.ETA3 and fixValue>=self.eta3_start and fixValue<self.eta3_end):
            # get the indices of the required slice
            idx = (np.s_[:],) * axis + (fixValue-self.eta3_start,)
            # return the min of the min found on each process's slice
            return MPI.COMM_WORLD.reduce(np.amin(self.f[idx]),op=MPI.MIN,root=0)
        
        # if we want the total of all points on a 3D slice where the value of eta1 is fixed
        # ensure that the required index is covered by this process
        elif (axis==self.Dimension.ETA1 and fixValue>=self.eta1_start and fixValue<self.eta1_end):
            # get the indices of the required slice
            idx = (np.s_[:],) * axis + (fixValue-self.eta1_start,)
            # return the min of the min found on each process's slice
            return MPI.COMM_WORLD.reduce(np.amin(self.f[idx]),op=MPI.MIN,root=0)
        
        # if we want the total of all points on a 3D slice where the value of eta4 is fixed
        # ensure that the required index is covered by this process
        elif (axis==self.Dimension.ETA4 and fixValue>=self.eta4_start and fixValue<self.eta4_end):
            # get the indices of the required slice
            idx = (np.s_[:],) * axis + (fixValue-self.eta4_start,)
            # return the min of the min found on each process's slice
            return MPI.COMM_WORLD.reduce(np.amin(self.f[idx]),op=MPI.MIN,root=0)
        
        # if we want the total of all points on a 3D slice where the value of eta2 is fixed
        # ensure that the required index is covered by this process
        elif (axis==self.Dimension.ETA2):
            # get the indices of the required slice
            idx = (np.s_[:],) * axis + (fixValue,)
            # return the min of the min found on each process's slice
            return MPI.COMM_WORLD.reduce(np.amin(self.f[idx]),op=MPI.MIN,root=0)
        
        # if the data is not on this process then send the largest possible value of f
        # this way min will always choose an alternative
        else:
            return MPI.COMM_WORLD.reduce(1,op=MPI.MIN,root=0)
    
    
    def getMax(self,axis = None,fixValue = None):
        
        # if we want the total of all points on the grid
        if (axis==None and fixValue==None):
            # return the max of the max found on each process
            return MPI.COMM_WORLD.reduce(np.amax(self.f),op=MPI.MAX,root=0)
        
        # if we want the total of all points on a 3D slice where the value of eta3 is fixed
        # ensure that the required index is covered by this process
        if (axis==self.Dimension.ETA3 and fixValue>=self.eta3_start and fixValue<self.eta3_end):
            # get the indices of the required slice
            idx = (np.s_[:],) * axis + (fixValue-self.eta3_start,)
            # return the max of the max found on each process's slice
            return MPI.COMM_WORLD.reduce(np.amax(self.f[idx]),op=MPI.MAX,root=0)
        
        # if we want the total of all points on a 3D slice where the value of eta1 is fixed
        # ensure that the required index is covered by this process
        elif (axis==self.Dimension.ETA1 and fixValue>=self.eta1_start and fixValue<self.eta1_end):
            # get the indices of the required slice
            idx = (np.s_[:],) * axis + (fixValue-self.eta1_start,)
            # return the max of the max found on each process's slice
            return MPI.COMM_WORLD.reduce(np.amax(self.f[idx]),op=MPI.MAX,root=0)
        
        # if we want the total of all points on a 3D slice where the value of eta4 is fixed
        # ensure that the required index is covered by this process
        elif (axis==self.Dimension.ETA4 and fixValue>=self.eta4_start and fixValue<self.eta4_end):
            # get the indices of the required slice
            idx = (np.s_[:],) * axis + (fixValue-self.eta4_start,)
            # return the max of the max found on each process's slice
            return MPI.COMM_WORLD.reduce(np.amax(self.f[idx]),op=MPI.MAX,root=0)
        
        # if we want the total of all points on a 3D slice where the value of eta2 is fixed
        # ensure that the required index is covered by this process
        elif (axis==self.Dimension.ETA2):
            # get the indices of the required slice
            idx = (np.s_[:],) * axis + (fixValue,)
            # return the max of the max found on each process's slice
            return MPI.COMM_WORLD.reduce(np.amax(self.f[idx]),op=MPI.MAX,root=0)
        
        # if the data is not on this process then send the smallest possible value of f
        # this way max will always choose an alternative
        else:
            return MPI.COMM_WORLD.reduce(0,op=MPI.MAX,root=0)
