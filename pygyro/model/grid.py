from mpi4py import MPI
import numpy as np
from enum import Enum, IntEnum
from math import pi

from ..                 import splines as spl
from .layout            import LayoutManager

class Layout(Enum):
    FIELD_ALIGNED = 1
    V_PARALLEL = 2
    POLOIDAL = 3

class Grid(object):
    class Dimension(IntEnum):
        ETA1 = 0
        ETA2 = 3
        ETA3 = 1
        ETA4 = 2
    
    def __init__( self, eta_grid: list, layouts: LayoutManager,
                    chosenLayout: str, comm : MPI.Comm = MPI.COMM_WORLD):
        # get MPI values
        self.global_comm = comm
        self.rank = comm.Get_rank()
        self.mpi_size = comm.Get_size()
        
        # remember layout
        self._layout_manager=layouts
        self._current_layout_name=chosenLayout
        self._layout = layouts.getLayout(chosenLayout)
        shapes = layouts.availableLayouts
        data = []
        for (name,shape) in shapes:
            data.append((name,np.empty(shape,float)))
        self._my_data = dict(data)
        self._f = self._my_data[chosenLayout]
        
        # save coordinate information
        # saving in list allows simpler reordering of coordinates
        self._Vals = eta_grid
        self._nDims = len(eta_grid)
        self._nGlobalCoords = [len(x) for x in eta_grid]
    
    @property
    def nGlobalCoords( self ):
        """ Number of points in each dimension.
        """
        return self._nGlobalCoords
    
    @property
    def eta_grid( self ):
        """ get grid of global coordinates
        """
        return self._Vals
    
    def getCoords( self, i : int ):
        """ get local coordinates along axis i
        """
        return enumerate(self._Vals[self._layout.dims_order[i]][self._layout.starts[i]:self._layout.ends[i]])
    
    def getGlobalIndices( self, indices: list):
        """ convert local indices to global indices
        """
        result = indices.copy()
        for i,toAdd in enumerate(self._layout.starts):
            result[self._layout.dims_order[i]]=indices[i]+toAdd
        return result
    
    def get2DSlice( self, slices: list ):
        """ get the 2D slice at the provided list of coordinates
        """
        assert(len(slices)==self._nDims-2)
        slices.append(slice(self._nGlobalCoords[self._layout.dims_order[-2]]),
                      slice(self._nGlobalCoords[self._layout.dims_order[-1]]))
        return self._f[slices]
    
    def get1DSlice(self, slices: list):
        """ get the 1D slice at the provided list of coordinates
        """
        assert(len(slices)==self._nDims-1)
        slices.append(slice(self._nGlobalCoords[self._layout.dims_order[-1]]))
        return self._f[slices]
    
    def setLayout(self,new_layout: str):
        self._layout_manager.transpose( self._f,
                                        self._my_data[new_layout],
                                        self._current_layout_name,
                                        new_layout)
        self._f=self._my_data[new_layout]
        self._layout=self._layout_manager.getLayout(new_layout)
        
    
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
        if (eta1==None):
            eta1_slice=slice(0,self.eta1_end-self.eta1_start)
            dimSize.append(self.nEta1)
            dimIdx.append(self.Dimension.ETA1)
        else:
            eta1_slice=None
            if (eta1>=self.eta1_start and eta1<self.eta1_end):
                eta1_slice=eta1-self.eta1_start
        if (eta3==None):
            eta3_slice=slice(0,self.eta3_end-self.eta3_start)
            dimSize.append(self.nEta3)
            dimIdx.append(self.Dimension.ETA3)
        else:
            eta3_slice=None
            if (eta3>=self.eta3_start and eta3<self.eta3_end):
                eta3_slice=eta3-self.eta3_start
        if (eta4==None):
            eta4_slice=slice(0,self.eta4_end-self.eta4_start)
            dimSize.append(self.nEta4)
            dimIdx.append(self.Dimension.ETA4)
        else:
            eta4_slice=None
            if (eta4>=self.eta4_start and eta4<self.eta4_end):
                eta4_slice=eta4-self.eta4_start
        if (eta2==None):
            eta2_slice=slice(0,self.nEta2)
            dimSize.append(self.nEta2)
            dimIdx.append(self.Dimension.ETA2)
        else:
            eta2_slice=eta2
        
        # if the data is not on this process then at least one of the slices is equal to None
        # in this case send something of size 0
        if (None in [eta1_slice,eta3_slice,eta4_slice,eta2_slice]):
            sendSize=0
            toSend = np.ndarray(0)
        else:
            # set sendSize and data to be sent
            toSend = self.f[eta1_slice,eta3_slice,eta4_slice,eta2_slice].flatten()
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
            return self.global_comm.reduce(np.amin(self.f),op=MPI.MIN,root=0)
        
        # if we want the total of all points on a 3D slice where the value of eta3 is fixed
        # ensure that the required index is covered by this process
        if (axis==self.Dimension.ETA3 and fixValue>=self.eta3_start and fixValue<self.eta3_end):
            # get the indices of the required slice
            idx = (np.s_[:],) * axis + (fixValue-self.eta3_start,)
            # return the min of the min found on each process's slice
            return self.global_comm.reduce(np.amin(self.f[idx]),op=MPI.MIN,root=0)
        
        # if we want the total of all points on a 3D slice where the value of eta1 is fixed
        # ensure that the required index is covered by this process
        elif (axis==self.Dimension.ETA1 and fixValue>=self.eta1_start and fixValue<self.eta1_end):
            # get the indices of the required slice
            idx = (np.s_[:],) * axis + (fixValue-self.eta1_start,)
            # return the min of the min found on each process's slice
            return self.global_comm.reduce(np.amin(self.f[idx]),op=MPI.MIN,root=0)
        
        # if we want the total of all points on a 3D slice where the value of eta4 is fixed
        # ensure that the required index is covered by this process
        elif (axis==self.Dimension.ETA4 and fixValue>=self.eta4_start and fixValue<self.eta4_end):
            # get the indices of the required slice
            idx = (np.s_[:],) * axis + (fixValue-self.eta4_start,)
            # return the min of the min found on each process's slice
            return self.global_comm.reduce(np.amin(self.f[idx]),op=MPI.MIN,root=0)
        
        # if we want the total of all points on a 3D slice where the value of eta2 is fixed
        # ensure that the required index is covered by this process
        elif (axis==self.Dimension.ETA2):
            # get the indices of the required slice
            idx = (np.s_[:],) * axis + (fixValue,)
            # return the min of the min found on each process's slice
            return self.global_comm.reduce(np.amin(self.f[idx]),op=MPI.MIN,root=0)
        
        # if the data is not on this process then send the largest possible value of f
        # this way min will always choose an alternative
        else:
            return self.global_comm.reduce(1,op=MPI.MIN,root=0)
    
    
    def getMax(self,axis = None,fixValue = None):
        
        # if we want the total of all points on the grid
        if (axis==None and fixValue==None):
            # return the max of the max found on each process
            return self.global_comm.reduce(np.amax(self.f),op=MPI.MAX,root=0)
        
        # if we want the total of all points on a 3D slice where the value of eta3 is fixed
        # ensure that the required index is covered by this process
        if (axis==self.Dimension.ETA3 and fixValue>=self.eta3_start and fixValue<self.eta3_end):
            # get the indices of the required slice
            idx = (np.s_[:],) * axis + (fixValue-self.eta3_start,)
            # return the max of the max found on each process's slice
            return self.global_comm.reduce(np.amax(self.f[idx]),op=MPI.MAX,root=0)
        
        # if we want the total of all points on a 3D slice where the value of eta1 is fixed
        # ensure that the required index is covered by this process
        elif (axis==self.Dimension.ETA1 and fixValue>=self.eta1_start and fixValue<self.eta1_end):
            # get the indices of the required slice
            idx = (np.s_[:],) * axis + (fixValue-self.eta1_start,)
            # return the max of the max found on each process's slice
            return self.global_comm.reduce(np.amax(self.f[idx]),op=MPI.MAX,root=0)
        
        # if we want the total of all points on a 3D slice where the value of eta4 is fixed
        # ensure that the required index is covered by this process
        elif (axis==self.Dimension.ETA4 and fixValue>=self.eta4_start and fixValue<self.eta4_end):
            # get the indices of the required slice
            idx = (np.s_[:],) * axis + (fixValue-self.eta4_start,)
            # return the max of the max found on each process's slice
            return self.global_comm.reduce(np.amax(self.f[idx]),op=MPI.MAX,root=0)
        
        # if we want the total of all points on a 3D slice where the value of eta2 is fixed
        # ensure that the required index is covered by this process
        elif (axis==self.Dimension.ETA2):
            # get the indices of the required slice
            idx = (np.s_[:],) * axis + (fixValue,)
            # return the max of the max found on each process's slice
            return self.global_comm.reduce(np.amax(self.f[idx]),op=MPI.MAX,root=0)
        
        # if the data is not on this process then send the smallest possible value of f
        # this way max will always choose an alternative
        else:
            return self.global_comm.reduce(0,op=MPI.MAX,root=0)
