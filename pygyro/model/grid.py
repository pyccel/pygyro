from mpi4py import MPI
import numpy as np
from math import pi

from ..                 import splines as spl
from .layout            import LayoutManager

class Grid(object):
    def __init__( self, eta_grid: list, bsplines: list, layouts: LayoutManager,
                    chosenLayout: str, comm : MPI.Comm = MPI.COMM_WORLD):
        # get MPI values
        self.global_comm = comm
        self.rank = comm.Get_rank()
        self.mpi_size = comm.Get_size()
        
        # remember layout
        self._layout_manager=layouts
        self._current_layout_name=chosenLayout
        self._layout = layouts.getLayout(chosenLayout)
        self._my_data = [np.empty(self._layout_manager.bufferSize),
                         np.empty(self._layout_manager.bufferSize)]
        self._dataIdx = 0
        self._buffIdx = 1
        
        # Remember views on the data
        shapes = layouts.availableLayouts
        self._f = np.split(self._my_data[self._dataIdx],[self._layout.size])[0].reshape(self._layout.shape)
        
        # save coordinate information
        # saving in list allows simpler reordering of coordinates
        self._Vals = eta_grid
        self._splines = bsplines
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
        """ get enumerate of local coordinates along axis i
        """
        return enumerate(self._Vals[self._layout.dims_order[i]] \
                                   [self._layout.starts[i]:self._layout.ends[i]])
    
    def getEta( self, i : int ):
        """ get enumerate of local coordinates along axis i
        """
        return enumerate(self._Vals[i][ \
                self._layout.starts[self._inv_dims_order[i]] : \
                self._layout.ends[self._inv_dims_order[i]]])
    
    def getCoordVals( self, i : int ):
        """ get values of local coordinates along axis i
        """
        return self._Vals[self._layout.dims_order[i]][self._layout.starts[i]:self._layout.ends[i]]
    
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
        slices.extend([slice(self._nGlobalCoords[self._layout.dims_order[-2]]),
                      slice(self._nGlobalCoords[self._layout.dims_order[-1]])])
        return self._f[slices]
    
    def get2DSpline( self ):
        """ get the splines associated with the last 2 dimensions
        """
        return (self._splines[self._layout.dims_order[-2]],
                self._splines[self._layout.dims_order[-1]])
    
    def getSpline( self, i ):
        """ get the i-th spline (in global coordinates r,theta,z,v)
        """
        return self._splines[i]
    
    def get1DSlice( self, slices: list ):
        """ get the 1D slice at the provided list of coordinates
        """
        assert(len(slices)==self._nDims-1)
        slices.append(slice(self._nGlobalCoords[self._layout.dims_order[-1]]))
        return self._f[slices]
    
    def get1DSpline( self ):
        """ get the spline associated with the last dimension
        """
        return self._splines[self._layout.dims_order[-1]]
    
    def setLayout(self,new_layout: str):
        self._layout_manager.transpose(
                        self._my_data[self._dataIdx],
                        self._my_data[self._buffIdx],
                        self._current_layout_name,
                        new_layout)
        self._dataIdx, self._buffIdx = self._buffIdx, self._dataIdx
        self._layout = self._layout_manager.getLayout(new_layout)
        self._f = np.split(self._my_data[self._dataIdx],[self._layout.size])[0].reshape(self._layout.shape)
        self._current_layout_name = new_layout
    
    ####################################################################
    ####                   Functions for figures                    ####
    ####################################################################
    
    
    def getSliceFromDict(self,d : dict, comm: MPI.Comm, rank: int):
        """
        Utility function to access getSliceForFig without using 6 if statements
        The function takes a dictionary which plots the fixed dimensions to the
        index at which they are fixed
        
        >>> getSliceFromDict({self.Dimension.ETA3: 5, self.Dimension.ETA4 : 2})
        """
        dim1 = d.get(self._layout.dims_order[0],None)
        dim2 = d.get(self._layout.dims_order[1],None)
        dim3 = d.get(self._layout.dims_order[2],None)
        dim4 = d.get(self._layout.dims_order[3],None)
        return self.getSliceForFig([dim1,dim2,dim3,dim4],comm,rank)
    
    def getSliceForFig(self, dims: list, comm: MPI.Comm, rank: int):
        """
        Class to retrieve a 2D slice. Any values set to None will vary.
        Any dimensions not set to None will be fixed at the global index provided
        """
        
        # helper variables to correctly reshape the slice after gathering
        dimSize=[]
        dim_slices = []
        shape = []
        
        # Get slices for each dimension (eta2_slice, eta1_slice, eta3_slice, eta4_slice)
        # If value is None then all values along that dimension should be returned
        # this means that the size and dimension index will be stored
        # If value is not None then only values at that index should be returned
        # if that index cannot be found on the current process then None will
        # be stored
        
        for i,dim_i in enumerate(dims):
            if (dim_i==None):
                dim_slices.append(slice(0,self._layout.ends[i]-self._layout.starts[i]))
                dimSize.append(self._layout.ends[i]-self._layout.starts[i])
            else:
                if (dim_i>=self._layout.starts[i] and dim_i<self._layout.ends[i]):
                    dim_slices.append(dim_i-self._layout.starts[i])
                    dimSize.append(1)
                else:
                    dim_slices.append(None)
                    dimSize.append(None)
        
        # if the data is not on this process then at least one of the slices is equal to None
        # in this case send something of size 0
        sendInfo = self._layout_manager.mpiCoords
        if (None in dim_slices):
            toSend = np.ndarray(0)
            sendInfo.append(0)
        else:
            # set sendSize and data to be sent
            toSend = self._f[dim_slices].flatten()
            sendInfo.append(toSend.size)
        
        mpi_data=comm.gather(sendInfo,root=rank)
        
        if (comm.Get_rank()==rank):
            sizes = [coords.pop() for coords in mpi_data]
            # use sizes to get start points
            starts=np.zeros(len(sizes),int)
            starts[1:]=np.cumsum(sizes[:comm.Get_size()-1])
            
            # save memory for gatherv to fill
            sliceSize = np.sum(sizes)
            mySlice = np.empty(sliceSize, dtype=float)
            
            # Gather information from all ranks to rank 0 in the
            # direction of the comm
            comm.Gatherv(toSend,(mySlice, sizes, starts, MPI.DOUBLE), 0)
            return (self._layout,starts,mpi_data,mySlice)
        else:
            # Gather information from all ranks
            comm.Gatherv(toSend,toSend, 0)
    
    def getMin(self,axis = None,fixValue = None):
        
        # if we want the total of all points on the grid
        if (axis==None and fixValue==None):
            # return the min of the min found on each process
            return self.global_comm.reduce(np.amin(self._f),op=MPI.MIN,root=0)
        
        # if we want the total of all points on a (N-1)D slice where the
        # value of eta_i is fixed ensure that the required index is
        # covered by this process
        dim = self._layout.inv_dims_order[axis]
        if (fixValue>=self._layout.starts[dim] and 
            fixValue<self._layout.ends[dim]):
            idx = (np.s_[:],) * dim + (fixValue-self._layout.starts[dim],)
            return self.global_comm.reduce(np.amin(self._f[idx]),op=MPI.MIN,root=0)
        
        # if the data is not on this process then send the largest possible value of f
        # this way min will always choose an alternative
        else:
            return self.global_comm.reduce(1,op=MPI.MIN,root=0)
    
    def getMax(self,axis = None,fixValue = None):
        
        # if we want the total of all points on the grid
        if (axis==None and fixValue==None):
            # return the max of the max found on each process
            return self.global_comm.reduce(np.amax(self._f),op=MPI.MAX,root=0)
        
        # if we want the total of all points on a (N-1)D slice where the
        # value of eta_i is fixed ensure that the required index is
        # covered by this process
        dim = self._layout.inv_dims_order[axis]
        if (fixValue>=self._layout.starts[dim] and 
            fixValue<self._layout.ends[dim]):
            idx = (np.s_[:],) * dim + (fixValue-self._layout.starts[dim],)
            return self.global_comm.reduce(np.amax(self._f[idx]),op=MPI.MAX,root=0)
        
        # if the data is not on this process then send the smallest possible value of f
        # this way max will always choose an alternative
        else:
            return self.global_comm.reduce(0,op=MPI.MAX,root=0)
