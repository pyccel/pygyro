from mpi4py import MPI
import numpy as np
from math import pi
import h5py

from ..                 import splines as spl
from .layout            import LayoutManager

class Grid(object):
    def __init__( self, eta_grid: list, bsplines: list, layouts: LayoutManager,
                    chosenLayout: str, comm : MPI.Comm = MPI.COMM_WORLD, **kwargs):
        dtype = kwargs.pop('dtype',float)
        self.hasSaveMemory = kwargs.pop('allocateSaveMemory',False)
        
        # get MPI values
        self.global_comm = comm
        self.rank = comm.Get_rank()
        self.mpi_size = comm.Get_size()
        
        # remember layout
        self._layout_manager=layouts
        self._current_layout_name=chosenLayout
        self._layout = layouts.getLayout(chosenLayout)
        if (self.hasSaveMemory):
            self._my_data = [np.empty(self._layout_manager.bufferSize,dtype=dtype),
                             np.empty(self._layout_manager.bufferSize,dtype=dtype),
                             np.empty(self._layout_manager.bufferSize,dtype=dtype)]
            self.notSaved = True
        else:
            self._my_data = [np.empty(self._layout_manager.bufferSize,dtype=dtype),
                             np.empty(self._layout_manager.bufferSize,dtype=dtype)]
        
        self._dataIdx = 0
        self._buffIdx = 1
        self._saveIdx = 2
        
        # Remember views on the data
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
    
    
    def getGlobalIdxVals( self, i : int ):
        """ get global indices of local coordinates along axis i
        """
        return range(self._layout.starts[i],self._layout.ends[i])
    
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
    
    def getLayout( self, name: str ):
        """ Return requested layout
        """
        return self._layout_manager.getLayout(name)
    
    @property
    def currentLayout( self ):
        """ Return name of current layout
        """
        return self._current_layout_name
    
    def saveGridValues( self ):
        """ Save current values into a buffer.
            This location is protected until it is freed or restored
        """
        assert(self.hasSaveMemory)
        assert(self.notSaved)
        
        self._my_data[self._saveIdx][:self._layout.size] = self._f[:].flatten()
        self._savedLayout = self._current_layout_name
        
        self.notSaved = False
    
    def freeGridSave( self ):
        """ Signal that the saved grid data is no longer needed and can
            be overwritten
        """
        assert(self.hasSaveMemory)
        assert(not self.notSaved)
        self.notSaved = True
    
    def restoreGridValues( self ):
        """ Restore the values from the saved grid data
        """
        assert(self.hasSaveMemory)
        assert(not self.notSaved)
        
        self._dataIdx, self._saveIdx = self._saveIdx, self._dataIdx
        self.notSaved = True
        self._current_layout_name = self._savedLayout
        self._layout = self._layout_manager.getLayout(self._current_layout_name)
        self._f = np.split(self._my_data[self._dataIdx],[self._layout.size])[0].reshape(self._layout.shape)
    
    def getH5Dataset( self, file ):
        """ Create a hdf5 dataset containing all points in the current layout
        """
        dset = file.create_dataset("dset",self._layout.fullShape, dtype = self._f.dtype)
        slices = tuple([slice(s,e) for s,e in zip(self._layout.starts,self._layout.ends)])
        dset[slices]=self._f[:]
        attr_data = np.array(self._layout.dims_order)
        dset.attrs.create("Layout", attr_data, (self._nDims,), h5py.h5t.STD_I32BE)
