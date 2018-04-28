from mpi4py import MPI
import numpy as np
from enum import Enum, IntEnum
from math import pi

from ..                 import splines as spl
from .layout            import LayoutManager

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
        self._my_data = np.empty(self._layout_manager.bufferSize)
        # Remember views on the data
        shapes = layouts.availableLayouts
        views = []
        for (name,shape) in shapes:
            views.append((name,np.split(self._my_data,[np.prod(shape)])[0].reshape(shape)))
        self._views = dict(views)
        self._f = np.split(self._my_data,[self._layout.size])[0].reshape(self._layout.shape)
        
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
    
    def get1DSlice(self, slices: list):
        """ get the 1D slice at the provided list of coordinates
        """
        assert(len(slices)==self._nDims-1)
        slices.append(slice(self._nGlobalCoords[self._layout.dims_order[-1]]))
        return self._f[slices]
    
    def setLayout(self,new_layout: str):
        self._layout_manager.in_place_transpose(
                        self._my_data,
                        self._current_layout_name,
                        new_layout)
        self._layout = self._layout_manager.getLayout(new_layout)
        self._f      = self._views[new_layout]
        self._current_layout_name = new_layout
    
