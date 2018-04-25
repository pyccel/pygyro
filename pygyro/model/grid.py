import numpy as np
from enum import Enum, IntEnum
from math import pi

from ..                 import splines as spl
from .layout            import LayoutManager

class Grid(object):
    """
    Grid: Class containing data values

    Parameters
    ----------
    eta_grid : list of array_like
        The coordinates of the grid points in each dimension

    layouts : LayoutManager
        A layout manager containing all the possible layouts

    chosenLayout : str
        The name of the start layout

    """
    def __init__( self, eta_grid: list, layouts: LayoutManager,
                    chosenLayout: str):
        
        # remember layout
        self._layout_manager=layouts
        self._current_layout_name=chosenLayout
        self._layout = layouts.getLayout(chosenLayout)
        shapes = layouts.availableLayouts
        self._my_data = np.empty(self._layout_manager.bufferSize)
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
        return enumerate(self._Vals[self._layout.dims_order[i]][self._layout.starts[i]:self._layout.ends[i]])
    
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
        """ change current layout
        """
        self._f = self._layout_manager.in_place_transpose(
                        self._f,
                        self._current_layout_name,
                        new_layout)
        self._layout = self._layout_manager.getLayout(new_layout)
        self._current_layout_name = new_layout
