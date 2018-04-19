def compute_2d_process_grid( npts : list, mpi_size : int ):
    """ Compute 2D grid of processes for the two distributed dimensions
        in each layout.
    """
    nprocs1 = 1
    nprocs2 = 1

    return nprocs1, nprocs2

#===============================================================================

class Layout:

    def __init__( self, name:str, nprocs:list, dims_order:list, eta_grids:list ):

        self._name = name
        self._nprocs = nprocs
        self._dims_order = dims_order
        self._ndims      = len( dims_order )

        assert len( dims_order ) == len( nprocs )

    @property
    def name( self ):
        """ Get layout name.
        """
        return self._name

    @property
    def ndims( self ):
        """ Number of dimensions in layout.
        """
        return self._ndims

    @property
    def dims_order( self ):
        """ Get order of dimensions eta1, eta2, etc... in layout.
        """
        return self._dims_order

    @property
    def starts( self ):
        """ Get global starting points for all indices.
        """
        raise NotImplementedError

    @property
    def shape( self ):
        """ Get shape of data chunk in this layout.
        """
        raise NotImplementedError

    @property
    def nprocs( self ):
        """ Number of processes along each dimension.
        """
        return self._nprocs

#===============================================================================

class TransposeOperator:

    def __init__( self, comm : MPI.Comm, *layouts : Layout ):

        # TODO
        # - Verify compatibility of layouts
        # - Extract nprocs1, nprocs2

        topology = comm.Create_cart( [nprocs1, nprocs2], periods=[False,False],
                reorder=False )

        subcomm1 = topology.Sub( [True, False] )
        subcomm2 = topology.Sub( [False, True] )

        # TODO
        # - store data in object
        self._layouts = layouts


    def transpose( self, source, dest, layout_source, layout_dest ):

        assert source.shape == layout_source.shape
        assert   dest.shape == layout_dest  .shape

        assert layout_source in self._layouts
        assert layout_dest   in self._layouts

        # TODO
        # - Check that path is available
        # - Perform all-to-all operation with subcomm.Alltoallv(...)
