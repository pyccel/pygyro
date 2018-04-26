from mpi4py import MPI
import numpy as np
import warnings

class Layout:
    """
    Layout: Class containing information about how to access data
    in a given layout

    Parameters
    ----------
    name : str
        A name which is used to identify the layout.

    nprocs : list of int
        The number of processes in each distribution direction.

    dims_order : list of int
        The order of the dimensions in this layout.
        E.g. [0,2,1] means that the ordering is (eta1,eta3,eta2)
        
        This should contain at least as many elements as nprocs.

    eta_grids : list of array_like
        The coordinates of the grid points in each dimension

    myRank : list of int
        The rank of the current process in each distribution direction.
    
    Notes
    -----
    We assume that the first dimension in the layout is distributed
    along the number of processes indicated by nprocs[0] etc.

    """
    def __init__( self, name:str, nprocs:list, dims_order:list, eta_grids:list, myRank: list ):
        self._name = name
        self._ndims      = len( dims_order )
        self._dims_order = dims_order
        
        # check input makes sense
        assert len( dims_order ) == len( eta_grids )
        assert len( nprocs ) == len( myRank )
        
        self._inv_dims_order = [0]*self._ndims
        for i,j in enumerate(self._dims_order):
            self._inv_dims_order[j]=i
        
        # get number of processes in each dimension (sorted [eta1,eta2,eta3,eta4])
        self._nprocs = [1]*self._ndims
        myRanks = [0]*self._ndims
        for j,n in enumerate(nprocs):
            self._nprocs[j]=n
            myRanks[j]=myRank[j]
        
        # initialise values used for accessing data (sorted [eta1,eta2,eta3,eta4])
        self._mpi_lengths = []
        self._starts = np.zeros(self._ndims,int)
        self._ends = np.empty(self._ndims,int)
        self._shape = np.empty(self._ndims,int)
        
        for i,nRanks in enumerate(self._nprocs):
            ranks=np.arange(0,nRanks)
            n=len(eta_grids[dims_order[i]])
            Overflow=n%nRanks
            
            # get start indices for all processes
            starts=n//nRanks*ranks + np.minimum(ranks,Overflow)
            # append end index
            starts=np.append(starts,n)
            self._mpi_lengths.append(starts[1:]-starts[:-1])
            
            # save start and indices from list using cartesian ranks
            self._starts[i] = starts[myRanks[i]]
            self._ends[i]   = starts[myRanks[i]+1]
            self._shape[i]  = self._ends[i]-self._starts[i]
        
        self._size = np.prod(self._shape)
    
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
    def inv_dims_order( self ):
        """ Get order of dimensions eta1, eta2, etc... in layout.
        """
        return self._inv_dims_order
    
    @property
    def starts( self ):
        """ Get global starting points for all indices.
        """
        return self._starts
    
    @property
    def ends( self ):
        """ Get global end points for all indices.
        """
        return self._ends
    
    @property
    def shape( self ):
        """ Get shape of data chunk in this layout.
        """
        return self._shape
    
    @property
    def size( self ):
        """ Get size of data chunk in memory.
        """
        return self._size
    
    @property
    def nprocs( self ):
        """ Number of processes along each dimension.
        """
        return self._nprocs
    
    def mpi_lengths( self , i: int ):
        """ Get length of dimension i on all processes
        """
        return self._mpi_lengths[i]

#===============================================================================

class LayoutManager:
    """
    LayoutManager: Class containing information about the different layouts
    available. It handles conversion from one layout to another

    Parameters
    ----------
    comm : MPI.Comm
        The communicator on which the data will be distributed

    layouts : dict
        The keys should be strings which will be used to identify layouts.
        
        The values should be an array_like containing the ordering of
        the dimensions in this layout.
        E.g. [0,2,1] means that the ordering is (eta1,eta3,eta2)
        The length of the values should be at least as long as the
        length of nprocs.

    nprocs : list of int
        The number of processes in each distribution direction.

    eta_grids : list of array_like
        The coordinates of the grid points in each dimension

    """
    def __init__( self, comm : MPI.Comm, layouts : dict, nprocs: list, eta_grids: list ):
        self.rank=comm.Get_rank()
        
        self._nDims=len(nprocs)
        self._nprocs = nprocs
        
        topology = comm.Create_cart( nprocs, periods=[False]*self._nDims,
                reorder=False )
        
        # Get communicator for each dimension
        self._subcomms = []
        for i in range(self._nDims):
            self._subcomms.append(topology.Sub( [i==j for j in range(self._nDims)] ))
        
        mpi_coords = topology.Get_coords(comm.Get_rank())
        
        # Create the layouts and save them in a dictionary
        # Find the largest layout
        layoutObjects = []
        self._buffer_size = 0
        self._shapes = []
        for name,dim_order in layouts.items():
            new_layout = Layout(name,nprocs,dim_order,eta_grids,mpi_coords)
            layoutObjects.append((name,new_layout))
            self._shapes.append((name,new_layout.shape))
            if (new_layout.size>self._buffer_size):
                self._buffer_size=new_layout.size
        self._layouts = dict(layoutObjects)
        self.nLayouts=len(self._layouts)
        
        # Allocate the buffer
        self._buffer = np.empty(self._buffer_size)
        
        # Calculate direct layout connections
        myMap = []
        for n,(name1,l1) in enumerate(layoutObjects):
            myMap.append((name1,[]))
            for i,(name2,l2) in enumerate(layoutObjects[:n]):
                if (self.compatible(l1,l2)):
                    myMap[i][1].append(name1)
                    myMap[n][1].append(name2)
        # Save the connections in a dictionary
        DirectConnections=dict(myMap)
        
        # Save all layout paths in a map and save any remaining unconnected layouts
        unvisited = self._makeConnectionMap(DirectConnections)
        
        # all layouts should have been found
        if (unvisited!=set()):
            s=str()
            for name in unvisited:
                s+=(" '"+name+"'")
            raise RuntimeError("The following layouts could not be connected to preceeding layouts :"+s)
        
    
    def getLayout( self, name):
        return self._layouts[name]
    
    @property
    def availableLayouts( self ):
        return self._shapes
    
    @property
    def bufferSize( self ):
        return self._buffer_size
    
    def transpose( self, source, dest, source_name, dest_name ):
        """
        Change from one layout to another leaving the source layout intact.

        Parameters
        ----------
        source : View of array_like
            A view of the data to be transposed

        dest : array_like
            The entire memory block where the data will be stored
            (including cells which will not be used)

        source_name : str
            The name of the source layout

        dest_name : str
            The name of the destination layout

        Notes
        -----
        source and dest are assumed to not overlap in memory

        """
        
        # Verify that the input makes sense
        assert source_name in self._layouts
        assert dest_name   in self._layouts
        assert all(source.shape == self._layouts[source_name].shape)
        assert dest  .size == self._buffer.size
        
        # If the source and destination are the same then copy the data
        if (source_name==dest_name):
            layout_dest   = self._layouts[  dest_name]
            destView = np.split(dest,[layout_dest.size])[0].reshape(layout_dest.shape)
            destView[:]=source
            return
        
        # Check that a direct path is available
        if (len(self._route_map[source_name][dest_name])==1):
            # if so then carry out the transpose
            layout_source = self._layouts[source_name]
            layout_dest   = self._layouts[  dest_name]
            destView = np.split(dest,[layout_dest.size])[0].reshape(layout_dest.shape)
            assert(not destView.flags['OWNDATA'])
            self._transpose(source,destView,layout_source,layout_dest)
        else:
            # if not reroute the transpose via intermediate steps
            self._transposeRedirect(source,dest,source_name,dest_name)
    
    def in_place_transpose( self, data, source_name, dest_name ):
        """
        Change from one layout to another in the same memory space.

        Parameters
        ----------
        data : array_like
            The entire memory block where the data is currently stored.
            This is also the block where the result will be stored

        source_name : str
            The name of the source layout

        dest_name : str
            The name of the destination layout

        """
        
        # Verify that the input makes sense
        assert source_name in self._layouts
        assert dest_name   in self._layouts
        
        # If the source and destination are the same then there is nothing
        # to be done
        if (source_name==dest_name):
            return
        
        # Check that a direct path is available
        if (len(self._route_map[source_name][dest_name])==1):
            layout_source = self._layouts[source_name]
            layout_dest   = self._layouts[  dest_name]
            assert data.size>=layout_source.size
            assert data.size>=layout_dest.size
            # if so then carry out the transpose
            self._in_place_transpose(data,layout_source,layout_dest)
        else:
            # if not reroute the transpose via intermediate steps
            self._in_place_transposeRedirect(data,source_name,dest_name)
    
    def _transposeRedirect(self,source,dest,source_name,dest_name):
        """
        Function for changing layout via multiple steps leaving the
        source layout intact.
        """
        
        # Get route from one layout to another
        steps = self._route_map[source_name][dest_name]
        nSteps = len(steps)
        
        # warn about multiple steps
        warnings.warn("Changing from %s layout to %s layout requires %i steps" %
                (source_name,dest_name,nSteps))
        
        # take the first step to move the data
        nowLayoutKey=steps[0]
        nowLayout=self._layouts[nowLayoutKey]
        
        assert dest.size>=nowLayout.size
        
        destView = np.split(dest,[nowLayout.size])[0].reshape(nowLayout.shape)
        
        self._transpose(source,destView,self._layouts[source_name],nowLayout)
        
        # carry out subsequent steps in place
        for i in range(1,nSteps):
            nextLayoutKey=steps[i]
            nextLayout=self._layouts[nextLayoutKey]
            assert dest.size>=nextLayout.size
            self._in_place_transpose(dest,nowLayout,nextLayout)
            nowLayout=nextLayout
            nowLayoutKey=nextLayoutKey
    
    def _in_place_transposeRedirect(self,data,source_name,dest_name):
        """
        Function for changing layout via multiple steps.
        """
        # Get route from one layout to another
        steps = self._route_map[source_name][dest_name]
        nSteps = len(steps)
        
        # warn about multiple steps
        warnings.warn("Changing from %s layout to %s layout requires %i steps" %
                (source_name,dest_name,nSteps))
        
        # carry out the steps one by one
        nowLayoutKey=source_name
        nowLayout=self._layouts[source_name]
        for i in range(nSteps):
            nextLayoutKey=steps[i]
            nextLayout=self._layouts[nextLayoutKey]
            
            assert data.size>=nextLayout.size
            
            self._in_place_transpose(data,nowLayout,nextLayout)
            
            nowLayout=nextLayout
            nowLayoutKey=nextLayoutKey
    
    def _transpose(self, source, dest, layout_source, layout_dest):
        # check that the data has the right shape
        assert all(  dest.shape == layout_dest.shape)
        
        # get axis information
        axis = self._get_swap_axes(layout_source,layout_dest)
        if (axis[0]>=self._nDims):
            dest[:]=np.swapaxes(source,axis[0],axis[1])
            return
        
        # carry out transpose
        comm = self._subcomms[axis[0]]
        self._rearrange_to_buffer(source,layout_source,layout_dest,axis,comm)
        self._rearrange_from_buffer(dest,layout_source,layout_dest,axis,comm)
    
    def _in_place_transpose(self, data, layout_source, layout_dest):
        # get views of the important parts of the data
        source = np.split(data,[layout_source.size])[0].reshape(layout_source.shape)
        dest   = np.split(data,[layout_dest  .size])[0].reshape(layout_dest  .shape)
        
        # get axis information
        axis = self._get_swap_axes(layout_source,layout_dest)
        if (axis[0]>=self._nDims):
            dest[:]=np.swapaxes(source,axis[0],axis[1])
            return
        
        # carry out transpose
        comm = self._subcomms[axis[0]]
        self._rearrange_to_buffer(source,layout_source,layout_dest,axis,comm)
        self._rearrange_from_buffer(dest,layout_source,layout_dest,axis,comm)
    
    def _get_swap_axes(self,layout_source,layout_dest):
        # Find the axes which will be swapped
        axis = []
        for i,n in enumerate(layout_source.dims_order):
            if (n!=layout_dest.dims_order[i]):
                axis.append(i)
        # axis[0] is the distributed axis in the source layout
        # axis[1] is the distributed axis in the destination layout
        
        assert(axis[1]==layout_source.ndims-1)
        return axis
    
    def _rearrange_to_buffer(self, source, layout_source : Layout, layout_dest : Layout, axis : list, comm : MPI.Comm):
        
        # get the number of processes that the data is split across
        nSplits=layout_source.nprocs[axis[0]]
        
        size = comm.Get_size()
        
        # Get the sizes of the sent and received blocks.
        # This is equal to the product of the number of points in each
        # dimension except the dimension that was/will be distributed
        # multiplied by the number of points in the distributed direction
        # (e.g. when going from n_eta1=10, n_eta2=10, n_eta3=20, n_eta4=10
        # to n_eta1=20, n_eta2=10, n_eta3=10, n_eta4=10
        # the sent sizes are (10*10*[10 10]*10) and the received sizes
        # are ([10 10]*10*10*10)
        sourceSizes = layout_source.size *                      \
                      layout_dest  .mpi_lengths( axis[0] ) //   \
                      layout_source.shape[axis[1]]
        
        destSizes   = layout_dest  .size *                      \
                      layout_source.mpi_lengths( axis[0] ) //   \
                      layout_dest  .shape[axis[1]]
        
        # get the start points in the array
        sourceStarts              = np.zeros(size,int)
        self._buffer_splits       = np.zeros(size+1,int)
        sourceStarts[1:]          = np.cumsum(sourceSizes)[:-1]
        self._buffer_splits  [1:] = np.cumsum(  destSizes)
        
        # check the sizes have been computed correctly
        assert(sum(sourceSizes)==layout_source.size)
        assert(sum(destSizes)==layout_dest.size)
        
        # Data is split and sent to the appropriate processes
        # Spitting using Alltoallv and a transpose only works when the dimension to split
        # is the last dimension
        # reshape is used instead of flatten as flatten creates a copy but reshape doesn't
        source.transpose().reshape(source.size).flags
        comm.Alltoallv( ( source.transpose().reshape(source.size)   ,
                          ( sourceSizes, sourceStarts )             ,
                          MPI.DOUBLE                                ) ,
                        
                        ( self._buffer                              , 
                          ( destSizes  , self._buffer_splits[:-1]   ) ,
                          MPI.DOUBLE                                ) )
    
    def _rearrange_from_buffer(self, dest, layout_source : Layout, 
                                layout_dest : Layout, axis : list , comm: MPI.Comm):
        rank = comm.Get_rank()
        
        # For the data to be correctly reconstructed it is first split
        # back into its constituent parts
        # newData is a view on self._buffer
        newData=np.split(self._buffer,self._buffer_splits[1:])
        newData.pop()
        lastDim=len(layout_source.shape)-1
        
        for i,arr in enumerate(newData):
            # The data must be reshaped into the transmitted shape
            # (with source ordering)
            shape=layout_source.shape.copy()
            # The size of the data in the previously distributed direction
            # depends on which processo provided the data. Due to the
            # splitting method this is equal to the array index
            shape[axis[0]]=layout_source.mpi_lengths(axis[0])[i]
            # The size of the data in the newly distributed direction is
            # the same for all blocks on this process
            shape[axis[1]]=layout_dest.mpi_lengths(axis[0])[rank]
            # The data is reshaped to the trasmitted shape
            arr=np.reshape(arr,shape[::-1])
            # The array is then rearranged to the used ordering by
            # once more transposing to return the ordering to its original
            # state then swapping the axes that were/are distributed
            newData[i]=np.swapaxes(arr.transpose(),axis[0],axis[1])
        
        # Stitch the pieces together in the correct order.
        # This is done along the non-distributed axis which, after the
        # axes swap is the second axis
        np.concatenate(newData,axis=axis[1], out = dest)
    
    def compatible(self, l1: Layout, l2: Layout):
        """
        Check if the data can be passed from one layout to the other in
        one step

        Parameters
        ----------
        l1 : Layout
            The first layout

        l2 : Layout
            The second layout

        """
        dims = []
        nDims = len(l1.dims_order)
        lastDimDistributed = False
        # check the ordering of the dimensions
        for i,o in enumerate(l1.dims_order):
            if (o!=l2.dims_order[i]):
                # Save dimensions where the ordering differs
                dims.append(o)
                dims.append(l2.dims_order[i])
                if (i==nDims-1):
                    lastDimDistributed=True
        # The dimension ordering of compatible layouts should be identical
        # except in the last dimension and one other dimension.
        # The values in these dimensions should be swapped
        # e.g. if l1.dims_order=[a,b,c,d] l2.dims_order can be
        # [d,b,c,a], [a,d,c,b], or [a,b,d,c]
        # This means if a,b are the swapped values that dim should contain [a,b,b,a]
        return (len(dims)==4 and lastDimDistributed and dims[0]==dims[3] and dims[1]==dims[2])

    def _makeConnectionMap(self, DirectConnections: dict):
        """
        Find the shortest path connected each pair of layouts.
        Return any unconnected layouts
        """
        # Make a map to be used for reference when transposing
        # The keys are the source layout name
        # The keys of the resulting dictionary are the destination layout name
        # The values are lists containing the steps that must be taken
        # to travel from the source to the destination
        MyMap = []
        for name1 in DirectConnections.keys():
            Routing = []
            for name2 in DirectConnections.keys():
                # Make a map of layout names to lists
                Routing.append((name2,[]))
            MyMap.append((name1,dict(Routing)))
        self._route_map = dict(MyMap)
        
        # Create helpful arrays
        visitedNodes   = []
        remainingNodes = set(DirectConnections.keys())
        nodesToVisit   = [list(DirectConnections.keys())[0]]
        remainingNodes.remove(nodesToVisit[0])
        
        # While there are still nodes connected to the nodes visited so far
        while (len(nodesToVisit)>0):
            # get node
            currentNode = nodesToVisit.pop(0)
            
            # Add connected nodes that have not yet been considered to list of nodes to check
            for name in DirectConnections[currentNode]:
                if (name in remainingNodes):
                    nodesToVisit.append(name)
                    remainingNodes.remove(name)
            
            # Find a path to each node that has already been considered
            for aim in visitedNodes:
                if (aim in DirectConnections[currentNode]):
                    # If the node is directly connected then save the path
                    self._route_map[currentNode][aim].append(aim)
                    self._route_map[aim][currentNode].append(currentNode)
                else:
                    # If the node is not directly connected, find the connected
                    # node which has the shortest path to the node for which we
                    # are aiming
                    viaList = []
                    winningVia = None
                    shortestLength = len(DirectConnections)
                    for via in DirectConnections[currentNode]:
                        if (not via in visitedNodes):
                            continue
                        elif(len(self._route_map[via][aim])<shortestLength):
                            viaList = self._route_map[via][aim]
                            winningVia = via
                    # Save the path via this node
                    self._route_map[currentNode][aim].append(winningVia)
                    self._route_map[currentNode][aim].extend(viaList)
                    self._route_map[aim][currentNode].extend(self._route_map[aim][winningVia])
                    self._route_map[aim][currentNode].append(currentNode)
            # Remember that this node has now been visited
            visitedNodes.append(currentNode)
        
        # Return unconnected nodes
        return remainingNodes
