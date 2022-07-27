from mpi4py import MPI
import numpy as np
import warnings
import operator

from abc import ABC


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

    def __init__(self, name: str, nprocs: list, dims_order: list, eta_grids: list, myRank: list):
        self._name = name
        self._ndims = len(dims_order)
        self._dims_order = tuple(dims_order)

        # check input makes sense
        assert len(dims_order) == len(eta_grids)
        assert len(nprocs) == len(myRank)

        self._inv_dims_order = [0]*self._ndims
        for i, j in enumerate(self._dims_order):
            self._inv_dims_order[j] = i

        self._inv_dims_order = tuple(self._inv_dims_order)

        # get number of processes in each dimension (sorted [eta1,eta2,eta3,eta4])
        self._nprocs = [1]*self._ndims
        myRanks = [0]*self._ndims
        for j, n in enumerate(nprocs):
            self._nprocs[j] = n
            myRanks[j] = myRank[j]

        # initialise values used for accessing data (sorted [eta1,eta2,eta3,eta4])
        self._mpi_lengths = []
        self._mpi_starts = []
        self._starts = np.zeros(self._ndims, int)
        self._ends = np.empty(self._ndims, int)
        self._shape = np.empty(self._ndims, int)
        self._max_shape = np.empty(self._ndims, int)

        for i, nRanks in enumerate(self._nprocs):
            ranks = np.arange(0, nRanks+1)

            n = len(eta_grids[dims_order[i]])

            # Find block sizes
            small_size = n//nRanks
            big_size = small_size+1

            # Find number of blocks of each size
            nBig = n % nRanks

            # Get start indices for all processes
            # The different sized blocks should be optimally distributed
            # This means that if the distribution is changed elsewhere,
            # the load balance should remain good
            starts = small_size*ranks+nBig*ranks//nRanks

            self._mpi_starts.append(starts[:-1])
            # append end index
            self._mpi_lengths.append(starts[1:]-starts[:-1])

            # save start and indices from list using cartesian ranks
            self._starts[i] = starts[myRanks[i]]
            self._ends[i] = starts[myRanks[i]+1]
            self._shape[i] = self._ends[i]-self._starts[i]
            self._max_shape[i] = big_size if nBig > 0 else small_size

        self._size = np.prod(self._shape)
        self._max_size = np.prod(self._max_shape)

        self._full_shape = tuple([len(eta_grids[i]) for i in dims_order])
        self._shape = tuple(self._shape)
        self._mpi_starts = tuple(self._mpi_starts)
        self._mpi_lengths = tuple(self._mpi_lengths)
        self._ranks = myRanks

    @property
    def name(self):
        """ Get layout name.
        """
        return self._name

    @property
    def ndims(self):
        """ Number of dimensions in layout.
        """
        return self._ndims

    @property
    def dims_order(self):
        """ Get order of dimensions eta1, eta2, etc... in layout.
        """
        return self._dims_order

    @property
    def inv_dims_order(self):
        """ Get global order of dimensions eta1, eta2, etc... from layout ordering.
        """
        return self._inv_dims_order

    @property
    def starts(self):
        """ Get global starting points for all indices.
        """
        return self._starts

    @property
    def ends(self):
        """ Get global end points for all indices.
        """
        return self._ends

    @property
    def shape(self):
        """ Get shape of data chunk in this layout.
        """
        return self._shape

    @property
    def fullShape(self):
        """ Get shape of all data in this layout.
        """
        return self._full_shape

    @property
    def max_block_shape(self):
        """ Get shape of largest data chunk in this layout.
        """
        return self._max_shape

    @property
    def size(self):
        """ Get size of data chunk in memory.
        """
        return self._size

    @property
    def max_block_size(self):
        """ Get size of the largest data chunk.
        """
        return self._max_size

    @property
    def nprocs(self):
        """ Number of processes along each dimension.
        """
        return self._nprocs

    @property
    def ranks(self):
        """ Rank of the thread on each process associated with this layout
        """
        return self._ranks

    def mpi_starts(self, i: int):
        """ Get global starting points for dimension i on all processes
        """
        return self._mpi_starts[i]

    def mpi_lengths(self, i: int):
        """ Get length of dimension i on all processes
        """
        return self._mpi_lengths[i]

# ===============================================================================


class LayoutManager(ABC):
    """
    LayoutManager: Class containing information about the different layouts.
    It handles conversion from one layout to another
    This class is a super class and should never be instantiated
    """

    def getLayout(self, name: str):
        """ Returns the requested layout
        """
        return self._layouts[name]

    @property
    def nProcs(self):
        """ The number of processes available on the distributed dimensions
        """
        return self._nprocs

    @property
    def mpiCoords(self):
        """ The coordinates of the current processor on the MPI grid communicator
        """
        return self._mpi_coords.copy()

    @property
    def nDistributedDirections(self):
        """ The number of distributed directions
        """
        return self._nDims

    @property
    def availableLayouts(self):
        """ The names and shapes of possible layouts
        """
        return self._shapes

    @property
    def bufferSize(self):
        """ The size of the buffer required to hold all data
        """
        return self._buffer_size

    def _makeConnectionMap(self, DirectConnections: dict):
        """
        Find the shortest path connected each pair of layouts.
        Return any unconnected layouts
        """
        if (len(DirectConnections) == 1):
            return True

        # Make a map to be used for reference when transposing
        # The keys are the source layout name
        # The keys of the resulting dictionary are the destination layout name
        # The values are lists containing the steps that must be taken
        # to travel from the source to the destination
        # A similar map is created containing the number of steps between layouts
        MyMap = []
        distanceMap = []

        for name1 in DirectConnections.keys():
            Routing = []
            dist = []

            for name2 in DirectConnections.keys():
                # Make a map of layout names to lists
                if (name1 != name2):
                    Routing.append((name2, []))
                    dist.append((name2, len(DirectConnections)+1))

            MyMap.append((name1, dict(Routing)))

            distanceMap.append((name1, dict(dist)))

        self._route_map = dict(MyMap)
        distanceMap = dict(distanceMap)

        # Initialise the distances and routes where known
        for name in DirectConnections.keys():
            for stepTo in DirectConnections[name]:
                distanceMap[name][stepTo] = 1
                self._route_map[name][stepTo].append(stepTo)

        for source in DirectConnections.keys():
            # Use Dijkstra's algorithm to find the shortest path from a
            # node 'source' to any other node

            # Create a set of all nodes which must be found
            unvisitedNodes = set(DirectConnections.keys())
            unvisitedNodes.remove(source)

            while (len(unvisitedNodes) > 0):
                # Chose a current node (the nearest node to the source which
                # has not yet been visited
                via = min(unvisitedNodes, key=lambda x: distanceMap[source][x])
                unvisitedNodes.remove(via)

                # Check the path to each neighbour node via the current node
                for aim in DirectConnections[via]:

                    if (aim not in unvisitedNodes):
                        continue

                    # If this route is shorter than any previously considered
                    # route then save it
                    if (distanceMap[source][via]+distanceMap[via][aim] < distanceMap[source][aim]):
                        distanceMap[source][aim] = distanceMap[source][via] + \
                            distanceMap[via][aim]

                        distanceMap[aim][source] = distanceMap[via][source] + \
                            distanceMap[aim][via]

                        self._route_map[source][aim] = self._route_map[source][via] + \
                            self._route_map[via][aim]

                        self._route_map[aim][source] = self._route_map[aim][via] + \
                            self._route_map[via][source]

                    # If this route is the same length as the shortest previously
                    # considered route:
                    elif (distanceMap[source][via]+distanceMap[via][aim] == distanceMap[source][aim]):
                        # Then save it if the route names are smaller
                        # This ensures the algorithm is determinist
                        if (self._route_map[source][via]+self._route_map[via][aim] < self._route_map[source][aim]):
                            self._route_map[source][aim] = self._route_map[source][via] + \
                                self._route_map[via][aim]

                            self._route_map[aim][source] = self._route_map[aim][via] + \
                                self._route_map[via][source]

        # Return true if all nodes are connected, false otherwise
        return max(max(distanceMap.values(), key=lambda x: max(x.values())).values()) != len(DirectConnections)+1

# ===============================================================================


def getLayoutHandler(comm: MPI.Comm, layouts: dict, nprocs: list, eta_grids: list):
    """
    getLayoutHandler: Create a LayoutHandler object with the described
    layouts spread over the designated number of processes.

    This function handles the creation of the sub-communicators which
    are taken as an argument for the creation of a LayoutHandler.

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

    Returns
    -------
    An instance of the LayoutHandler class

    """
    nDims = len(nprocs)

    topology = comm.Create_cart(nprocs, periods=[False]*nDims)

    # Get communicator for each dimension
    subcomms = []
    for i in range(nDims):
        subcomms.append(topology.Sub([i == j for j in range(nDims)]))

    mpi_coords = topology.Get_coords(comm.Get_rank())
    return LayoutHandler(subcomms, mpi_coords, layouts, nprocs, eta_grids)


class LayoutHandler(LayoutManager):
    """
    LayoutHandler: Class containing information about the different layouts
    available. It handles conversion from one layout to another

    Parameters
    ----------
    comms : list of MPI.Comm
        The sub-communicators on which the data will be distributed

    coords : list of int
        The rank on each sub-communicator

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

    def __init__(self, comms: list, coords: list, layouts: dict, nprocs: list, eta_grids: list):
        self._subcomms = comms
        self._mpi_coords = coords

        self._nAxes = len(nprocs)
        self._nDims = len(nprocs)-nprocs.count(1)
        self._nprocsList = list(np.atleast_1d(nprocs))
        self._nprocs = nprocs

        # Create the layouts and save them in a dictionary
        # Find the largest layout
        layoutObjects = []
        self._shapes = []
        for name, dim_order in layouts.items():
            new_layout = Layout(name, nprocs, dim_order,
                                eta_grids, self._mpi_coords)
            layoutObjects.append((name, new_layout))
            self._shapes.append((name, new_layout.shape))
        self._layouts = dict(layoutObjects)
        self.nLayouts = len(self._layouts)

        # Initialise the buffer size before the loop
        self._buffer_size = layoutObjects[0][1].size

        # Calculate direct layout connections
        myMap = []
        for n, (name1, l1) in enumerate(layoutObjects):
            myMap.append((name1, []))
            for i, (name2, l2) in enumerate(layoutObjects[:n]):
                if (self.compatible(l1, l2)):
                    myMap[i][1].append(name1)
                    myMap[n][1].append(name2)

                    # Find the size of the block required to switch between these layouts
                    blockshape = list(l1.shape)
                    axis = self._get_swap_axes(l1, l2)

                    if (len(axis) != 0):
                        blockshape[axis[0]] = l1.max_block_shape[axis[0]]
                        blockshape[axis[1]] = l2.max_block_shape[axis[0]]

                        # Find the maximum memory required
                        if (axis[0] < len(self._subcomms) and self._subcomms[axis[0]] != None):
                            buffsize = np.prod(blockshape) * \
                                self._subcomms[axis[0]].Get_size()
                        else:
                            buffsize = np.prod(blockshape)

                    else:
                        buffsize = np.prod(blockshape)

                    if (buffsize > self._buffer_size):
                        self._buffer_size = buffsize

        # Save the connections in a dictionary
        DirectConnections = dict(myMap)

        # Save all layout paths in a map and save any remaining unconnected layouts
        full = self._makeConnectionMap(DirectConnections)

        # all layouts should have been found
        if (not full):
            raise RuntimeError("Not all layouts could not be connected")

    @property
    def communicators(self):
        """ The communicators used by the LayoutManager
        """
        return tuple(self._subcomms)

    def getLayout(self, name: str):
        """ Return the requested layout
        """
        return self._layouts[name]

    def transpose(self, source, dest, source_name, dest_name, buf=None):
        """
        Change from one layout to another, if a buffer is provided then
        this will leave the source layout intact. Otherwise the source
        will be used as the buffer

        Parameters
        ----------
        source : array_like
            The entire memory block where the data is currently stored
            (including unused cells)

        dest : array_like
            The entire memory block where the data will be stored
            (including cells which will not be used)

        source_name : str
            The name of the source layout

        dest_name : str
            The name of the destination layout

        buf : array_like
            A memory block large enough to store all data. If this
            parameter is not provided then source will not be kept intact

        Notes
        -----
        source and dest are assumed to not overlap in memory

        """
        # If this thread is only here for plotting purposes then ignore the command
        if (self._buffer_size == 0):
            return

        # Verify that the input makes sense
        assert source_name in self._layouts
        assert dest_name in self._layouts
        assert source.size >= self._buffer_size
        assert dest  .size >= self._buffer_size

        # If the source and destination are the same then copy the data
        if (source_name == dest_name):
            layout_dest = self._layouts[dest_name]
            destView = np.split(dest, [layout_dest.size])[0]
            destView[:] = np.split(source, [layout_dest.size])[0]
            return

        # Check that a direct path is available
        if (len(self._route_map[source_name][dest_name]) == 1):
            # if so then carry out the transpose
            layout_source = self._layouts[source_name]
            layout_dest = self._layouts[dest_name]

            if (buf is None):
                self._transpose(source, dest, layout_source, layout_dest)
            else:
                self._transpose_source_intact(
                    source, dest, buf, layout_source, layout_dest)

        else:
            # if not reroute the transpose via intermediate steps
            if (buf is None):
                self._transposeRedirect(source, dest, source_name, dest_name)
            else:
                self._transposeRedirect_source_intact(
                    source, dest, buf, source_name, dest_name)

    def _transposeRedirect(self, source, dest, source_name, dest_name):
        """
        Function for changing layout via multiple steps leaving the
        source layout intact.
        """

        # Get route from one layout to another
        steps = self._route_map[source_name][dest_name]
        nSteps = len(steps)

        # warn about multiple steps
        warnings.warn("Changing from {0} layout to {1} layout requires {2} steps"
                      .format(source_name, dest_name, nSteps))

        # carry out the steps one by one
        nowLayout = self._layouts[source_name]

        fromBuf = source
        toBuf = dest

        for i in range(nSteps):
            nextLayoutKey = steps[i]
            nextLayout = self._layouts[nextLayoutKey]
            self._transpose(fromBuf, toBuf, nowLayout, nextLayout)
            nowLayout = nextLayout
            fromBuf, toBuf = toBuf, fromBuf

        # Ensure the result is found in the expected place
        if (nSteps % 2 == 0):
            dest[:] = source

    def _transposeRedirect_source_intact(self, source, dest, buf, source_name, dest_name):
        """
        Function for changing layout via multiple steps.
        """
        # Get route from one layout to another
        steps = self._route_map[source_name][dest_name]
        nSteps = len(steps)

        # warn about multiple steps
        warnings.warn("Changing from {0} layout to {1} layout requires {2} steps"
                      .format(source_name, dest_name, nSteps))

        # take the first step to move the data
        nowLayoutKey = steps[0]
        nowLayout = self._layouts[nowLayoutKey]

        # The first step needs to place the data in such a way that the
        # final result will be stored in the destination
        if (nSteps % 2 == 0):
            self._transpose_source_intact(
                source, buf, dest, self._layouts[source_name], nowLayout)

            fromBuf = buf
            toBuf = dest

        else:
            self._transpose_source_intact(
                source, dest, buf, self._layouts[source_name], nowLayout)

            fromBuf = dest
            toBuf = buf

        # take the remaining steps
        for i in range(1, nSteps):
            nextLayoutKey = steps[i]
            nextLayout = self._layouts[nextLayoutKey]

            self._transpose(fromBuf, toBuf, nowLayout, nextLayout)

            nowLayout = nextLayout
            nowLayoutKey = nextLayoutKey
            fromBuf, toBuf = toBuf, fromBuf

    def _transpose(self, source, dest, layout_source, layout_dest):
        """
        TODO
        """
        # get axis information
        axis = self._get_swap_axes(layout_source, layout_dest)

        # get views of the important parts of the data
        sourceView = np.split(source, [layout_source.size])[
            0].reshape(layout_source.shape)

        # If both axes being swapped are not distributed
        if (len(axis) == 0):
            destView = np.split(dest, [layout_dest.size])[
                0].reshape(layout_dest.shape)

            assert(destView.base is dest)

            transposition = [layout_source.dims_order.index(
                i) for i in layout_dest.dims_order]
            destView[:] = np.transpose(sourceView, transposition)

            return

        # carry out transpose
        comm = self._subcomms[axis[0]]
        self._extract_from_source(
            sourceView, dest, layout_source, layout_dest, axis, comm)
        self._rearrange_from_buffer(
            dest, source, layout_source, layout_dest, axis, comm)

        f = np.split(dest, [layout_dest.size])[0].reshape(layout_dest.shape)

    def _transpose_source_intact(self, source, dest, buf, layout_source, layout_dest):
        """
        TODO
        """
        # get views of the important parts of the data
        sourceView = np.split(source, [layout_source.size])[
            0].reshape(layout_source.shape)

        # get axis information
        axis = self._get_swap_axes(layout_source, layout_dest)

        # If the axes being swapped are not distributed
        if (len(axis) == 0):
            dest = np.split(dest, [layout_dest  .size])[
                0].reshape(layout_dest  .shape)
            transposition = [layout_source.dims_order.index(
                i) for i in layout_dest.dims_order]
            dest[:] = np.transpose(sourceView, transposition)
            return

        # carry out transpose
        comm = self._subcomms[axis[0]]
        self._extract_from_source(
            sourceView, dest, layout_source, layout_dest, axis, comm)

        self._rearrange_from_buffer(
            dest, buf, layout_source, layout_dest, axis, comm)

    def _get_swap_axes(self, layout_source, layout_dest):
        """
        TODO
        """
        # Find the axes which will be swapped
        axis = []
        for i, n in enumerate(self._nprocsList):
            source_dim = layout_source.dims_order[i]
            dest_dim = layout_dest.dims_order[i]

            if (n > 1 and source_dim != dest_dim):
                axis.append(i)
                axis.append(layout_source.dims_order.index(dest_dim))
                axis.append(layout_dest.dims_order.index(source_dim))

        # axis[0] is the distributed axis in the source layout
        # axis[1] is the axis in the source layout which will
        #         be distributed in the destination layout
        # axis[2] is the axis in the destination layout which
        #         was distributed in the source layout

        return axis

    @staticmethod
    def _extract_from_source(source, tobuffer, layout_source: Layout,
                             layout_dest: Layout, axis: list, comm: MPI.Comm):
        """
        Take the information from the source and save it blockwise into
        the buffer. The saved blocks are also transposed
        """
        # get the number of processes that the data is split across
        nSplits = layout_source.nprocs[axis[0]]

        start = 0

        # Get the shape of the block
        shape = list(layout_source.shape)
        shape[axis[0]] = layout_source.max_block_shape[axis[0]]
        shape[axis[1]] = layout_dest.max_block_shape[axis[0]]
        size = np.prod(shape)

        ranges = [slice(x) for x in layout_source.shape]
        source_range = [slice(x) for x in layout_source.shape]

        # Find the order to which the axes will be transposed for sending
        # This is the same as before but the axis to be concatenated
        # must be the 0-th axis
        order = list(range(layout_source.ndims))

        if (axis[0] != 0):
            order[0], order[axis[0]] = order[axis[0]], order[0]
            shape[0], shape[axis[0]] = shape[axis[0]], shape[0]
            ranges[0], ranges[axis[0]] = ranges[axis[0]], ranges[0]

        for (split_length, mpi_start) in zip(layout_dest.mpi_lengths(axis[0]), layout_dest.mpi_starts(axis[0])):
            # Get a view on the buffer which is the same size as the block
            arr = tobuffer[start:start+size].reshape(shape)
            assert(arr.base is tobuffer)

            # Use the list of slices to access the relevant elements on the block
            ranges[axis[1]] = slice(split_length)
            arrView = arr[tuple(ranges)]
            assert(arrView.base is tobuffer)

            source_range[axis[1]] = slice(mpi_start, mpi_start+split_length)

            # Save the block into the buffer via the source
            # This may result in a copy, however this copy would only be
            # the size of the block
            # The data should however be written directly in the buffer
            # as the shapes agree
            arrView[:] = source[tuple(source_range)].transpose(order)

            start += size

    @staticmethod
    def _rearrange_from_buffer(data, buf, layout_source: Layout,
                               layout_dest: Layout, axis: list, comm: MPI.Comm):
        """
        Swap the axes of the blocks
        Pass the blocks to the correct processes, concatenating in the process
        Transpose the result to get the final layout in the destination
        """
        mpi_size = comm.Get_size()

        # Get the shape of the send block
        source_shape = list(layout_source.shape)
        source_shape[axis[1]] = layout_dest.max_block_shape[axis[0]]
        source_shape[axis[0]] = layout_source.max_block_shape[axis[0]]*mpi_size

        size = np.prod(source_shape)

        # Get a view on the data to be sent
        sendBuf = np.split(data, [size], axis=0)[0]
        assert(sendBuf.base is data)

        # Get a view where the data will be received
        rcvBuf = np.split(buf, [size], axis=0)[0]
        assert(rcvBuf.base is buf)

        comm.Alltoall(sendBuf, rcvBuf)

        source_order = list(layout_source.dims_order)

        # Reorder the shape to the current format
        if (axis[0] != 0):
            source_order[0], source_order[axis[0]
                                          ] = source_order[axis[0]], source_order[0]
            source_shape[0], source_shape[axis[0]
                                          ] = source_shape[axis[0]], source_shape[0]

        transposition = [source_order.index(i) for i in layout_dest.dims_order]

        # Get a view on the destination
        destView = np.split(data, [layout_dest.size])[
            0].reshape(layout_dest.shape)
        assert(destView.base is data)

        # Get a view on the actual data received
        bufView = np.split(buf, [size])[0].reshape(source_shape)
        assert(bufView.base is buf)

        if (layout_dest.shape[axis[2]] % mpi_size == 0 and layout_source.shape[axis[1]] % mpi_size == 0):
            # If all blocks are the same shape with no padding then the
            # transposition can be carried out directly
            destView[:] = np.transpose(bufView, transposition)

        else:
            for r in range(mpi_size):
                start = layout_source.max_block_shape[axis[0]]*r

                # Get a view on the block
                bufRanges = [slice(x) for x in source_shape]
                bufRanges[axis[1]] = slice(layout_dest.shape[axis[0]])
                bufRanges[0] = slice(
                    start, start+layout_source.mpi_lengths(axis[0])[r])

                assert(bufView[tuple(bufRanges)].base is buf)

                # Get a view on the block in the destination memory
                destRanges = [slice(x) for x in layout_dest.shape]
                destRanges[axis[2]] = slice(layout_source.mpi_starts(axis[0])[r],
                                            layout_source.mpi_starts(axis[0])[r]+layout_source.mpi_lengths(axis[0])[r])

                # Transpose the data. As the axes to be concatenated are the first dimension
                # the concatenation is done automatically
                destView[tuple(destRanges)] = np.transpose(
                    bufView[tuple(bufRanges)], transposition)

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

        # Collect a list of all distributed dimensions which do not
        # represent the same axis
        for i, n in enumerate(self._nprocsList):
            source_dim = l1.dims_order[i]
            dest_dim = l2.dims_order[i]
            if (n > 1 and source_dim != dest_dim):
                dims.append(i)

        # Endure that the distribution pattern is not changed on more
        # than one dimension
        return len(dims) < 2

# ===============================================================================


class LayoutSwapper(LayoutManager):
    """
    LayoutSwapper: Class containing information about the different layouts
    available. It handles conversion from one layout to another. Including
    conversions using different numbers of processors

    Parameters
    ----------
    comm : MPI.Comm
        The communicator on which the data will be distributed

    layouts : list of dict
        Each element of the list is associated with a different LayoutManager.


        For each element:

        The keys should be strings which will be used to identify layouts.

        The values should be an array_like containing the ordering of
        the dimensions in this layout.
        E.g. [0,2,1] means that the ordering is (eta1,eta3,eta2)
        The length of the values should be at least as long as the
        length of nprocs.

    nprocs : list of list of int
        Each element of the outer list should be a list containing the
        number of processes in each distribution direction.

        Each element of the list is associated with the corresponding
        element in the layouts list

    eta_grids : list of array_like
        The coordinates of the grid points in each dimension

    start : str
        The start layout

    Notes
    -----
    For one Layout Manager created with the command:

    : LayoutManager( comm, layouts, nprocs, eta_grids )

    the command should be

    : LayoutSwapper( comm, [layouts], [nprocs], eta_grids )

    """

    def __init__(self, comm: MPI.Comm, layouts: list, nprocs: list, eta_grids: list, start: str):
        self._comm = comm

        self._nLayoutManagers = len(layouts)
        if(self._nLayoutManagers != len(nprocs)):
            raise RuntimeError(
                'There must be an equal number of layout sets and distribution setups')

        self._nDims = [1 if isinstance(x, int) else max(
            len(x)-x.count(1), 1) for x in nprocs]

        self._maxDims = max([1 if isinstance(x, int) else len(x)
                            for x in nprocs])
        self._largestLayoutManager, x = max(
            enumerate(self._nDims), key=operator.itemgetter(1))

        self._totProcs = np.prod(nprocs[self._largestLayoutManager])

        # Get a list of the distribution pattern for each layout type
        self._nprocs = []
        for n in nprocs:

            assert(self._totProcs % np.prod(n) == 0)

            if (isinstance(n, int)):
                N = [n]
            else:
                N = list(n)
            for i in range(len(N), self._maxDims):
                N.append(1)

            self._nprocs.append(N)
        self._nprocs = tuple(self._nprocs)

        self._maxProcs = np.max(self._nprocs, axis=0)

        # Find the order of the layout types so it is clear which transposes
        # are valid (all communicators should be used in 1 of the handlers
        sortOrder = sorted(range(len(self._nDims)),
                           key=self._nDims.__getitem__, reverse=True)

        # Create the LayoutHandlers
        self._managers = [None for i in range(self._nLayoutManagers)]

        # Starting with the most distributed layout handler
        max_idx = sortOrder[0]

        # Create the communicators that will be used by all LayoutHandlers
        topology = comm.Create_cart(self._nprocs[max_idx], periods=[
                                    False]*len(self._nprocs[max_idx]))

        # Get communicator for each dimension
        subcomms = []
        for k in range(self._maxDims):
            subcomms.append(topology.Sub(
                [k == j for j in range(len(self._nprocs[max_idx]))]))

        # Find the rank on each communicator
        mpi_coords = topology.Get_coords(comm.Get_rank())

        # Create the LayoutHandler
        self._managers[max_idx] = LayoutHandler(
            subcomms, mpi_coords, layouts[max_idx], self._nprocs[max_idx], eta_grids)

        for idx in sortOrder[1:]:
            # Find the relevant subcommunicators and ranks
            availableSubcomms = self._nprocs[max_idx].copy()
            the_subcomms = []
            coords = []
            the_procs = []

            for i, n in enumerate(np.atleast_1d(nprocs[idx])):
                # For each distributed direction
                if (availableSubcomms.count(n) == 1):
                    # If there is only 1 possible communicator use that communicator
                    axis = availableSubcomms.index(n)

                else:
                    # If there are more than 1 possible communicators
                    # Find the possiblities
                    poss = np.nonzero(np.array(availableSubcomms) == n)[0]
                    res = []

                    for axis in poss:
                        # For each possibilty find how many direct transitions
                        # this enables
                        DimInPos = [d[axis]
                                    for n, d in layouts[max_idx].items()]
                        DimToPos = [d[i] for n, d in layouts[idx].items()]
                        nPossTrans = sum([DimInPos.count(d) for d in DimToPos])

                        # If there is at least one possiblity save the result
                        if (nPossTrans > 0):
                            res.append((axis, nPossTrans))

                    # Choose the communicator which allows the most direct
                    # transitions
                    assert(len(res) > 0)
                    axis = max(res, key=lambda x: x[1])[0]

                # Save the communicator used and stop it being accidentally
                # reused for a second dimension
                the_subcomms.append(subcomms[axis])
                coords.append(mpi_coords[axis])
                availableSubcomms[axis] = None
                the_procs.append(n)

            # Create the LayoutHandler
            self._managers[idx] = LayoutHandler(
                the_subcomms, coords, layouts[idx], the_procs, eta_grids)

        # Find the largest buffer required to ensure that enough memory
        # is allocated
        buffSize = [x.bufferSize for x in self._managers]
        self._buffer_size = max(buffSize)

        # Create a dictionary to link layouts to their Handlers
        self._handlers = dict()
        for i, h in enumerate(layouts):
            for n, l in h.items():
                self._handlers[n] = i

        # Calculate direct layout connections
        self._layouts = list(self._handlers.keys())
        myMap = []

        for n, name1 in enumerate(self._layouts):
            myMap.append((name1, []))

            for i, name2 in enumerate(self._layouts[:n]):
                if (not self._compatibleLayout(name1, name2)):
                    continue

                # If the layouts are compatible then save the direct
                # transition data to the map
                myMap[i][1].append(name1)
                myMap[n][1].append(name2)

                # Get the associated LayoutHandlers
                h1 = self._managers[self._handlers[name1]]
                h2 = self._managers[self._handlers[name2]]

                # If they are the same then the maximum block is dealt
                # with by the Handler
                if (h1 == h2):
                    continue

                # Find the layouts
                l1 = h1.getLayout(name1)
                l2 = h2.getLayout(name2)

                # Find the number of distributed directions to order l1
                # and l2 correctly for the getAxes function
                n1 = h1.nDistributedDirections
                n2 = h2.nDistributedDirections
                if (n1 == n2):
                    # If the number of distribution directions does not
                    # change then this is secretly just a transpose
                    continue
                elif (n1 > n2):
                    idx_2, idx_1 = self.getAxes(l2, l1)
                else:
                    idx_1, idx_2 = self.getAxes(l1, l2)

                # Find the shape of each block
                blockShape1 = list(l1.shape)
                blockShape1[idx_1] = l1.max_block_shape[idx_1]
                blockSize1 = np.prod(blockShape1)

                blockShape2 = list(l2.shape)
                blockShape2[idx_2] = l2.max_block_shape[idx_2]
                blockSize2 = np.prod(blockShape2)

                # Ensure that there is enough memory for this gather operation
                if (blockSize1 > blockSize2):
                    comm = h2.communicators[idx_2]
                    mpi_size = comm.Get_size()

                    self._buffer_size = max(
                        self._buffer_size, blockSize2*mpi_size)
                else:
                    comm = h1.communicators[idx_1]
                    mpi_size = comm.Get_size()

                    self._buffer_size = max(
                        self._buffer_size, blockSize1*mpi_size)

                # Find the distribution patterns
                nprocs1 = l1.nprocs
                ndims1 = self._nprocs[self._handlers[name1]]
                nprocs2 = l2.nprocs
                ndims2 = self._nprocs[self._handlers[name2]]

        # Save the connections in a dictionary
        DirectConnections = dict(myMap)

        # Save all layout paths in a map and save any remaining unconnected layouts
        full = self._makeConnectionMap(DirectConnections)

        # all layouts should have been found
        if (not full):
            raise RuntimeError("Not all layouts could not be connected")

        # Remember the current LayoutHandler to help the properties
        self._current_manager = self._managers[self._handlers[start]]

        # Save the names and shapes of possible layouts
        self._shapes = [
            l for manager in self._managers for l in manager.availableLayouts]

    def _compatibleLayout(self, layout1: str, layout2: str):
        """ Tests whether 2 layouts are compatible by testing whether
            they are on the same LayoutHandler or whether the dimensions
            are ordered in the same way and the LayoutHandlers are compatible
        """
        i1 = self._handlers[layout1]
        i2 = self._handlers[layout2]

        if (i1 == i2):
            # If the LayoutHandler is the same then the Handler knows
            # whether the layouts are compatible
            l1 = self._managers[i1].getLayout(layout1)
            l2 = self._managers[i1].getLayout(layout2)

            return self._managers[i1].compatible(l1, l2)

        else:
            # If different LayoutHandlers are used:

            # Get the handler
            handler1 = self._managers[i1]
            handler2 = self._managers[i2]

            # Get the number of distributed directions
            nDim1 = len(handler1.nProcs)
            nDim2 = len(handler2.nProcs)

            # Only 1 distributed direction should change
            if (abs(nDim1-nDim2) > 1):
                return False

            # If the distribution is the same then the communicators should
            # also be the same
            if (nDim1 == nDim2):
                return all([c in handler1.communicators for c in handler2.communicators])

            # Ensure that 2 is the larger handler to facilitate steps
            if (nDim1 > nDim2):
                handler1, handler2 = handler2, handler1
                nDim1, nDim2 = nDim2, nDim1
                layout1, layout2 = layout2, layout1

            l1 = handler1.getLayout(layout1)
            l2 = handler2.getLayout(layout2)

            dims1 = l1.dims_order
            dims2 = l2.dims_order

            # Find the axis which will be distributed
            possComms = list(handler2.communicators)
            for i, c in enumerate(handler1.communicators):
                if c in possComms:
                    j = possComms.index(c)
                    if (dims1[i] == dims2[j]):
                        possComms[j] = None

            idx_s = np.nonzero(np.array(possComms) != None)[0]

            # Ensure that there is only 1 possible axis which is modified
            return len(idx_s) == 1

    def getLayout(self, name: str):
        """ Return a requested layout
        """
        return self._managers[self._handlers[name]].getLayout(name)

    @property
    def nProcs(self):
        """ The number of processes available on the distributed dimensions
        """
        return self._current_manager.nProcs

    @property
    def mpiCoords(self):
        """ The coordinates of the current processor on the MPI grid communicator
        """
        return self._current_manager.mpiCoords

    @property
    def nDistributedDirections(self):
        """ The number of distributed directions
        """
        return self._current_manager.nDistributedDirections

    def transpose(self, source, dest, source_name, dest_name, buf=None):
        """
        Change from one layout to another, if a buffer is provided then
        this will leave the source layout intact. Otherwise the source
        will be used as the buffer

        Parameters
        ----------
        source : array_like
            The entire memory block where the data is currently stored
            (including unused cells)

        dest : array_like
            The entire memory block where the data will be stored
            (including cells which will not be used)

        source_name : str
            The name of the source layout

        dest_name : str
            The name of the destination layout

        buf : array_like
            A memory block large enough to store all data. If this
            parameter is not provided then source will not be kept intact

        Notes
        -----
        source and dest are assumed to not overlap in memory

        """
        # If this thread is only here for plotting purposes then ignore the command
        if (self._buffer_size == 0):
            return

        # Verify that the input makes sense
        assert source_name in self._layouts
        assert dest_name in self._layouts
        assert source.size == self._buffer_size
        assert dest  .size == self._buffer_size

        if (self._handlers[source_name] == self._handlers[dest_name]):
            # If the source and destination are on the same LayoutHandler
            # then let it handle everything
            self._managers[self._handlers[source_name]].transpose(
                source, dest, source_name, dest_name, buf)
            self._current_manager = self._managers[self._handlers[dest_name]]
            return

        # Check that a direct path is available
        if (len(self._route_map[source_name][dest_name]) == 1):
            # if so then carry out the transpose
            layout_source = self.getLayout(source_name)
            layout_dest = self.getLayout(dest_name)
            if (buf is None):
                self._transpose(source, dest, layout_source, layout_dest)
            else:
                self._transpose_source_intact(
                    source, dest, buf, layout_source, layout_dest)
        else:
            # if not reroute the transpose via intermediate steps
            if (buf is None):
                self._transposeRedirect(source, dest, source_name, dest_name)
            else:
                self._transposeRedirect_source_intact(
                    source, dest, buf, source_name, dest_name)
        self._current_manager = self._managers[self._handlers[dest_name]]

    def _transpose(self, source, dest, layout_source: Layout, layout_dest: Layout):
        """
        TODO
        """
        # Find the distribution patterns
        source_nprocs = list(layout_source.nprocs)
        source_ndims = self._managers[self._handlers[layout_source.name]
                                      ].nDistributedDirections
        dest_ndims = self._managers[self._handlers[layout_dest.name]
                                    ].nDistributedDirections

        if (dest_ndims == source_ndims):
            # If the source has the same number of dimensions then the
            # distribution is not changed and it is a simple transpose
            sourceView = np.split(source, [layout_source.size])[
                0].reshape(layout_source.shape)
            destView = np.split(dest, [layout_dest.size])[
                0].reshape(layout_dest.shape)

            transposition = [layout_source.dims_order.index(
                i) for i in layout_dest.dims_order]

            # Copy the relevant information
            destView[:] = np.transpose(sourceView, transposition)

        elif (dest_ndims > source_ndims):
            # If the source has fewer dimensions then the layout was not
            # distributed and all necessary information is already on the thread
            sourceView = np.split(source, [layout_source.size])[
                0].reshape(layout_source.shape)
            destView = np.split(dest, [layout_dest.size])[
                0].reshape(layout_dest.shape)

            # Find the axis which will be distributed
            idx_s, idx_d = self.getAxes(layout_source, layout_dest)

            # Get the information about the position in the communicators
            comm = self._managers[self._handlers[layout_dest.name]
                                  ].communicators[idx_d]
            rank = comm.Get_rank()

            # Find the start and end of the data
            start = layout_dest.mpi_starts(idx_d)[rank]
            length = layout_dest.mpi_lengths(idx_d)[rank]

            sourceSlice = [slice(n) for n in layout_source.shape]
            sourceSlice[idx_s] = slice(start, start+length)

            transposition = [layout_source.dims_order.index(
                i) for i in layout_dest.dims_order]

            # Copy the relevant information
            destView[:] = np.transpose(
                sourceView[tuple(sourceSlice)], transposition)

        else:
            # Find the axis which will be distributed
            idx_d, idx_s = self.getAxes(layout_dest, layout_source)

            # Get the information about the communicators
            comm = self._managers[self._handlers[layout_source.name]
                                  ].communicators[idx_s]
            mpi_size = comm.Get_size()

            # Find the size of the distributed block
            blockShape = list(layout_source.shape)
            blockShape[idx_s] = layout_source.max_block_shape[idx_s]
            blockSize = np.prod(blockShape)

            # Get a view on the block
            sourceView = np.split(source, [blockSize])[0]

            # Gather the data from the blocks on the processes
            destView = np.split(dest, [blockSize*mpi_size])[0]
            comm.Allgather((sourceView, MPI.DOUBLE),
                           (destView, MPI.DOUBLE))

            # Get a view on the received blocks
            blocks = np.split(dest, blockSize*np.arange(1, mpi_size+1))
            # Get a view on the result
            destView = np.split(source, [layout_dest.size])[
                0].reshape(layout_dest.shape)

            # Use slices to access the relevant data
            slices = [slice(x) for x in layout_dest.shape]

            transposition = [layout_source.dims_order.index(
                i) for i in layout_dest.dims_order]

            for i, b in enumerate(blocks[:-1]):
                # Find the original block shape
                blockShape = list(layout_source.shape)
                blockShape[idx_s] = layout_source.mpi_lengths(idx_s)[i]
                blockSize = np.prod(blockShape)

                # Find the relevant data
                slices[idx_d] = slice(layout_source.mpi_starts(idx_s)[i],
                                      layout_source.mpi_starts(idx_s)[i]+layout_source.mpi_lengths(idx_s)[i])

                block = np.split(b, [blockSize])[0].reshape(blockShape)

                # Copy the block into the correct part of the memory
                destView[tuple(slices)] = np.transpose(block, transposition)

            # The data now resides on the wrong memory chunk and must be copied
            dest[:] = source[:]

    def _transpose_source_intact(self, source, dest, buf, layout_source: Layout, layout_dest: Layout):
        """
        TODO
        """
        # Find the distribution patterns
        source_nprocs = list(layout_source.nprocs)
        source_ndims = self._managers[self._handlers[layout_source.name]
                                      ].nDistributedDirections
        dest_ndims = self._managers[self._handlers[layout_dest.name]
                                    ].nDistributedDirections

        if (dest_ndims == source_ndims):
            # If the source has the same number of dimensions then the
            # distribution is not changed and it is a simple transpose
            sourceView = np.split(source, [layout_source.size])[
                0].reshape(layout_source.shape)
            destView = np.split(dest, [layout_dest.size])[
                0].reshape(layout_dest.shape)

            transposition = [layout_source.dims_order.index(
                i) for i in layout_dest.dims_order]

            # Copy the relevant information
            destView[:] = np.transpose(sourceView, transposition)

        elif (dest_ndims > source_ndims):
            # If the source has fewer dimensions then the layout was not
            # distributed and all necessary information is already on the thread
            sourceView = np.split(source, [layout_source.size])[
                0].reshape(layout_source.shape)
            destView = np.split(dest, [layout_dest.size])[
                0].reshape(layout_dest.shape)

            # Find the axis which will be distributed
            idx_s, idx_d = self.getAxes(layout_source, layout_dest)

            # Get the information about the position in the communicators
            comm = self._managers[self._handlers[layout_dest.name]
                                  ].communicators[idx_d]
            rank = comm.Get_rank()

            # Find the start and end of the data
            start = layout_dest.mpi_starts(idx_d)[rank]
            length = layout_dest.mpi_lengths(idx_d)[rank]

            sourceSlice = [slice(n) for n in layout_source.shape]
            sourceSlice[idx_s] = slice(start, start+length)

            transposition = [layout_source.dims_order.index(
                i) for i in layout_dest.dims_order]

            # Copy the relevant information
            destView[:] = np.transpose(
                sourceView[tuple(sourceSlice)], transposition)
        else:
            # Find the axis which will be distributed
            idx_d, idx_s = self.getAxes(layout_dest, layout_source)

            # Get the information about the communicators
            comm = self._managers[self._handlers[layout_source.name]
                                  ].communicators[idx_s]
            mpi_size = comm.Get_size()

            # Find the size of the distributed block
            blockShape = list(layout_source.shape)
            blockShape[idx_s] = layout_source.max_block_shape[idx_s]
            blockSize = np.prod(blockShape)

            # Get a view on the block
            sourceView = np.split(source, [blockSize])[0]

            # Gather the data from the blocks on the processes
            destView = np.split(buf, [blockSize*mpi_size])[0]
            comm.Allgather((sourceView, MPI.DOUBLE),
                           (destView, MPI.DOUBLE))

            # Get a view on the received blocks
            blocks = np.split(buf, blockSize*np.arange(1, mpi_size+1))
            # Get a view on the result
            destView = np.split(dest, [layout_dest.size])[
                0].reshape(layout_dest.shape)

            # Use slices to access the relevant data
            slices = [slice(x) for x in layout_dest.shape]

            transposition = [layout_source.dims_order.index(
                i) for i in layout_dest.dims_order]

            for i, b in enumerate(blocks[:-1]):
                # Find the original block shape
                blockShape = list(layout_source.shape)
                blockShape[idx_s] = layout_source.mpi_lengths(idx_s)[i]
                blockSize = np.prod(blockShape)

                # Find the relevant data
                slices[idx_d] = slice(layout_source.mpi_starts(idx_s)[i],
                                      layout_source.mpi_starts(idx_s)[i]+layout_source.mpi_lengths(idx_s)[i])

                block = np.split(b, [blockSize])[0].reshape(blockShape)

                # Copy the block into the correct part of the memory
                destView[tuple(slices)] = np.transpose(block, transposition)

    def _transposeRedirect(self, source, dest, source_name, dest_name):
        """
        Function for changing layout via multiple steps leaving the
        source layout intact.
        """

        # Get route from one layout to another
        steps = self._route_map[source_name][dest_name]
        nSteps = len(steps)

        # warn about multiple steps
        warnings.warn("Changing from {0} layout to {1} layout requires {2} steps"
                      .format(source_name, dest_name, nSteps))

        # carry out the steps one by one
        nowLayoutKey = source_name

        fromBuf = source
        toBuf = dest

        for i in range(nSteps):
            nextLayoutKey = steps[i]
            self.transpose(fromBuf, toBuf, nowLayoutKey, nextLayoutKey)
            nowLayoutKey = nextLayoutKey
            fromBuf, toBuf = toBuf, fromBuf

        # Ensure the result is found in the expected place
        if (nSteps % 2 == 0):
            dest[:] = source

    def _transposeRedirect_source_intact(self, source, dest, buf, source_name, dest_name):
        """
        Function for changing layout via multiple steps.
        """
        # Get route from one layout to another
        steps = self._route_map[source_name][dest_name]
        nSteps = len(steps)

        # warn about multiple steps
        warnings.warn("Changing from {0} layout to {1} layout requires {2} steps"
                      .format(source_name, dest_name, nSteps))

        # take the first step to move the data
        nowLayoutKey = steps[0]

        # The first step needs to place the data in such a way that the
        # final result will be stored in the destination
        if (nSteps % 2 == 0):
            self.transpose(source, buf, source_name, nowLayoutKey, dest)
            fromBuf = buf
            toBuf = dest
        else:
            self.transpose(source, dest, source_name, nowLayoutKey, buf)
            fromBuf = dest
            toBuf = buf

        # take the remaining steps
        for i in range(1, nSteps):
            nextLayoutKey = steps[i]

            self.transpose(fromBuf, toBuf, nowLayoutKey, nextLayoutKey)

            nowLayoutKey = nextLayoutKey
            fromBuf, toBuf = toBuf, fromBuf

    def getAxes(self, layout_gathered: Layout, layout_scattered: Layout):
        """
        TODO
        """
        # Get handlers
        handlerG = self._managers[self._handlers[layout_gathered.name]]
        handlerS = self._managers[self._handlers[layout_scattered.name]]

        # Find the position of the communicator which appears in the
        # scattered layout but not in the gathered layout
        possComms = list(handlerS.communicators)
        for c in handlerG.communicators:
            if c in possComms:
                i = possComms.index(c)
                possComms[i] = None

        idx_s = np.nonzero(np.array(possComms) != None)[0][0]

        # Find which axis in the gathered layout has the same dimension
        idx_g = layout_gathered.dims_order.index(
            layout_scattered.dims_order[idx_s])
        return (idx_g, idx_s)
