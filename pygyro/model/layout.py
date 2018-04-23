from mpi4py import MPI
import numpy as np
import warnings

class Layout:

    def __init__( self, name:str, nprocs:list, dims_order:list, eta_grids:list, myRank: list ):
        self._name = name
        self._ndims      = len( dims_order )
        self._dims_order = dims_order
        
        # check input makes sense
        assert len( dims_order ) == len( eta_grids )
        assert len( nprocs ) == len( myRank )
        
        # get number of processes in each dimension (sorted [eta1,eta2,eta3,eta4])
        self._nprocs = [1]*self._ndims
        myRanks = [0]*self._ndims
        for j,n in enumerate(nprocs):
            self._nprocs[j]=n
            myRanks[j]=myRank[j]
        
        # initialise values used for accessing data (sorted [eta1,eta2,eta3,eta4])
        self._mpi_starts = []
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
            self._mpi_starts.append(starts)
            # append end index
            starts=np.append(starts,n)
            self._mpi_lengths.append(starts[1:]-starts[:-1])
            
            # save start and indices from list using cartesian ranks
            self._starts[i] = starts[myRanks[i]]
            self._ends[i]   = starts[myRanks[i]+1]
            self._shape[i]  = self._ends[i]-self._starts[i]
    
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
    def nprocs( self ):
        """ Number of processes along each dimension.
        """
        return self._nprocs
    
    def mpi_starts( self , i: int ):
        """ Get global starting points for dimension i on all processes
        """
        return self._mpi_starts[i]
    
    def mpi_lengths( self , i: int ):
        """ Get length of dimension i on all processes
        """
        return self._mpi_lengths[i]

#===============================================================================

class LayoutManager:

    def __init__( self, comm : MPI.Comm, layouts : dict, nprocs: list, eta_grids: list ):
        self.rank=comm.Get_rank()
        
        nDims=len(nprocs)
        self._nprocs = nprocs
        
        topology = comm.Create_cart( nprocs, periods=[False]*nDims,
                reorder=False )
        
        # Get communicator for each dimension
        self._subcomms = []
        for i in range(nDims):
            self._subcomms.append(topology.Sub( [i==j for j in range(nDims)] ))
        
        mpi_coords = topology.Get_coords(comm.Get_rank())
        
        # Create the layouts and save them in a dictionary
        layoutObjects = []
        for name,dim_order in layouts.items():
            layoutObjects.append((name,Layout(name,nprocs,dim_order,eta_grids,mpi_coords)))
        self._layouts = dict(layoutObjects)
        self.nLayouts=len(self._layouts)
        
        # Calculate layout connections
        myMap = []
        for n,(name1,l1) in enumerate(layoutObjects):
            myMap.append((name1,[]))
            for i,(name2,l2) in enumerate(layoutObjects[:n]):
                if (self.compatible(l1,l2)):
                    myMap[i][1].append(name1)
                    myMap[n][1].append(name2)
        # Save the connections in a dictionary
        self._map=dict(myMap)
        
        # Verify compatibility of layouts
        unvisited = set(self._layouts.keys())
        toCheck=[layoutObjects[0][0]]
        while (len(toCheck)>0):
            toCheckNext=[]
            for a_layout in toCheck:
                if (a_layout in unvisited):
                    toCheckNext.extend(self._map[a_layout])
                    unvisited.remove(a_layout)
            toCheck=toCheckNext
        
        # all keys should have been found
        if (unvisited!=set()):
            s=str()
            for name in unvisited:
                s+=(" '"+name+"'")
            raise RuntimeError("The following layouts could not be connected to preceeding layouts :"+s)
    
    def getLayout( self, name):
        return self._layouts[name]
    
    def transpose( self, source, dest, source_name, dest_name ):
        """ Function for changing layout
        """
        
        # Verify that the input makes sense
        assert source_name in self._layouts
        assert dest_name   in self._layouts
        assert all(source.shape == self._layouts[source_name].shape)
        
        # Check that a direct path is available
        if (dest_name in self._map[source_name]):
            # if so then carry out the transpose
            self._transpose(source,dest,self._layouts[source_name],self._layouts[dest_name])
        else:
            # if not reroute the transpose via intermediate steps
            self.transposeRedirect(source,dest,source_name,dest_name)
    
    def transposeRedirect(self,source,dest,source_name,dest_name):
        """
        Function for changing layout via multiple steps.
        
        This function is not optimal.
        It allocates memory as it does not have access to the data grids
        allocated in Grid.
        
        This function will ideally never be used. If operations are
        carried out in the correct order, it will not be.
        """
        
        steps = []
        unvisited = set(self._layouts.keys())
        unvisited.remove(dest_name)
        
        toAppend=[]
        for nextStep in self._map[dest_name]:
            if (nextStep in unvisited):
                unvisited.remove(nextStep)
                toAppend.append((nextStep,dest_name))
        steps.append(dict(toAppend))
        
        while (source_name in unvisited):
            toAppend=[]
            for mapKey in steps[-1]:
                for nextStep in self._map[mapKey]:
                    if (nextStep in unvisited):
                        unvisited.remove(nextStep)
                        toAppend.append((nextStep,mapKey))
            steps.append(dict(toAppend))
        
        nowLayoutKey=source_name
        nowLayout=self._layouts[source_name]
        nSteps=0
        f_intermediate1=source
        while(nowLayoutKey!=dest_name):
            nextLayoutKey=steps.pop()[nowLayoutKey]
            nextLayout=self._layouts[nextLayoutKey]
            f_intermediate2=np.empty(nextLayout.shape)
            self._transpose(f_intermediate1,f_intermediate2,nowLayout,nextLayout)
            f_intermediate1=f_intermediate2
            nowLayout=nextLayout
            nowLayoutKey=nextLayoutKey
            nSteps+=1
        dest[:]=f_intermediate2
        
        warnings.warn("Changing from %s layout to %s layout required %i steps" %
                (source_name,dest_name,nSteps))
    
    def _transpose(self, source, dest, layout_source, layout_dest):
        assert all(  dest.shape == layout_dest.shape)
        
        # Find the axes which will be swapped
        axis = []
        for i,n in enumerate(layout_source.dims_order):
            if (n!=layout_dest.dims_order[i]):
                axis.append(i)
        # axis[0] is the distributed axis in the source layout
        # axis[1] is the distributed axis in the destination layout
        
        # get the number of processes that the data is split across
        nSplits=layout_source.nprocs[axis[0]]
        
        comm = self._subcomms[axis[0]]
        size = comm.Get_size()
        rank = comm.Get_rank()
        
        # Get the sizes of the sent and received blocks.
        # This is equal to the product of the number of points in each
        # dimension except the dimension that was/will be distributed
        # multiplied by the number of points in the distributed direction
        # (e.g. when going from n_eta1=10, n_eta2=10, n_eta3=20, n_eta4=10
        # to n_eta1=20, n_eta2=10, n_eta3=10, n_eta4=10
        # the sent sizes are (10*10*[10 10]*10) and the received sizes
        # are ([10 10]*10*10*10)
        sourceSizes = np.prod( layout_source.shape ) *          \
                      layout_dest  .mpi_lengths( axis[0] ) //   \
                      layout_source.shape[axis[1]]
        
        destSizes   = np.prod( layout_dest  .shape ) *          \
                      layout_source.mpi_lengths( axis[0] ) //   \
                      layout_dest  .shape[axis[1]]
        
        # get the start points in the array
        sourceStarts     = np.zeros(size,int)
        destStarts       = np.zeros(size,int)
        sourceStarts[1:] = np.cumsum(sourceSizes)[:-1]
        destStarts  [1:] = np.cumsum(  destSizes)[:-1]
        
        # check the sizes have been computed correctly
        assert(sum(sourceSizes)==source.size)
        assert(sum(destSizes)==dest.size)
        
        # find a non-shaped place to store the received data
        newData = dest.reshape(dest.size,1)
        
        # Data is split and sent to the appropriate processes
        # Spitting using Alltoallv and a transpose only works when the dimension to split
        # is the last dimension
        # reshape is used instead of flatten as flatten creates a copy but reshape doesn't
        comm.Alltoallv( ( source.transpose().reshape(source.size,1) ,
                          ( sourceSizes, sourceStarts )             ,
                          MPI.DOUBLE                                ) ,
                        
                        ( newData                                   , 
                          ( destSizes  , destStarts   )             ,
                          MPI.DOUBLE                                ) )
        
        # For the data to be correctly reconstructed it is first split
        # back into its constituent parts
        newData=np.split(newData,destStarts[1:])
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
            assert(np.prod(shape)==destSizes[i])
            # The data is reshaped to the trasmitted shape
            arr=np.reshape(arr,np.flip(shape,0))
            # The array is then rearranged to the used ordering by
            # once more transposing to return the ordering to its original
            # state then swapping the axes that were/are distributed
            newData[i]=np.swapaxes(arr.transpose(),axis[0],axis[1])
        
        # Stitch the pieces together in the correct order.
        # This is done along the non-distributed axis which, after the
        # axes swap is the second axis
        # This will result in a new array being mallocked,
        # however this is necessary to ensure contiguous memory
        dest[:]=np.concatenate(newData,axis=axis[1])
    
    
    
    def compatible(self, l1: Layout, l2: Layout):
        dims = []
        # check the ordering of the dimensions
        for i,o in enumerate(l1.dims_order):
            if (o!=l2.dims_order[i]):
                # Save dimensions where the ordering differs
                dims.append(o)
                dims.append(l2.dims_order[i])
                # if these dimensions are both distributed then they cannot be swapped
                if (l1.nprocs[o]>1 and l1.nprocs[l2.dims_order[i]]>1):
                    return False
        # There should only be 1 difference between compatible layouts
        # e.g. if l1.dims_order=[a,b,c,d] l2.dims_order can be
        # [a,b,d,c], [a,c,b,d], [c,b,a,d], [b,a,c,d] etc.
        # This means if a,b are the swapped values that dim should contain [a,b,b,a]
        return (len(dims)==4 and dims[0]==dims[3] and dims[1]==dims[2])
