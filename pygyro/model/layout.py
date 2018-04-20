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
        
        #TODO: ORDERING!!
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
        
        self._layouts = []
        self._layout_names = []
        for name,dim_order in layouts.items():
            self._layouts.append(Layout(name,nprocs,dim_order,eta_grids,mpi_coords))
            self._layout_names.append(name)
        self.nLayouts=len(self._layouts)
        
        # Calculate layout connections
        self._map = []
        for l in self._layouts:
            n=len(self._map)
            self._map.append([])
            for i in range(0,n):
                if (self.compatible(l,self._layouts[i])):
                    self._map[i].append(n)
                    self._map[n].append(i)
        
        # Verify compatibility of layouts
        found = [False]*self.nLayouts
        i=0
        found[0]=True
        toCheck=self._map[0]
        while (len(toCheck)>0):
            toCheckNext=[]
            for i in toCheck:
                if (found[i]==False):
                    toCheckNext.extend(self._map[i])
                    found[i]=True
            toCheck=toCheckNext
        
        if (not all(found)):
            s=str()
            for i,b in enumerate(found):
                if (not b):
                    s+=(" '"+self._layouts[i].name+"'")
            raise RuntimeError("The following layouts could not be connected to preceeding layouts :"+s)
        
        # TODO
        # - store data in object
    
    def getLayout( self, name):
        return self._layouts[self._layout_names.index(name)]
    
    def transpose( self, source, dest, layout_source, dest_name ):
        
        # verify that the input makes sense
        assert all(source.shape == layout_source.shape)
        assert layout_source in self._layouts
        assert dest_name     in self._layout_names
        
        sourceIdx = self._layouts.index(layout_source)
        destIdx = self._layout_names.index(dest_name)
        
        # Check that path is available
        if (destIdx in self._map[sourceIdx]):
            self._transpose(source,dest,layout_source,self._layouts[destIdx])
        else:
            self.transposeRedirect(source,dest,sourceIdx,destIdx)
    
    def transposeRedirect(self,source,dest,sourceIdx,destIdx):
        
        steps = []
        unvisited = [True]*self.nLayouts
        unvisited[destIdx]=False
        
        toAppend=[]
        for k in self._map[destIdx]:
            if (unvisited[k]):
                unvisited[k]=False
                toAppend.append((k,destIdx))
        steps.append(dict(toAppend))
        
        while (unvisited[sourceIdx]):
            toAppend=[]
            for mapIdx in steps[-1]:
                for k in self._map[mapIdx]:
                    if (unvisited[k]):
                        unvisited[k]=False
                        toAppend.append((k,mapIdx))
            steps.append(dict(toAppend))
        
        nowLayoutIdx=sourceIdx
        nowLayout=self._layouts[sourceIdx]
        nSteps=0
        f_intermediate1=source
        while(nowLayoutIdx!=destIdx):
            nextLayoutIdx=steps.pop()[nowLayoutIdx]
            nextLayout=self._layouts[nextLayoutIdx]
            f_intermediate2=np.empty(nextLayout.shape)
            self._transpose(f_intermediate1,f_intermediate2,nowLayout,nextLayout)
            f_intermediate1=f_intermediate2
            nowLayout=nextLayout
            nowLayoutIdx=nextLayoutIdx
            nSteps+=1
        dest[:]=f_intermediate2
        
        warnings.warn("Changing from %s layout to %s layout required %i steps" %
                (self._layouts[sourceIdx].name,self._layouts[destIdx].name,nSteps))
    
    def _transpose(self, source, dest, layout_source, layout_dest):
        assert all(  dest.shape == layout_dest.shape)
        
        axis = []
        for i,n in enumerate(layout_source.dims_order):
            if (n!=layout_dest.dims_order[i]):
                axis.append(i)
        
        nSplits=layout_source.nprocs[axis[0]]
        
        comm = self._subcomms[axis[0]]
        size = comm.Get_size()
        rank = comm.Get_rank()
        
        sourceSizes = np.prod( layout_source.shape ) *          \
                      layout_dest  .mpi_lengths( axis[0] ) //   \
                      layout_source.shape[axis[1]]
        
        destSizes   = np.prod( layout_dest  .shape ) *          \
                      layout_source.mpi_lengths( axis[0] ) //   \
                      layout_dest  .shape[axis[1]]
        
        sourceStarts     = np.zeros(size,int)
        destStarts       = np.zeros(size,int)
        sourceStarts[1:] = np.cumsum(sourceSizes)[:-1]
        destStarts  [1:] = np.cumsum(  destSizes)[:-1]
        
        assert(sum(sourceSizes)==source.size)
        assert(sum(destSizes)==dest.size)
        
        newData = dest.reshape(dest.size,1)
        
        MPI.COMM_WORLD.Barrier()
        # Spitting using Alltoallv and a transpose only works when the dimension to split
        # is the last dimension
        
        # reshape is used instead of flatten as flatten creates a copy but reshape doesn't
        comm.Alltoallv( ( source.transpose().reshape(source.size,1) ,
                          ( sourceSizes, sourceStarts )             ,
                          MPI.DOUBLE                                ) ,
                        
                        ( newData                                   , 
                          ( destSizes  , destStarts   )             ,
                          MPI.DOUBLE                                ) )
        
        newData=np.split(newData,destStarts[1:])
        lastDim=len(layout_source.shape)-1
        for i,arr in enumerate(newData):
            shape=layout_source.shape.copy()
            shape[axis[0]]=layout_source.mpi_lengths(axis[0])[i]
            shape[axis[1]]=layout_dest.mpi_lengths(axis[0])[rank]
            assert(np.prod(shape)==destSizes[i])
            arr=np.reshape(arr,np.flip(shape,0))
            newData[i]=np.swapaxes(arr.transpose(),axis[0],axis[1])
        
        dest[:]=np.concatenate(newData,axis=axis[1])
    
    def compatible(self, l1: Layout, l2: Layout):
        if (l1.nprocs!=l2.nprocs):
            return False
        else:
            dims = []
            for i,o in enumerate(l1.dims_order):
                if (o!=l2.dims_order[i]):
                    dims.append(o)
                    dims.append(l2.dims_order[i])
                    if (l1.nprocs[o]>1 and l1.nprocs[l2.dims_order[i]]>1):
                        return False
            return (len(dims)==4 and dims[0]==dims[3] and dims[1]==dims[2])
