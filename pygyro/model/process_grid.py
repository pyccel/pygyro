

def compute_2d_process_grid( npts : list, mpi_size : int ):
    """ Compute 2D grid of processes for the two distributed dimensions
        in each layout.
    """
    
    max_proc1 = min( npts[0], npts[3] )
    max_proc2 = min( npts[2], npts[3] )
    
    return compute_2d_process_grid_from_max(max_proc1,max_proc2,mpi_size)

def compute_2d_process_grid_from_max( max_proc1 : int, max_proc2: int, mpi_size : int ):
    nprocs1 = 1
    nprocs2 = mpi_size
    
    while (nprocs2>max_proc2):
        nprocs1+=1
        while(nprocs1<=mpi_size and mpi_size%nprocs1!=0):
            nprocs1+=1
        nprocs2=mpi_size//nprocs1
        if (nprocs1==mpi_size+1):
            raise RuntimeError("There is no valid combination of processors for this grid")
    
    divisions1 = max_proc1/nprocs1
    divisions2 = max_proc2/nprocs2
    ratio = max(divisions1,divisions2)/min(divisions1,divisions2)
    
    i=0
    while (True):
        i+=1
        new_n1=nprocs1+1
        while(new_n1<mpi_size and mpi_size%new_n1!=0):
            new_n1+=1
        if (new_n1>mpi_size):
            break
        new_n2=mpi_size//new_n1
        if (new_n2<=max_proc2):
            divisions1 = max_proc1/new_n1
            divisions2 = max_proc2/new_n2
            new_ratio = max(divisions1,divisions2)/min(divisions1,divisions2)
            if (new_ratio<ratio):
                nprocs1=new_n1
                nprocs2=new_n2
                ratio=new_ratio
            else:
                break
    
    #print(ratio)

    return nprocs1, nprocs2
