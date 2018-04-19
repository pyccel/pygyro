

def compute_2d_process_grid( npts : list, mpi_size : int ):
    """ Compute 2D grid of processes for the two distributed dimensions
        in each layout.
    """
    
    # Get maximum values for number of processes in each distributed dimension
    max_proc1 = min( npts[0], npts[3] )
    max_proc2 = min( npts[2], npts[3] )
    
    return compute_2d_process_grid_from_max(max_proc1,max_proc2,mpi_size)

def compute_2d_process_grid_from_max( max_proc1 : int, max_proc2: int, mpi_size : int ):
    # initialise the number of processes in each distributed dimension
    nprocs1 = 1
    nprocs2 = mpi_size
    
    # if the initial values are not valid
    while (nprocs2>max_proc2):
        # increase number of processes in the first dimension until a divisor is found
        nprocs1+=1
        while(nprocs1<=mpi_size and mpi_size%nprocs1!=0):
            nprocs1+=1
        
        # if an acceptable value is not found then throw an error
        if (nprocs1==mpi_size+1):
            raise RuntimeError("There is no valid combination of processors for this grid")
        
        # save the corresponding second dimension
        nprocs2=mpi_size//nprocs1
        
        # loop to check that the new values are valid
    
    
    # Find the ratio between the minimum number of elements saved in a distributed direction
    divisions1 = max_proc1/nprocs1
    divisions2 = max_proc2/nprocs2
    ratio = max(divisions1,divisions2)/min(divisions1,divisions2)
    
    while (True):
        # Look for the next valid division of processes by increasing
        # the number of processes in the first dimension until a
        # divisor is found
        new_n1=nprocs1+1
        while(new_n1<mpi_size and mpi_size%new_n1!=0):
            new_n1+=1
        new_n2=mpi_size//new_n1
        
        # if there are no more valid divisions
        # then the current setup is optimal
        if (new_n1>mpi_size):
            break
        
        if (new_n2<=max_proc2):
            # Recalculate the ratio between the minimum number
            # of elements saved in a distributed direction
            divisions1 = max_proc1/new_n1
            divisions2 = max_proc2/new_n2
            new_ratio = max(divisions1,divisions2)/min(divisions1,divisions2)
            if (new_ratio<ratio):
                # if the ratio is better save the improved values
                nprocs1=new_n1
                nprocs2=new_n2
                ratio=new_ratio
            else:
                # if the ratio isn't better then the current setup is optimal
                break

    return nprocs1, nprocs2
