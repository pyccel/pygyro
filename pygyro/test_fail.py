from mpi4py import MPI
import numpy as np
import pytest

@pytest.mark.parallel
def test_Alltoall():
    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    myproc = comm.Get_rank()
    data = np.linspace(0,10,90)

    s_sizes = [90//nproc]*nproc
    r_sizes = [90//nproc]*nproc
    for i in range(nproc):
        if i<90%nproc:
            s_sizes[i]+=1
        if myproc<90%nproc:
            r_sizes[i]+=1

    s_starts = np.zeros(nproc,int)
    r_starts = np.zeros(nproc,int)
    s_starts[1:] = np.cumsum(s_sizes)[:-1]
    r_starts[1:] = np.cumsum(r_sizes)[:-1]

    incoming = np.empty(np.sum(r_sizes),float)

    comm.Alltoallv((data, (s_sizes, s_starts), MPI.DOUBLE), 
                   (incoming, (r_sizes,r_starts), MPI.DOUBLE))
    
    incoming=incoming.reshape(nproc,np.sum(r_sizes)//nproc)
    for i in range(1,nproc):
        assert((incoming[0,:]==incoming[i,:]).all())
