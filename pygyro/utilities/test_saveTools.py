from mpi4py import MPI
from .savingTools           import setupSave
import os
import pytest

@pytest.mark.serial
def test_Save_s():
    npts = [10,20,10,10]
    n1=setupSave(3,3,3,3,npts,0.1)
    n2=setupSave(3,3,3,3,npts,0.1)
    n3=setupSave(3,3,3,3,npts,0.1,"testFilename")
    assert(os.path.isdir(n1))
    assert(os.path.isdir(n2))
    assert(os.path.isdir(n3))
    os.remove("{0}/initParams.h5".format(n1))
    os.remove("{0}/initParams.h5".format(n2))
    os.remove("{0}/initParams.h5".format(n3))
    os.rmdir(n1)
    os.rmdir(n2)
    os.rmdir(n3)

@pytest.mark.parallel
def test_Save_p():
    npts = [10,20,10,10]
    n1=setupSave(3,3,3,3,npts,0.1)
    n2=setupSave(3,3,3,3,npts,0.1)
    n3=setupSave(3,3,3,3,npts,0.1,"testFilename")
    if (MPI.COMM_WORLD.Get_rank()==0):
        assert(os.path.isdir(n1))
        assert(os.path.isdir(n2))
        assert(os.path.isdir(n3))
        os.remove("{0}/initParams.h5".format(n1))
        os.remove("{0}/initParams.h5".format(n2))
        os.remove("{0}/initParams.h5".format(n3))
        os.rmdir(n1)
        os.rmdir(n2)
        os.rmdir(n3)
