from mpi4py import MPI
import numpy as np
import pytest

from .Grid import Layout, Grid

def test_Grid():
    Grid([0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],Layout.RADIAL)
    Grid([0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],Layout.BLOCK)
    Grid([0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],"radial")
    Grid([0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],"block")
    with pytest.raises(NotImplementedError):
        Grid([0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],"madeup")

def test_layoutSwap():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    r=np.arange(0,size*3)
    z=np.arange(0,size*2)

    nv=10
    ntheta=20
    nr=10
    nz=15

    f = np.empty((nv,ntheta,nr,nz*size),dtype=object,order='F')

    for i in range(0,nv):
        for j in range(0,ntheta):
            for k in range(0,nr):
                for l in np.arange(0,size*nz):
                    f[i,j,k,l]="V%dQ%iR%iZ%i" % (i,j,k+rank*nr,l)
    
    #r,rKnots,theta,thetaKnots,v,vKnots,z,zKnots,rStarts,zStarts,f,layout):
    grid = Grid(range(0,nr),[0],range(0,nz),[0],[0,1],[0],[0,1],[0],
            np.arange(0,size*nr,nr),np.arange(0,size*nz,nz),f,"radial")
    grid.swapLayout()
    if (size>1):
        assert not np.equal(f.shape,grid.f.shape).all()
    grid.swapLayout()
    assert np.equal(f.shape,grid.f.shape).all()
    assert np.equal(f,grid.f).all()
