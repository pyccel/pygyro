from mpi4py import MPI
import numpy as np
from functools import reduce
import pytest

from  .setups                 import setupGrid
from  ..model.grid            import Layout

def define_f(rVals,qVals,zVals,vVals,grid):
    nr=len(rVals)
    nq=len(qVals)
    nz=len(zVals)
    nv=len(vVals)
    for i,theta in grid.getThetaCoords():
        I=qVals.index(theta)
        for j,r in grid.getRCoords():
            J=rVals.index(r)
            for k,z in grid.getZCoords():
                K=zVals.index(z)
                for l,v in grid.getVCoords():
                    L=vVals.index(v)
                    grid.f[i,j,k,l]=I*nv*nz*nr+J*nv*nz+K*nv+L

def compare_f(rVals,qVals,zVals,vVals,grid):
    nr=len(rVals)
    nq=len(qVals)
    nz=len(zVals)
    nv=len(vVals)
    for i,theta in grid.getThetaCoords():
        I=qVals.index(theta)
        for j,r in grid.getRCoords():
            J=rVals.index(r)
            for k,z in grid.getZCoords():
                K=zVals.index(z)
                for l,v in grid.getVCoords():
                    L=vVals.index(v)
                    assert(grid.f[i,j,k,l]==I*nv*nz*nr+J*nv*nz+K*nv+L)

"""
def test_RadialToBlockSwap():
    nr=50
    ntheta=10
    nz=90
    nv=20
    gridRad = RadialSetup(nr,ntheta,nz,nv,constants.rMin,constants.rMax,0.0,10.0,5.0,m=15,n=20)
    gridBlock = BlockSetup(nr,ntheta,nz,nv,constants.rMin,constants.rMax,0.0,10.0,5.0,m=15,n=20)
    gridRad.swapLayout()
    assert np.equal(gridBlock.f,gridRad.f).all()

def test_RadialToBlockSwap():
    nr=50
    ntheta=10
    nz=90
    nv=20
    gridRad = RadialSetup(nr,ntheta,nz,nv,constants.rMin,constants.rMax,0.0,10.0,5.0,m=15,n=20)
    gridBlock = BlockSetup(nr,ntheta,nz,nv,constants.rMin,constants.rMax,0.0,10.0,5.0,m=15,n=20)
    gridBlock.swapLayout()
    assert np.equal(gridBlock.f,gridRad.f).all()

def test_compareSetups():
    nr=10
    ntheta=20
    nz=10
    nv=10
    grid1 = BlockSetup(nr,ntheta,nz,nv,constants.rMin,constants.rMax,0.0,10.0,5.0)
    grid2 = RadialSetup(nr,ntheta,nz,nv,constants.rMin,constants.rMax,0.0,10.0,5.0)
    comm=MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    blockCreated = None
    radialCreated = None
    sizes1=comm.gather(grid1.f.size,root=0)
    sizes2=comm.gather(grid2.f.size,root=0)
    shape1=comm.gather(grid1.f.shape,root=0)
    shape2=comm.gather(grid2.f.shape,root=0)

    if rank == 0:
        starts1=np.zeros(len(sizes1))
        starts1[1:]=sizes1[:size-1]
        starts1=starts1.cumsum().astype(int)
        starts2=np.zeros(len(sizes2))
        starts2[1:]=sizes2[:size-1]
        starts2=starts2.cumsum().astype(int)
        blockCreated = np.empty(sum(sizes1), dtype=float)
        radialCreated = np.empty(sum(sizes2), dtype=float)
        comm.Gatherv(grid1.f.reshape(grid1.f.size),(blockCreated, sizes1, starts1, MPI.DOUBLE), 0)
        comm.Gatherv(grid2.f.reshape(grid2.f.size),(radialCreated, sizes2, starts2, MPI.DOUBLE), 0)
        blockCreated=np.split(blockCreated,starts1[1:])
        radialCreated=np.split(radialCreated,starts2[1:])
        for i in range(0,len(blockCreated)):
            blockCreated[i]=blockCreated[i].reshape(shape1[i])
            radialCreated[i]=radialCreated[i].reshape(shape2[i])
        blockCreated=np.concatenate(blockCreated,axis=3)
        radialCreated=np.concatenate(radialCreated,axis=2)
        assert np.equal(blockCreated,radialCreated).all()
    else:
        comm.Gatherv(grid1.f.reshape(grid1.f.size),grid1.f,0)
        comm.Gatherv(grid2.f.reshape(grid2.f.size),grid2.f,0)
"""

@pytest.mark.serial
def test_FieldAligned_setup():
    nr=10
    ntheta=20
    nz=10
    nv=10
    setupGrid(nr, ntheta, nz, nv, Layout.FIELD_ALIGNED)

@pytest.mark.serial
def test_Poloidal_setup():
    nr=10
    ntheta=20
    nz=10
    nv=10
    setupGrid(nr, ntheta, nz, nv, Layout.POLOIDAL)

@pytest.mark.serial
def test_vParallel_setup():
    nr=10
    ntheta=20
    nz=10
    nv=10
    setupGrid(nr, ntheta, nz, nv, Layout.V_PARALLEL)

@pytest.mark.parallel
@pytest.mark.parametrize("splitN", [1,2,3,4,5,6])
def test_FieldAligned_setup_parallel(splitN):
    nr=10
    ntheta=20
    nz=10
    nv=10
    size=MPI.COMM_WORLD.Get_size()
    n1=max(size//splitN,1)
    n2=size//n1
    if (n1*n2!=size):
        return
    
    setupGrid(nr, ntheta, nz, nv, Layout.FIELD_ALIGNED,nProcR=n1,nProcV=n2)

@pytest.mark.parallel
@pytest.mark.parametrize("splitN", [1,2,3,4,5,6])
def test_Poloidal_setup_parallel(splitN):
    nr=10
    ntheta=20
    nz=10
    nv=10
    size=MPI.COMM_WORLD.Get_size()
    n1=max(size//splitN,1)
    n2=size//n1
    if (n1*n2!=size):
        return
    
    setupGrid(nr, ntheta, nz, nv, Layout.POLOIDAL,nProcZ=n1,nProcV=n2)

@pytest.mark.parallel
@pytest.mark.parametrize("splitN", [1,2,3,4,5,6])
def test_vParallel_setup_parallel(splitN):
    nr=10
    ntheta=20
    nz=10
    nv=10
    size=MPI.COMM_WORLD.Get_size()
    n1=max(size//splitN,1)
    n2=size//n1
    if (n1*n2!=size):
        return
    
    setupGrid(nr, ntheta, nz, nv, Layout.V_PARALLEL,nProcR=n1,nProcZ=n2)
