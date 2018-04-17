from mpi4py import MPI
import numpy as np
import pytest
from math import pi

from .grid import Layout, Grid

def define_f(rVals,qVals,zVals,vVals,grid):
    nr=len(rVals)
    nq=len(qVals)
    nz=len(zVals)
    nv=len(vVals)
    for i,theta in grid.getThetaCoords():
        I=(np.abs(qVals-theta)).argmin()
        for j,r in grid.getRCoords():
            J=(np.abs(rVals-r)).argmin()
            for k,z in grid.getZCoords():
                K=(np.abs(zVals-z)).argmin()
                for l,v in grid.getVCoords():
                    L=(np.abs(vVals-v)).argmin()
                    grid.f[i,j,k,l]=I*nv*nz*nr+J*nv*nz+K*nv+L

def compare_f(rVals,qVals,zVals,vVals,grid):
    nr=len(rVals)
    nq=len(qVals)
    nz=len(zVals)
    nv=len(vVals)
    for i,theta in grid.getThetaCoords():
        I=(np.abs(qVals-theta)).argmin()
        for j,r in grid.getRCoords():
            J=(np.abs(rVals-r)).argmin()
            for k,z in grid.getZCoords():
                K=(np.abs(zVals-z)).argmin()
                for l,v in grid.getVCoords():
                    L=(np.abs(vVals-v)).argmin()
                    assert(grid.f[i,j,k,l]==I*nv*nz*nr+J*nv*nz+K*nv+L)

@pytest.mark.serial
def test_Grid():
    Grid([0,1],[0,1],[0,1],[0,1],Layout.FIELD_ALIGNED,nProcR=2,nProcV=2)
    Grid([0,1],[0,1],[0,1],[0,1],Layout.POLOIDAL,nProcV=2,nProcZ=2)
    with pytest.raises(NotImplementedError):
        Grid([0,1],[0,1],[0,1],[0,1],"madeup")

@pytest.mark.parallel
def test_ErrorsRaised():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if (size==1):
        return
    i=2
    while (i<size):
        n1=i
        n2=size//i
        if (n1*n2==size):
            i=size
        else:
            i=i+1
    
    if (n1*n2!=size):
        return
    else:
        with pytest.raises(ValueError):
            Grid([0,1],[0,1],[0,1],[0,1],Layout.FIELD_ALIGNED,nProcR=2,nProcZ=2)
        with pytest.raises(ValueError):
            Grid([0,1],[0,1],[0,1],[0,1],Layout.POLOIDAL,nProcR=2,nProcZ=2)
        with pytest.raises(ValueError):
            Grid([0,1],[0,1],[0,1],[0,1],Layout.V_PARALLEL,nProcR=2,nProcV=2)

@pytest.mark.parallel
@pytest.mark.parametrize("splitN", [1,2,3,4,5,6])
def test_layoutSwap(splitN):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    n1=max(size//splitN,1)
    n2=size//n1
    if (n1*n2!=size):
        return
    
    nv=10
    ntheta=20
    nr=10
    nz=15
    
    #grid = setupGrid(nr,ntheta,nz,nv,Layout.FIELD_ALIGNED,nProcR=n1,nProcV=n2)
    grid = Grid(np.linspace(0.5,14.5,nr),np.linspace(0,2*pi,ntheta,endpoint=False),
                np.linspace(0,50,nz),np.linspace(-5,5,nv),Layout.FIELD_ALIGNED,nProcR=n1,nProcV=n2)
    oldShape = grid.f.shape
    define_f(grid.Vals[Grid.Dimension.R],grid.Vals[Grid.Dimension.THETA],
            grid.Vals[Grid.Dimension.Z],grid.Vals[Grid.Dimension.V],grid)
    grid.setLayout(Layout.V_PARALLEL)
    compare_f(grid.Vals[Grid.Dimension.R],grid.Vals[Grid.Dimension.THETA],
            grid.Vals[Grid.Dimension.Z],grid.Vals[Grid.Dimension.V],grid)
    if (n2>1):
        assert not np.equal(oldShape,grid.f.shape).all()
    grid.setLayout(Layout.POLOIDAL)
    compare_f(grid.Vals[Grid.Dimension.R],grid.Vals[Grid.Dimension.THETA],
            grid.Vals[Grid.Dimension.Z],grid.Vals[Grid.Dimension.V],grid)
    if (n1>1):
        assert not np.equal(oldShape,grid.f.shape).all()
    grid.setLayout(Layout.V_PARALLEL)
    compare_f(grid.Vals[Grid.Dimension.R],grid.Vals[Grid.Dimension.THETA],
            grid.Vals[Grid.Dimension.Z],grid.Vals[Grid.Dimension.V],grid)
    grid.setLayout(Layout.FIELD_ALIGNED)
    compare_f(grid.Vals[Grid.Dimension.R],grid.Vals[Grid.Dimension.THETA],
            grid.Vals[Grid.Dimension.Z],grid.Vals[Grid.Dimension.V],grid)
    with pytest.raises(RuntimeWarning):
        grid.setLayout(Layout.POLOIDAL)
    compare_f(grid.Vals[Grid.Dimension.R],grid.Vals[Grid.Dimension.THETA],
            grid.Vals[Grid.Dimension.Z],grid.Vals[Grid.Dimension.V],grid)
    with pytest.raises(RuntimeWarning):
        grid.setLayout(Layout.FIELD_ALIGNED)
    compare_f(grid.Vals[Grid.Dimension.R],grid.Vals[Grid.Dimension.THETA],
            grid.Vals[Grid.Dimension.Z],grid.Vals[Grid.Dimension.V],grid)
