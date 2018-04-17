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
        # get nearest global index
        I=(np.abs(qVals-theta)).argmin()
        for j,r in grid.getRCoords():
            # get nearest global index
            J=(np.abs(rVals-r)).argmin()
            for k,z in grid.getZCoords():
                # get nearest global index
                K=(np.abs(zVals-z)).argmin()
                for l,v in grid.getVCoords():
                    # get nearest global index
                    L=(np.abs(vVals-v)).argmin()
                    
                    # set value using global indices
                    grid.f[i,j,k,l]=I*nv*nz*nr+J*nv*nz+K*nv+L

def compare_f(rVals,qVals,zVals,vVals,grid):
    nr=len(rVals)
    nq=len(qVals)
    nz=len(zVals)
    nv=len(vVals)
    for i,theta in grid.getThetaCoords():
        # get nearest global index
        I=(np.abs(qVals-theta)).argmin()
        for j,r in grid.getRCoords():
            # get nearest global index
            J=(np.abs(rVals-r)).argmin()
            for k,z in grid.getZCoords():
                # get nearest global index
                K=(np.abs(zVals-z)).argmin()
                for l,v in grid.getVCoords():
                    # get nearest global index
                    L=(np.abs(vVals-v)).argmin()
                    
                    # ensure value is as expected from function define_f()
                    assert(grid.f[i,j,k,l]==I*nv*nz*nr+J*nv*nz+K*nv+L)

@pytest.mark.serial
def test_Grid():
    # ensure that MPI setup is ok
    Grid([0,1],[0,1],[0,1],[0,1],Layout.FIELD_ALIGNED,nProcR=2,nProcV=2)
    Grid([0,1],[0,1],[0,1],[0,1],Layout.POLOIDAL,nProcV=2,nProcZ=2)
    Grid([0,1],[0,1],[0,1],[0,1],Layout.V_PARALLEL,nProcR=2,nProcZ=2)
    # ensure that a non-specified layout throws an error
    with pytest.raises(NotImplementedError):
        Grid([0,1],[0,1],[0,1],[0,1],"madeup")

@pytest.mark.parallel
def test_ErrorsRaised():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if (size==1):
        return
    
    # get a logical combination of processors
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
            Grid([0,1],[0,1],[0,1],[0,1],Layout.FIELD_ALIGNED,nProcR=n1,nProcZ=n2)
        with pytest.raises(ValueError):
            Grid([0,1],[0,1],[0,1],[0,1],Layout.POLOIDAL,nProcR=n1,nProcZ=n2)
        with pytest.raises(ValueError):
            Grid([0,1],[0,1],[0,1],[0,1],Layout.V_PARALLEL,nProcR=n1,nProcV=n2)

@pytest.mark.parallel
@pytest.mark.parametrize("splitN", [1,2,3,4,5,6])
def test_layoutSwap(splitN):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # get a logical combination of processors
    n1=max(size//splitN,1)
    n2=size//n1
    if (n1*n2!=size):
        return
    
    nv=10
    ntheta=20
    nr=10
    nz=15
    
    # create equally spaced grid
    grid = Grid(np.linspace(0.5,14.5,nr),np.linspace(0,2*pi,ntheta,endpoint=False),
                np.linspace(0,50,nz),np.linspace(-5,5,nv),Layout.FIELD_ALIGNED,nProcR=n1,nProcV=n2)
    # save shape
    oldShape = grid.f.shape
    # fill f based on global indices
    define_f(grid.Vals[Grid.Dimension.R],grid.Vals[Grid.Dimension.THETA],
            grid.Vals[Grid.Dimension.Z],grid.Vals[Grid.Dimension.V],grid)
    
    # change layout
    grid.setLayout(Layout.V_PARALLEL)
    
    # ensure that f still looks as expected
    compare_f(grid.Vals[Grid.Dimension.R],grid.Vals[Grid.Dimension.THETA],
            grid.Vals[Grid.Dimension.Z],grid.Vals[Grid.Dimension.V],grid)
    
    if (n2>1):
        # if shape should have changed ensure that this has happened
        assert not np.equal(oldShape,grid.f.shape).all()
    
    # change layout
    grid.setLayout(Layout.POLOIDAL)
    
    # ensure that f still looks as expected
    compare_f(grid.Vals[Grid.Dimension.R],grid.Vals[Grid.Dimension.THETA],
            grid.Vals[Grid.Dimension.Z],grid.Vals[Grid.Dimension.V],grid)
    
    if (n1>1):
        # if shape should have changed ensure that this has happened
        assert not np.equal(oldShape,grid.f.shape).all()
    
    # change layout
    grid.setLayout(Layout.V_PARALLEL)
    
    # ensure that f still looks as expected
    compare_f(grid.Vals[Grid.Dimension.R],grid.Vals[Grid.Dimension.THETA],
            grid.Vals[Grid.Dimension.Z],grid.Vals[Grid.Dimension.V],grid)
    
    # change layout
    grid.setLayout(Layout.FIELD_ALIGNED)
    
    # ensure that f still looks as expected
    compare_f(grid.Vals[Grid.Dimension.R],grid.Vals[Grid.Dimension.THETA],
            grid.Vals[Grid.Dimension.Z],grid.Vals[Grid.Dimension.V],grid)
    
    # All expected layout changes have been run.
    # The following two are possible but should raise errors as they require 2 steps
    with pytest.raises(RuntimeWarning):
        # change layout
        grid.setLayout(Layout.POLOIDAL)
    
    # ensure that f still looks as expected
    compare_f(grid.Vals[Grid.Dimension.R],grid.Vals[Grid.Dimension.THETA],
            grid.Vals[Grid.Dimension.Z],grid.Vals[Grid.Dimension.V],grid)
    with pytest.raises(RuntimeWarning):
        # change layout
        grid.setLayout(Layout.FIELD_ALIGNED)
    
    # ensure that f still looks as expected
    compare_f(grid.Vals[Grid.Dimension.R],grid.Vals[Grid.Dimension.THETA],
            grid.Vals[Grid.Dimension.Z],grid.Vals[Grid.Dimension.V],grid)
