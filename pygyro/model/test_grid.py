from mpi4py import MPI
import numpy as np
import pytest
from math import pi

from .grid import Layout, Grid

def define_f(eta1Vals,eta2Vals,eta3Vals,eta4Vals,grid):
    nEta1=len(eta1Vals)
    nEta2=len(eta2Vals)
    nEta3=len(eta3Vals)
    nEta4=len(eta4Vals)
    for i,eta1 in grid.getEta1Coords():
        # get nearest global index
        I=(np.abs(eta1Vals-eta1)).argmin()
        for j,eta3 in grid.getEta3Coords():
            # get nearest global index
            J=(np.abs(eta3Vals-eta3)).argmin()
            for k,eta4 in grid.getEta4Coords():
                # get nearest global index
                K=(np.abs(eta4Vals-eta4)).argmin()
                for l,eta2 in grid.getEta2Coords():
                    # get nearest global index
                    L=(np.abs(eta2Vals-eta2)).argmin()
                    
                    # set value using global indices
                    grid.f[i,j,k,l]=I*nEta4*nEta3*nEta2+J*nEta4*nEta2+K*nEta2+L

def compare_f(eta1Vals,eta2Vals,eta3Vals,eta4Vals,grid):
    nEta1=len(eta1Vals)
    nEta2=len(eta2Vals)
    nEta3=len(eta3Vals)
    nEta4=len(eta4Vals)
    for i,eta1 in grid.getEta1Coords():
        # get nearest global index
        I=(np.abs(eta1Vals-eta1)).argmin()
        for j,eta3 in grid.getEta3Coords():
            # get nearest global index
            J=(np.abs(eta3Vals-eta3)).argmin()
            for k,eta4 in grid.getEta4Coords():
                # get nearest global index
                K=(np.abs(eta4Vals-eta4)).argmin()
                for l,eta2 in grid.getEta2Coords():
                    # get nearest global index
                    L=(np.abs(eta2Vals-eta2)).argmin()
                    
                    # ensure value is as expected from function define_f()
                    assert(grid.f[i,j,k,l]==I*nEta4*nEta3*nEta2+J*nEta4*nEta2+K*nEta2+L)

@pytest.mark.serial
def test_Grid():
    # ensure that MPI setup is ok
    Grid([0,1],[0,1],[0,1],[0,1],Layout.FIELD_ALIGNED,nProcEta1=2,nProcEta4=2)
    Grid([0,1],[0,1],[0,1],[0,1],Layout.POLOIDAL,nProcEta4=2,nProcEta3=2)
    Grid([0,1],[0,1],[0,1],[0,1],Layout.V_PARALLEL,nProcEta1=2,nProcEta3=2)
    # ensure that a non-specified layout throws an error
    with pytest.raises(NotImplementedError):
        Grid([0,1],[0,1],[0,1],[0,1],"madeup")

@pytest.mark.serial
def test_CoordinateSave():
    nv=10
    ntheta=20
    nr=30
    nz=15
    
    r = np.linspace(0.5,14.5,nr)
    theta = np.linspace(0,2*pi,ntheta,endpoint=False)
    z = np.linspace(0,50,nz)
    v = np.linspace(-5,5,nv)
    
    # create equally spaced grid
    grid = Grid(r,theta,z,v,Layout.FIELD_ALIGNED)
    
    assert(grid.nEta1==nr)
    assert(grid.nEta2==ntheta)
    assert(grid.nEta3==nz)
    assert(grid.nEta4==nv)
    assert(all(grid.eta1_Vals==r))
    assert(all(grid.eta2_Vals==theta))
    assert(all(grid.eta3_Vals==z))
    assert(all(grid.eta4_Vals==v))

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
            Grid([0,1],[0,1],[0,1],[0,1],Layout.FIELD_ALIGNED,nProcEta1=n1,nProcEta3=n2)
        with pytest.raises(ValueError):
            Grid([0,1],[0,1],[0,1],[0,1],Layout.POLOIDAL,nProcEta1=n1,nProcEta3=n2)
        with pytest.raises(ValueError):
            Grid([0,1],[0,1],[0,1],[0,1],Layout.V_PARALLEL,nProcEta1=n1,nProcEta4=n2)

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
                np.linspace(0,50,nz),np.linspace(-5,5,nv),Layout.FIELD_ALIGNED,nProcEta1=n1,nProcEta4=n2)
    # save shape
    oldShape = grid.f.shape
    # fill f based on global indices
    define_f(grid.Vals[Grid.Dimension.ETA1],grid.Vals[Grid.Dimension.ETA2],
            grid.Vals[Grid.Dimension.ETA3],grid.Vals[Grid.Dimension.ETA4],grid)
    
    # change layout
    grid.setLayout(Layout.V_PARALLEL)
    
    # ensure that f still looks as expected
    compare_f(grid.Vals[Grid.Dimension.ETA1],grid.Vals[Grid.Dimension.ETA2],
            grid.Vals[Grid.Dimension.ETA3],grid.Vals[Grid.Dimension.ETA4],grid)
    
    if (n2>1):
        # if shape should have changed ensure that this has happened
        assert not np.equal(oldShape,grid.f.shape).all()
    
    # change layout
    grid.setLayout(Layout.POLOIDAL)
    
    # ensure that f still looks as expected
    compare_f(grid.Vals[Grid.Dimension.ETA1],grid.Vals[Grid.Dimension.ETA2],
            grid.Vals[Grid.Dimension.ETA3],grid.Vals[Grid.Dimension.ETA4],grid)
    
    if (n1>1):
        # if shape should have changed ensure that this has happened
        assert not np.equal(oldShape,grid.f.shape).all()
    
    # change layout
    grid.setLayout(Layout.V_PARALLEL)
    
    # ensure that f still looks as expected
    compare_f(grid.Vals[Grid.Dimension.ETA1],grid.Vals[Grid.Dimension.ETA2],
            grid.Vals[Grid.Dimension.ETA3],grid.Vals[Grid.Dimension.ETA4],grid)
    
    # change layout
    grid.setLayout(Layout.FIELD_ALIGNED)
    
    # ensure that f still looks as expected
    compare_f(grid.Vals[Grid.Dimension.ETA1],grid.Vals[Grid.Dimension.ETA2],
            grid.Vals[Grid.Dimension.ETA3],grid.Vals[Grid.Dimension.ETA4],grid)
    
    # All expected layout changes have been run.
    # The following two are possible but should raise errors as they require 2 steps
    with pytest.raises(RuntimeWarning):
        # change layout
        grid.setLayout(Layout.POLOIDAL)
    
    # ensure that f still looks as expected
    compare_f(grid.Vals[Grid.Dimension.ETA1],grid.Vals[Grid.Dimension.ETA2],
            grid.Vals[Grid.Dimension.ETA3],grid.Vals[Grid.Dimension.ETA4],grid)
    with pytest.raises(RuntimeWarning):
        # change layout
        grid.setLayout(Layout.FIELD_ALIGNED)
    
    # ensure that f still looks as expected
    compare_f(grid.Vals[Grid.Dimension.ETA1],grid.Vals[Grid.Dimension.ETA2],
            grid.Vals[Grid.Dimension.ETA3],grid.Vals[Grid.Dimension.ETA4],grid)
