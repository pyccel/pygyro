from mpi4py import MPI
import pytest

from  .                       import constants
from  .setups                 import setupCylindricalGrid
from  .initialiser            import getEquilibrium, getPerturbation
from ..utilities.grid_plotter import SlicePlotter4d, SlicePlotter3d, Plotter2d
from ..model.grid             import Layout

@pytest.mark.serial
def test_Equilibrium_FieldAligned():
    nr=10
    ntheta=20
    nz=10
    nv=20
    grid = setupCylindricalGrid(nr,ntheta,nz,nv,Layout.FIELD_ALIGNED,m=15,n=20)
    rank = MPI.COMM_WORLD.Get_rank()
    getEquilibrium(grid)
    Plotter2d(grid,'r','v').show()

@pytest.mark.serial
def test_Equilibrium_vPar():
    nr=10
    ntheta=20
    nz=10
    nv=20
    grid = setupCylindricalGrid(nr,ntheta,nz,nv,Layout.V_PARALLEL,m=15,n=20)
    rank = MPI.COMM_WORLD.Get_rank()
    getEquilibrium(grid)
    Plotter2d(grid,'r','v').show()

@pytest.mark.serial
def test_Equilibrium_Poloidal():
    nr=10
    ntheta=20
    nz=10
    nv=20
    grid = setupCylindricalGrid(nr,ntheta,nz,nv,Layout.POLOIDAL,m=15,n=20)
    rank = MPI.COMM_WORLD.Get_rank()
    getEquilibrium(grid)
    Plotter2d(grid,'r','v').show()

@pytest.mark.serial
def test_FieldAligned():
    nr=10
    ntheta=10
    nz=20
    nv=20
    grid = setupCylindricalGrid(nr,ntheta,nz,nv,Layout.FIELD_ALIGNED,m=15,n=20)
    #print(grid.f)
    SlicePlotter4d(grid).show()

@pytest.mark.serial
def test_vParallel():
    nr=10
    ntheta=10
    nz=20
    nv=20
    grid = setupCylindricalGrid(nr,ntheta,nz,nv,Layout.V_PARALLEL,m=15,n=20)
    SlicePlotter4d(grid).show()

@pytest.mark.serial
def test_Poloidal():
    nr=10
    ntheta=10
    nz=20
    nv=20
    grid = setupCylindricalGrid(nr,ntheta,nz,nv,Layout.POLOIDAL,m=15,n=20)
    SlicePlotter4d(grid).show()

@pytest.mark.serial
def test_Perturbation_FieldAligned():
    nr=50
    ntheta=200
    nz=10
    nv=20
    rank = MPI.COMM_WORLD.Get_rank()
    grid = setupCylindricalGrid(nr,ntheta,nz,nv,Layout.FIELD_ALIGNED,m=15,n=20)
    getPerturbation(grid,m=15,n=20)
    SlicePlotter3d(grid).show()

@pytest.mark.serial
def test_Perturbation_vParallel():
    nr=50
    ntheta=200
    nz=10
    nv=20
    rank = MPI.COMM_WORLD.Get_rank()
    grid = setupCylindricalGrid(nr,ntheta,nz,nv,Layout.V_PARALLEL,m=15,n=20)
    getPerturbation(grid,m=15,n=20)
    SlicePlotter3d(grid).show()

@pytest.mark.serial
def test_Perturbation_Poloidal():
    nr=50
    ntheta=200
    nz=10
    nv=20
    rank = MPI.COMM_WORLD.Get_rank()
    grid = setupCylindricalGrid(nr,ntheta,nz,nv,Layout.POLOIDAL,m=15,n=20)
    getPerturbation(grid,m=15,n=20)
    SlicePlotter3d(grid).show()

@pytest.mark.serial
def test_FieldPlot_FieldAligned():
    nr=20
    ntheta=200
    nz=10
    nv=20
    rank = MPI.COMM_WORLD.Get_rank()
    grid = setupCylindricalGrid(nr,ntheta,nz,nv,Layout.FIELD_ALIGNED,m=15,n=20)
    getPerturbation(grid,m=15,n=20)
    Plotter2d(grid,'q','z').show()

@pytest.mark.serial
def test_FieldPlot_vPar():
    nr=20
    ntheta=200
    nz=10
    nv=20
    rank = MPI.COMM_WORLD.Get_rank()
    grid = setupCylindricalGrid(nr,ntheta,nz,nv,Layout.V_PARALLEL,m=15,n=20)
    getPerturbation(grid,m=15,n=20)
    Plotter2d(grid,'q','z').show()

@pytest.mark.serial
def test_FieldPlot_Poloidal():
    nr=20
    ntheta=200
    nz=10
    nv=20
    rank = MPI.COMM_WORLD.Get_rank()
    grid = setupCylindricalGrid(nr,ntheta,nz,nv,Layout.POLOIDAL,m=15,n=20)
    getPerturbation(grid,m=15,n=20)
    Plotter2d(grid,'q','z').show()

