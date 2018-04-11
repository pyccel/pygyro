from mpi4py import MPI

from  .                       import constants
from  .setups                 import RadialSetup, BlockSetup
from  .initialiser            import getEquilibrium, getPerturbation
from ..utilities.grid_plotter import SlicePlotter4d, SlicePlotter3d, Plotter2d

def test_RadialSetup():
    nr=10
    ntheta=10
    nz=20
    nv=20
    grid = RadialSetup(nr,ntheta,nz,nv,constants.rMin,constants.rMax,0.0,10.0,5.0,m=15,n=20)
    #print(grid.f)
    SlicePlotter4d(grid).show()
    
def test_BlockSetup():
    nr=10
    ntheta=10
    nz=20
    nv=20
    grid = BlockSetup(nr,ntheta,nz,nv,constants.rMin,constants.rMax,0.0,10.0,5.0,m=15,n=20)
    SlicePlotter4d(grid).show()

def test_Equilibrium():
    nr=10
    ntheta=20
    nz=10
    nv=20
    grid = BlockSetup(nr,ntheta,nz,nv,constants.rMin,constants.rMax,0.0,10.0,5.0,m=15,n=20)
    rank = MPI.COMM_WORLD.Get_rank()
    getEquilibrium(grid.f,grid.rVals,grid.thetaVals,
            grid.zVals[grid.zStarts[rank]:grid.zStarts[rank+1]],
            grid.vVals)
    Plotter2d(grid,'r','v').show()

def test_Perturbation():
    nr=20
    ntheta=200
    nz=10
    nv=20
    rank = MPI.COMM_WORLD.Get_rank()
    grid = BlockSetup(nr,ntheta,nz,nv,constants.rMin,constants.rMax,0.0,100.0,5.0,m=15,n=20)
    getPerturbation(grid.f,grid.rVals,grid.thetaVals,
            grid.zVals[grid.zStarts[rank]:grid.zStarts[rank+1]],
            grid.vVals,m=15,n=20)
    #grid = RadialSetup(nr,ntheta,nz,nv,constants.rMin,constants.rMax,0.0,10.0,5.0,m=15,n=20)
    #getPerturbation(grid.f,grid.rVals[grid.rStarts[rank]:grid.rStarts[rank+1]],grid.thetaVals,
    #        grid.zVals,grid.vVals,m=15,n=20)
    SlicePlotter3d(grid).show()

def test_thetaZPlot():
    nr=20
    ntheta=200
    nz=10
    nv=20
    rank = MPI.COMM_WORLD.Get_rank()
    grid = BlockSetup(nr,ntheta,nz,nv,constants.rMin,constants.rMax,0.0,100.0,5.0,m=15,n=20)
    getPerturbation(grid.f,grid.rVals,grid.thetaVals,
            grid.zVals[grid.zStarts[rank]:grid.zStarts[rank+1]],
            grid.vVals,m=15,n=20)
    #grid = RadialSetup(nr,ntheta,nz,nv,constants.rMin,constants.rMax,0.0,10.0,5.0,m=15,n=20)
    #getPerturbation(grid.f,grid.rVals[grid.rStarts[rank]:grid.rStarts[rank+1]],grid.thetaVals,
    #        grid.zVals,grid.vVals,m=15,n=20)
    Plotter2d(grid,'q','z').show()

