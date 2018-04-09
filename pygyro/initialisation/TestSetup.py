from mpi4py import MPI
from .Setups import RadialSetup, BlockSetup
from src.initialisation.Initialiser import Initialiser, getEquilibrium, getPerturbation
import src.initialisation.Constants as Constants
import numpy as np
import unittest
from functools import reduce
from src import SlicePlotter4d, SlicePlotter3d, Plotter2d

class TestSetup(unittest.TestCase):
    def test_RadialSetup(self):
        nr=10
        ntheta=10
        nz=20
        nv=20
        grid = RadialSetup(nr,ntheta,nz,nv,Constants.rMin,Constants.rMax,0.0,10.0,5.0,m=15,n=20)
        #print(grid.f)
        SlicePlotter4d(grid).show()
        
    def test_BlockSetup(self):
        nr=10
        ntheta=10
        nz=20
        nv=20
        grid = BlockSetup(nr,ntheta,nz,nv,Constants.rMin,Constants.rMax,0.0,10.0,5.0,m=15,n=20)
        SlicePlotter4d(grid).show()
    
    def test_compareSetups(self):
        nr=10
        ntheta=20
        nz=10
        nv=10
        grid1 = BlockSetup(nr,ntheta,nz,nv,Constants.rMin,Constants.rMax,0.0,10.0,5.0)
        grid2 = RadialSetup(nr,ntheta,nz,nv,Constants.rMin,Constants.rMax,0.0,10.0,5.0)
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
            self.assertTrue(np.equal(blockCreated,radialCreated).all())
        else:
            comm.Gatherv(grid1.f.reshape(grid1.f.size),grid1.f,0)
            comm.Gatherv(grid2.f.reshape(grid2.f.size),grid2.f,0)
    
    def test_Equilibrium(self):
        nr=10
        ntheta=20
        nz=10
        nv=20
        grid = BlockSetup(nr,ntheta,nz,nv,Constants.rMin,Constants.rMax,0.0,10.0,5.0,m=15,n=20)
        rank = MPI.COMM_WORLD.Get_rank()
        getEquilibrium(grid.f,grid.rVals,grid.thetaVals,
                grid.zVals[grid.zStarts[rank]:grid.zStarts[rank+1]],
                grid.vVals,m=15,n=20)
        Plotter2d(grid,'r','v').show()
    
    def test_Perturbation(self):
        nr=20
        ntheta=200
        nz=10
        nv=20
        rank = MPI.COMM_WORLD.Get_rank()
        grid = BlockSetup(nr,ntheta,nz,nv,Constants.rMin,Constants.rMax,0.0,10.0,5.0,m=15,n=20)
        getPerturbation(grid.f,grid.rVals,grid.thetaVals,
                grid.zVals[grid.zStarts[rank]:grid.zStarts[rank+1]],
                grid.vVals,m=15,n=20)
        #grid = RadialSetup(nr,ntheta,nz,nv,Constants.rMin,Constants.rMax,0.0,10.0,5.0,m=15,n=20)
        #getPerturbation(grid.f,grid.rVals[grid.rStarts[rank]:grid.rStarts[rank+1]],grid.thetaVals,
        #        grid.zVals,grid.vVals,m=15,n=20)
        SlicePlotter3d(grid).show()

if __name__ == '__main__':
    unittest.main()
