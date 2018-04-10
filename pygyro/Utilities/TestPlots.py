from pygyro import BlockSetup, RadialSetup, SlicePlotter4d, SlicePlotter3d, Plotter2d, Constants
import unittest
from mpi4py import MPI

class TestPlot(unittest.TestCase):
    def test_RadialStitch(self):
        nr=10
        ntheta=10
        nz=20
        nv=20
        grid=RadialSetup(nr,ntheta,nz,nv,Constants.rMin,Constants.rMax,0.0,10.0,5.0)
        
        grid.f[:,:,:,:]=MPI.COMM_WORLD.Get_rank()
        p = SlicePlotter4d(grid)
        p.show()
    
    def test_BlockStitch(self):
        nr=10
        ntheta=10
        nz=20
        nv=20
        grid=BlockSetup(nr,ntheta,nz,nv,Constants.rMin,Constants.rMax,0.0,10.0,5.0)
        
        grid.f[:,:,:,:]=MPI.COMM_WORLD.Get_rank()
        p = SlicePlotter4d(grid)
        p.show()
    
    def test_3DPlot(self):
        nr=10
        ntheta=10
        nz=20
        nv=20
        grid=BlockSetup(nr,ntheta,nz,nv,Constants.rMin,Constants.rMax,0.0,10.0,5.0)
        
        p = SlicePlotter3d(grid)
        p.show()
        
        p = SlicePlotter3d(grid,'q','z','v')
        p.show()
        
        p = SlicePlotter3d(grid,'r','z','v')
        p.show()
        
        p = SlicePlotter3d(grid,'q','z','r')
        p.show()
        
        MPI.COMM_WORLD.Barrier()
    
        

if __name__ == '__main__':
    unittest.main()
