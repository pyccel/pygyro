from mpi4py import MPI

from ..initialisation        import constants
from ..initialisation.setups import BlockSetup, RadialSetup
from .grid_plotter           import SlicePlotter4d, SlicePlotter3d, Plotter2d

def test_RadialStitch():
    nr=10
    ntheta=10
    nz=20
    nv=20
    grid=RadialSetup(nr,ntheta,nz,nv,constants.rMin,constants.rMax,0.0,10.0,5.0)

    grid.f[:,:,:,:]=MPI.COMM_WORLD.Get_rank()
    p = SlicePlotter4d(grid)
    p.show()

def test_BlockStitch():
    nr=10
    ntheta=10
    nz=20
    nv=20
    grid=BlockSetup(nr,ntheta,nz,nv,constants.rMin,constants.rMax,0.0,10.0,5.0)
    
    grid.f[:,:,:,:]=MPI.COMM_WORLD.Get_rank()
    p = SlicePlotter4d(grid)
    p.show()

def test_3DPlot():
    nr=10
    ntheta=10
    nz=20
    nv=20
    grid=BlockSetup(nr,ntheta,nz,nv,constants.rMin,constants.rMax,0.0,80.0,5.0)
    
    p = SlicePlotter3d(grid)
    p.show()
    
    p = SlicePlotter3d(grid,'r','z','v')
    p.show()
    
    p = SlicePlotter3d(grid,'q','z','r')
    p.show()
    
    p = SlicePlotter3d(grid,'r','q','v')
    p.show()
    
    p = SlicePlotter3d(grid,'q','z','v')
    p.show()
    

    MPI.COMM_WORLD.Barrier()
