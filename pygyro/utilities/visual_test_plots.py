from mpi4py import MPI
import pytest

from ..initialisation.setups import setupCylindricalGrid
from .grid_plotter           import SlicePlotter4d, SlicePlotter3d, Plotter2d

@pytest.mark.parallel
def test_FluxSurface_Stitch():
    npts=[10,20,10,10]
    comm=MPI.COMM_WORLD
    size=comm.Get_size()
    
    grid=setupCylindricalGrid(npts, 'flux_surface')

    grid._f[:,:,:,:]=comm.Get_rank()
    p = SlicePlotter4d(grid)
    p.show()

@pytest.mark.parallel
def test_Poloidal_Stitch():
    npts=[10,20,10,10]
    comm=MPI.COMM_WORLD
    size=comm.Get_size()
    
    grid=setupCylindricalGrid(npts, 'poloidal')

    grid._f[:,:,:,:]=comm.Get_rank()
    p = SlicePlotter4d(grid)
    p.show()

@pytest.mark.parallel
def test_vParallel_Stitch():
    npts=[10,20,10,10]
    comm=MPI.COMM_WORLD
    size=comm.Get_size()
    
    grid=setupCylindricalGrid(npts, 'v_parallel')

    grid._f[:,:,:,:]=comm.Get_rank()
    p = SlicePlotter4d(grid)
    p.show()

@pytest.mark.parallel
def test_3DPlot():
    npts=[10,20,10,10]
    comm=MPI.COMM_WORLD
    size=comm.Get_size()
    
    grid=setupCylindricalGrid(npts, 'poloidal')
    
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
    

    comm.Barrier()

