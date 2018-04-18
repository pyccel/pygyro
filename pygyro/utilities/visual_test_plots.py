from mpi4py import MPI
import pytest

from ..model.grid            import Layout
from ..initialisation        import constants
from ..initialisation.setups import setupCylindricalGrid
from .grid_plotter           import SlicePlotter4d, SlicePlotter3d, Plotter2d

@pytest.mark.parallel
@pytest.mark.parametrize("splitN", [1,2,3,4,5,6])
def test_FieldAligned_Stitch(splitN):
    nr=10
    ntheta=20
    nz=10
    nv=10
    size=MPI.COMM_WORLD.Get_size()
    n1=max(size//splitN,1)
    n2=size//n1
    if (n1*n2!=size):
        return
    
    grid=setupCylindricalGrid(nr, ntheta, nz, nv, Layout.FIELD_ALIGNED,nProcEta1=n1,nProcEta4=n2)

    grid.f[:,:,:,:]=MPI.COMM_WORLD.Get_rank()
    p = SlicePlotter4d(grid)
    p.show()

@pytest.mark.parallel
@pytest.mark.parametrize("splitN", [1,2,3,4,5,6])
def test_Poloidal_Stitch(splitN):
    nr=10
    ntheta=20
    nz=10
    nv=10
    size=MPI.COMM_WORLD.Get_size()
    n1=max(size//splitN,1)
    n2=size//n1
    if (n1*n2!=size):
        return
    
    grid=setupCylindricalGrid(nr, ntheta, nz, nv, Layout.POLOIDAL,nProcEta3=n1,nProcEta4=n2)

    grid.f[:,:,:,:]=MPI.COMM_WORLD.Get_rank()
    p = SlicePlotter4d(grid)
    p.show()

@pytest.mark.parallel
@pytest.mark.parametrize("splitN", [1,2,3,4,5,6])
def test_V_Parallel_Stitch(splitN):
    nr=10
    ntheta=20
    nz=10
    nv=10
    size=MPI.COMM_WORLD.Get_size()
    n1=max(size//splitN,1)
    n2=size//n1
    if (n1*n2!=size):
        return
    
    grid=setupCylindricalGrid(nr, ntheta, nz, nv, Layout.V_PARALLEL,nProcEta1=n1,nProcEta3=n2)

    grid.f[:,:,:,:]=MPI.COMM_WORLD.Get_rank()
    p = SlicePlotter4d(grid)
    p.show()

@pytest.mark.parallel
@pytest.mark.parametrize("splitN", [1,2,3,4,5,6])
def test_3DPlot(splitN):
    nr=10
    ntheta=20
    nz=10
    nv=10
    size=MPI.COMM_WORLD.Get_size()
    n1=max(size//splitN,1)
    n2=size//n1
    if (n1*n2!=size):
        return
    
    grid=setupCylindricalGrid(nr, ntheta, nz, nv, Layout.POLOIDAL,nProcEta3=n1,nProcEta4=n2)
    
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

