from mpi4py import MPI
import pytest
from time   import sleep

from ..initialisation.setups import setupCylindricalGrid
from .grid_plotter           import SlicePlotter4d, SlicePlotter3d, Plotter2d, SlicePlotterNd

@pytest.mark.parallel
def test_3DPlot():
    npts=[10,20,10,11]
    comm=MPI.COMM_WORLD
    size=comm.Get_size()
    
    grid,constants,tStart=setupCylindricalGrid('poloidal',npts = npts)
    
    p = SlicePlotterNd(grid,0,1,True,[2],["z"])
    if (comm.Get_rank()==0):
        p.show()
    else:
        sleep(1)
        while (p.calculation_complete()):
            print("loop")
            sleep(1)
    

    comm.Barrier()
