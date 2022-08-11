from mpi4py import MPI
import sys

from pygyro.utilities.grid_plotter import SlicePlotterNd
from pygyro.initialisation.setups import setupCylindricalGrid, setupFromFile

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if len(sys.argv == 1):
    foldername = sys.argv[1]

    distribFunc, constants, t = setupFromFile(foldername, comm=comm,
                                              allocateSaveMemory=True,
                                              timepoint=tEnd)
else:
    print("Showing initialisation with default values")
    distribFunc, constants, t = setupCylindricalGrid(layout='v_parallel',
                                                     comm=comm,
                                                     allocateSaveMemory=True)

plotter = SlicePlotterNd(distribFunc, 0, 1, True, sliderDimensions=[
                         2, 3], sliderNames=['z', 'v'])

plotter.show()
