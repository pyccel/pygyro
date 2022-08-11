from mpi4py import MPI
import sys

from pygyro.utilities.grid_plotter import SlicePlotterNd
from pygyro.initialisation.setups import setupCylindricalGrid

foldername = sys.argv[1]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# distribFunc, constants, t = setupCylindricalGrid(constantFile='iota0.json',
#                                                 layout='v_parallel',
#                                                 comm=comm,
#                                                 allocateSaveMemory=True)
distribFunc, constants, t = setupFromFile(foldername, comm=comm,
                                          allocateSaveMemory=True,
                                          timepoint=tEnd)

plotter = SlicePlotterNd(distribFunc, 0, 1, True, sliderDimensions=[
                         2, 3], sliderNames=['z', 'v'])

plotter.show()
