"""
Plot the 4D distribution function (loaded from file or with default initialisation) using MPI.
"""
import argparse
import os
from mpi4py import MPI

from pygyro.utilities.grid_plotter import SlicePlotterNd
from pygyro.initialisation.setups import setupCylindricalGrid, setupFromFile

parser = argparse.ArgumentParser(
    description='Plot the 4D distribution function')
parser.add_argument('filename', nargs='?', type=str,
                    help='The file whose results should be plotted. If no file is specified then the default initialisation is shown')
args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

filename = args.filename

if filename:
    foldername = os.path.dirname(filename)
    tEnd = int(filename.split('_')[-1][:-3])
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
