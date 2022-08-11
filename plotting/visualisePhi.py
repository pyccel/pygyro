from mpi4py import MPI
import numpy as np

from pygyro.model.grid import Grid
from pygyro.model.layout import LayoutSwapper, getLayoutHandler
from pygyro.poisson.poisson_solver import DensityFinder, QuasiNeutralitySolver
from pygyro.utilities.grid_plotter import SlicePlotterNd
from pygyro.initialisation.setups import setupCylindricalGrid, setupFromFile
from pygyro.diagnostics.norms import l2


loadable = False

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

distribFunc, constants, t = setupCylindricalGrid(constantFile='iota0.json',
                                                 layout='v_parallel',
                                                 comm=comm,
                                                 allocateSaveMemory=True)

nprocs = distribFunc.getLayout(distribFunc.currentLayout).nprocs[:2]
layout_poisson = {'v_parallel_2d': [0, 2, 1],
                  'mode_solve': [1, 2, 0]}
layout_vpar = {'v_parallel_1d': [0, 2, 1]}
layout_poloidal = {'poloidal': [2, 1, 0]}
remapperPhi = LayoutSwapper(comm, [layout_poisson, layout_vpar, layout_poloidal],
                            [nprocs, nprocs[0], nprocs[1]
                             ], distribFunc.eta_grid[:3],
                            'mode_solve')
remapperRho = getLayoutHandler(
    comm, layout_poisson, nprocs, distribFunc.eta_grid[:3])
phi = Grid(distribFunc.eta_grid[:3], distribFunc.getSpline(slice(0, 3)),
           remapperPhi, 'mode_solve', comm, dtype=np.complex128)
rho = Grid(distribFunc.eta_grid[:3], distribFunc.getSpline(slice(0, 3)),
           remapperRho, 'v_parallel_2d', comm, dtype=np.complex128)

density = DensityFinder(6, distribFunc.getSpline(3),
                        distribFunc.eta_grid, constants)

QNSolver = QuasiNeutralitySolver(distribFunc.eta_grid[:3], 7, distribFunc.getSpline(0),
                                 constants, chi=0)

distribFunc.setLayout('v_parallel')
density.getPerturbedRho(distribFunc, rho)
QNSolver.getModes(rho)
rho.setLayout('mode_solve')
phi.setLayout('mode_solve')
QNSolver.solveEquation(phi, rho)
phi.setLayout('v_parallel_2d')
rho.setLayout('v_parallel_2d')
QNSolver.findPotential(phi)

norm = l2(distribFunc.eta_grid, remapperPhi.getLayout('v_parallel_2d'))

val = norm.l2NormSquared(phi)

print(val)

plotter = SlicePlotterNd(phi, 0, 1, True, sliderDimensions=[
                         2], sliderNames=['z'])

if (rank == 0):
    plotter.show()
else:
    plotter.calculation_complete()
