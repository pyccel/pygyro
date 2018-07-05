from mpi4py import MPI
import numpy as np
import pytest

from ..model.process_grid       import compute_2d_process_grid
from ..model.layout             import LayoutSwapper
from ..model.grid               import Grid
from ..initialisation.setups    import setupCylindricalGrid
from .poisson_solver            import PoissonSolver, DensityFinder

@pytest.mark.parallel
def test_PoissonSolver():
    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()
    
    npts = [32, 64, 32]
    nptsGrid = [*npts, 16]
    
    n1 = min(npts[0],npts[1])
    n2 = 2
    
    nprocs = compute_2d_process_grid( nptsGrid , mpi_size )
    
    # Create dictionary describing layouts
    layout_poisson = {'mode_solve': [1,2,0],
                      'v_parallel': [0,2,1]}
    layout_advection = {'dphi'      : [0,1,2],
                        'poloidal'  : [2,1,0],
                        'r_distrib' : [0,2,1]}
    
    nproc = nprocs[0]
    
    grid = setupCylindricalGrid(npts=nptsGrid,layout='v_parallel')
    
    remapper = LayoutSwapper( comm, [layout_poisson,layout_advection],[nprocs,nproc], grid.eta_grid[:3], 'v_parallel' )
    
    rho = Grid(grid.eta_grid[:3],grid.getSpline(slice(0,3)),remapper,'v_parallel',comm,dtype=np.complex128)
    phi = Grid(grid.eta_grid[:3],grid.getSpline(slice(0,3)),remapper,'v_parallel',comm,dtype=np.complex128)
    
    df = DensityFinder(3,grid.getSpline(3))
    
    df.getRho(grid,rho)
    
    psolver = PoissonSolver(grid.eta_grid,6,rho.getSpline(0))
    
    psolver.getModes(phi,rho)
    
    phi.setLayout('mode_solve')
    rho.setLayout('mode_solve')
    
    psolver.solveEquation(6,phi,rho)
    
    phi.setLayout('v_parallel')
    
    psolver.findPotential(phi)
