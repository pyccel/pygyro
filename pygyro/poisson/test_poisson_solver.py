from mpi4py import MPI
import numpy as np
import pytest

from ..model.process_grid       import compute_2d_process_grid_from_max
from ..model.layout             import LayoutSwapper
from ..model.grid               import Grid
from ..initialisation.setups    import setupCylindricalGrid
from .poisson_solver            import PoissonSolver, DensityFinder

def test_PoissonSolver():
    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()
    
    npts = [32, 64, 32]
    nptsGrid = [*npts, 16]
    
    n1 = min(npts[0],npts[1])
    n2 = 2
    
    eta_grids = [np.linspace( 0,10, num=num ) for num in npts ]
    
    nprocs = compute_2d_process_grid_from_max( n1 , n2 , mpi_size )
    
    # Create dictionary describing layouts
    layout_poisson = {'mode_solve': [1,2,0],
                      'v_parallel': [0,2,1]}
    layout_advection = {'dphi'      : [0,1,2],
                        'poloidal'  : [2,1,0],
                        'r_distrib' : [0,2,1]}
    
    nproc = nprocs[0]
    
    remapper = LayoutSwapper( comm, [layout_poisson,layout_advection],[nprocs,nproc], eta_grids, 'v_parallel' )
    
    rho = Grid(eta_grids,[],remapper,'v_parallel',comm,dtype=np.complex128)
    phi = Grid(eta_grids,[],remapper,'v_parallel',comm,dtype=np.complex128)
    
    grid = setupCylindricalGrid(npts=nptsGrid,layout='v_parallel')
    
    df = DensityFinder(3,grid)
    
    df.getRho(grid,rho)
    
    psolver = PoissonSolver(eta_grids)
    
    psolver.getModes(phi,rho)
    
    phi.setLayout('mode_solve')
