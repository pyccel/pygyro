from mpi4py     import MPI
import numpy    as np
import pytest

from .energy                                import KineticEnergy
from ..                                     import splines as spl
from ..model.process_grid                   import compute_2d_process_grid_from_max
from ..model.grid                           import Grid
from ..model.layout                         import LayoutSwapper
from ..initialisation.setups                import setupCylindricalGrid
from ..initialisation                       import constants

@pytest.mark.parallel
def test_KineticEnergy_NoVelocity():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    npts = [10,20,10,11]
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'poloidal')
    grid._f[:]=0
    idx_v = np.where(grid.eta_grid[3]==0)[0][0]
    glob_v_vals = grid.getGlobalIdxVals(0)
    if (idx_v in glob_v_vals):
        grid._f[idx_v-glob_v_vals[0]] = 1
    
    theLayout = grid.getLayout(grid.currentLayout)
    
    KE = KineticEnergy(grid.eta_grid,theLayout)
    KE_val = KE.getKE(grid)
    
    KEResult = comm.reduce(KE_val,op=MPI.SUM, root=0)
    
    if (rank==0):
        assert(abs(KEResult)<1e-7)

@pytest.mark.parallel
def test_KineticEnergy_Positive():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    npts = [10,20,10,11]
    grid = setupCylindricalGrid(npts   = npts,
                                layout = 'poloidal',
                                vMax = 4)
    grid._f[:]=0
    idx_v = np.where(grid.eta_grid[3]==-1)[0][0]
    glob_v_vals = grid.getGlobalIdxVals(0)
    if (idx_v in glob_v_vals):
        grid._f[idx_v-glob_v_vals[0]] = 1
    
    theLayout = grid.getLayout(grid.currentLayout)
    
    KE = KineticEnergy(grid.eta_grid,theLayout)
    KE_val = KE.getKE(grid)
    
    KEResult = comm.reduce(KE_val,op=MPI.SUM, root=0)
    
    if (rank==0):
        assert(abs(KEResult-0.5*((constants.rMax**2-constants.rMin**2)*np.pi*np.pi*constants.R0*2))<1e-7)
