from mpi4py import MPI
from math   import pi
import numpy as np

from ..splines.splines import make_knots, BSplines
from .layout import compute_2d_process_grid, Layout, TransposeOperator
from .grid   import Grid

#===============================================================================
# INPUT DATA
#===============================================================================

# Domain limits
rmin = 0.1
rmax = 14.5
vmax = 7.32

domain = [ [rmin,rmax], [0,2*pi], [0,2*pi], [-vmax, vmax]]

# Number of grid points
npts = [256, 512, 32, 128]

# Spline degree
degree = [3, 3, 3, 3]

# Periodicity
period = [False, True, True, False]

#===============================================================================

# Compute breakpoints, knots, spline space and grid points
breaks    = [np.linspace( *lims, num=num ) for lims,num in zip( domain, npts )]
knots     = [make_knots( b,d,p ) for b,d,p in zip( breaks, degree, period)]
bsplines  = [  BSplines( k,d,p ) for k,d,p in zip(  knots, degree, period)]
eta_grids = [bspl.greville for bspl in bsplines]

#===============================================================================
# CREATE DISTRIBUTED PHASE-SPACE GRID
#===============================================================================

########################################
# OPTION 1: Create layouts and transpose
#           operator, then initialize
#           Grid from them
########################################

# MPI info
comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()

# Compute 2D grid of processes for the two distributed dimensions in each layout
nproc1, nproc2 = compute_2d_process_grid( npts, mpi_size )

# Create each layout
layout_a = Layout(
    name       = 'flux_surface',
    nprocs     = [nproc1,1,1,nproc2],
    dims_order = [0,3,1,2],
    eta_grids  = eta_grids
)

layout_b = Layout(
    name       = 'v_parallel',
    nprocs     = [nproc1,1,nproc2,1],
    dims_order = [0,2,1,3],
    eta_grids  = eta_grids
)

layout_c = Layout(
    name       = 'poloidal',
    nprocs     = [1,1,nproc2,nproc1],
    dims_order = [3,2,1,0],
    eta_grids  = eta_grids
)

# Create transpose operators
remapper = TransposeOperator( comm, layout_a, layout_b, layout_c )

# Create grid
phase_space = Grid( layouts, remapper )

########################################
# OPTION 2: Hide layouts and remapper
#           inside Grid
########################################

# Create grid
phase_space = Grid( comm, eta_grids )
