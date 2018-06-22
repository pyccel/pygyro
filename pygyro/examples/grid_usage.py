from mpi4py import MPI
from math   import pi, tanh, exp, sqrt
import numpy as np

from ..splines.splines      import make_knots, BSplines
from ..model.process_grid   import compute_2d_process_grid
from ..model.layout         import getLayoutHandler
from ..model.grid           import Grid

#===============================================================================
# INPUT DATA
#===============================================================================

# Domain limits
rmin = 0.1
rmax = 14.5
vmax = 7.32

domain = [ [rmin,rmax], [0,2*pi], [0,2*pi], [-vmax, vmax]]

# Number of grid points
npts = [128, 256, 16, 64]

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

# MPI info
comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()

# Compute 2D grid of processes for the two distributed dimensions in each layout
nprocs = compute_2d_process_grid( npts, mpi_size )

# Create dictionary describing layouts
layouts = {'flux_surface': [0,3,1,2],
           'v_parallel'  : [0,2,1,3],
           'poloidal'    : [3,2,1,0]}

# Create layout manager
remapper = getLayoutHandler( comm, layouts, nprocs, eta_grids )

# Create grid
phase_space = Grid(eta_grids,remapper,'flux_surface')

# Iterate over all elements using a 2D slice (theta,z)
for i,r in phase_space.getCoords(0):
    for j,v in phase_space.getCoords(1):
        # Get surface
        FluxSurface = phase_space.get2DSlice([i,j])
        # Get coordinate values
        theta = phase_space.getCoordVals(2)
        z = phase_space.getCoordVals(3)
        
        # transpose theta to use ufuncs
        theta = theta.reshape(theta.size,1)
        
        # set phase_space to perturbation
        FluxSurface[:] = exp((r-7.3)**2/8.0)*np.cos(15.0*theta-11.0*z/239.8081535)

# Change to a different layout
phase_space.setLayout('v_parallel')

# Iterate over all elements using a 1D slice (v_parallel)
for i,r in phase_space.getCoords(0):
    for j,z in phase_space.getCoords(1):
        for k,theta in phase_space.getCoords(2):
            # Get surface
            VParSurface = phase_space.get1DSlice([i,j,k])
            # Get coordinate values
            vPar = phase_space.getCoordVals(3)
            
            # set phase_space to equilibrium
            n0=0.9923780370404612*exp(-0.055*2.9*tanh((r-7.3)/2.9))
            Ti=exp(-0.27586*1.45*tanh((r-7.3)/1.45))
            VParSurface[:] = n0*np.exp(-0.5*vPar*vPar/Ti)/sqrt(2*pi*Ti)
