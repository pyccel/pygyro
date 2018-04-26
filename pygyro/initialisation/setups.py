from mpi4py import MPI
from math import pi
import numpy as np

from ..                     import splines as spl
from ..model.layout         import LayoutManager
from ..model.grid           import Grid
from ..model.process_grid   import compute_2d_process_grid
from .initialiser           import initialise_flux_surface, initialise_poloidal, initialise_v_parallel
from .                      import constants

def setupCylindricalGrid(nr: int, ntheta: int, nz: int, nv: int, layout: str, **kwargs):
    """
    Setup using radial topology can be initialised using the following arguments:
    
    Compulsory arguments:
    nr     -- number of points in the radial direction
    ntheta -- number of points in the tangential direction
    nz     -- number of points in the axial direction
    nv     -- number of velocities for v perpendicular
    layout -- parallel distribution configuration
    
    Optional arguments:
    rMin   -- minimum radius, a float. (default constants.rMin)
    rMax   -- maximum radius, a float. (default constants.rMax)
    zMin   -- minimum value in the z direction, a float. (default constants.zMin)
    zMax   -- maximum value in the z direction, a float. (default constants.zMax)
    vMax   -- maximum velocity, a float. (default constants.vMax)
    vMin   -- minimum velocity, a float. (default -vMax)
    degree -- degree of splines. (default 3)
    
    >>> setupGrid(256,512,32,128,Layout.FIELD_ALIGNED)
    """
    rMin=kwargs.pop('rMin',constants.rMin)
    rMax=kwargs.pop('rMax',constants.rMax)
    zMin=kwargs.pop('zMin',constants.zMin)
    zMax=kwargs.pop('zMax',constants.zMax)
    vMax=kwargs.pop('vMax',constants.vMax)
    vMin=kwargs.pop('vMin',-vMax)
    m=kwargs.pop('m',constants.m)
    n=kwargs.pop('n',constants.n)
    rDegree=kwargs.pop('rDegree',3)
    qDegree=kwargs.pop('thetaDegree',3)
    zDegree=kwargs.pop('zDegree',3)
    vDegree=kwargs.pop('vDegree',3)
    comm=kwargs.pop('comm',MPI.COMM_WORLD)
    
    mpi_size = comm.Get_size()
    
    domain = [ [rMin,rMax], [zMin,zMax], [0,2*pi], [vMin, vMax]]
    npts   = [nr, ntheta, nz, nv]
    degree = [rDegree, qDegree, zDegree, vDegree]
    period = [False, True, True, False]
    
    # Compute breakpoints, knots, spline space and grid points
    nkts      = [n+1+d*(int(p)-1)              for (n,d,p)    in zip( npts,degree, period )]
    breaks    = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots     = [spl.make_knots( b,d,p )       for (b,d,p)    in zip( breaks, degree, period )]
    bsplines  = [spl.BSplines( k,d,p )         for (k,d,p)    in zip(  knots, degree, period )]
    eta_grids = [bspl.greville                 for bspl       in bsplines]

    # Compute 2D grid of processes for the two distributed dimensions in each layout
    nprocs = compute_2d_process_grid( npts, mpi_size )
    
    # Create dictionary describing layouts
    layouts = {'flux_surface': [0,3,1,2],
               'v_parallel'  : [0,2,1,3],
               'poloidal'    : [3,2,1,0]}

    # Create layout manager
    remapper = LayoutManager( comm, layouts, nprocs, eta_grids )

    # Create grid
    grid = Grid(eta_grids,remapper,layout)
    
    if (layout=='flux_surface'):
        initialise_flux_surface(grid,m,n)
    elif (layout=='v_parallel'):
        initialise_v_parallel(grid,m,n)
    elif (layout=='poloidal'):
        initialise_poloidal(grid,m,n)
    return grid
