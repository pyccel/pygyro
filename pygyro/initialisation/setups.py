from mpi4py import MPI
from math import pi
import numpy as np
import warnings

from ..                     import splines as spl
from ..model.layout         import getLayoutHandler
from ..model.grid           import Grid
from ..model.process_grid   import compute_2d_process_grid
from .initialiser           import initialise_flux_surface, initialise_poloidal, initialise_v_parallel
from .                      import constants

def setupCylindricalGrid(npts: list, layout: str, **kwargs):
    """
    Setup using radial topology can be initialised using the following arguments:
    
    Compulsory arguments:
    npts   -- number of points in each direction 
              (radial, tangential, axial, v parallel)
    layout -- parallel distribution start configuration
    
    Optional arguments:
    rMin   -- minimum radius, a float. (default constants.rMin)
    rMax   -- maximum radius, a float. (default constants.rMax)
    zMin   -- minimum value in the z direction, a float. (default constants.zMin)
    zMax   -- maximum value in the z direction, a float. (default constants.zMax)
    vMax   -- maximum velocity, a float. (default constants.vMax)
    vMin   -- minimum velocity, a float. (default -vMax)
    
    m  -- (default constants.m)
    n  -- (default constants.n)
    
    rDegree     -- degree of splines in the radial direction. (default 3)
    thetaDegree -- degree of splines in the tangential direction. (default 3)
    zDegree     -- degree of splines in the axial direction. (default 3)
    vDegree     -- degree of splines in the v parallel direction. (default 3)
    
    eps         -- perturbation size
    
    comm        -- MPI communicator. (default MPI.COMM_WORLD)
    plotThread  -- whether there is a thread to be used only for plotting (default False)
    drawRank    -- Thread to be used for plotting (default 0)
    
    >>> setupGrid(256,512,32,128,Layout.FIELD_ALIGNED)
    """
    rMin=kwargs.pop('rMin',constants.rMin)
    rMax=kwargs.pop('rMax',constants.rMax)
    constants.rp = 0.5*(rMin + rMax)
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
    eps=kwargs.pop('eps',constants.eps)
    comm=kwargs.pop('comm',MPI.COMM_WORLD)
    plotThread=kwargs.pop('plotThread',False)
    drawRank=kwargs.pop('drawRank',0)
    
    for name,value in kwargs.items():
        warnings.warn("{0} is not a recognised parameter for setupCylindricalGrid".format(name))
    
    rank=comm.Get_rank()
    
    if (plotThread):
        layout_comm = comm.Split(rank==drawRank,comm.Get_rank())
    else:
        layout_comm = comm
    
    mpi_size = layout_comm.Get_size()
    
    domain = [ [rMin,rMax], [0,2*pi], [zMin,zMax], [vMin, vMax]]
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
    if (plotThread and rank==drawRank):
        remapper = getLayoutHandler( layout_comm, layouts, nprocs, [[],[],[],[]] )
    else:
        remapper = getLayoutHandler( layout_comm, layouts, nprocs, eta_grids )
    
    # Create grid
    grid = Grid(eta_grids,bsplines,remapper,layout,comm)
    
    if (layout=='flux_surface'):
        initialise_flux_surface(grid,m,n,eps)
    elif (layout=='v_parallel'):
        initialise_v_parallel(grid,m,n,eps)
    elif (layout=='poloidal'):
        initialise_poloidal(grid,m,n,eps)
    return grid
