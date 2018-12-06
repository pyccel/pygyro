from mpi4py     import MPI
from math       import pi
from glob       import glob
import numpy    as np
import warnings
import h5py
import os

from ..                     import splines as spl
from ..model.layout         import getLayoutHandler
from ..model.grid           import Grid
from ..model.process_grid   import compute_2d_process_grid
from .initialiser           import initialise_flux_surface, initialise_poloidal, initialise_v_parallel
from .constants             import get_constants, Constants

def setupCylindricalGrid(layout: str, constantFile: str = None, **kwargs):
    """
    Setup using radial topology can be initialised using the following arguments:
    
    Compulsory arguments:
    npts                -- number of points in each direction 
                            (radial, tangential, axial, v parallel)
    layout              -- parallel distribution start configuration
    allocateSaveMemory  -- boolean indicating whether the grid can temporarily save a dataset
    dtype               -- The data type used by the grid
    
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
    
    if (constantFile==None):
        constants = Constants()
    else:
        constants = get_constants(constantFile)
    
    for f in dir(constants):
        val=getattr(constants,f)
        if not callable(val) and f[0]!='_':
            setattr(constants,f,kwargs.pop(f,val))
    
    comm=kwargs.pop('comm',MPI.COMM_WORLD)
    plotThread=kwargs.pop('plotThread',False)
    drawRank=kwargs.pop('drawRank',0)
    allocateSaveMemory=kwargs.pop('allocateSaveMemory',False)
    dtype=kwargs.pop('dtype',float)
    
    for name,value in kwargs.items():
        warnings.warn("{0} is not a recognised parameter for setupCylindricalGrid".format(name))
    
    rank=comm.Get_rank()
    
    if (plotThread):
        layout_comm = comm.Split(rank==drawRank,comm.Get_rank())
    else:
        layout_comm = comm
    
    mpi_size = layout_comm.Get_size()
    
    domain = [ [constants.rMin,constants.rMax], [0,2*pi], [constants.zMin,constants.zMax], [constants.vMin, constants.vMax]]
    degree = constants.splineDegrees
    period = [False, True, True, False]
    
    # Compute breakpoints, knots, spline space and grid points
    nkts      = [n+1+d*(int(p)-1)              for (n,d,p)    in zip( constants.npts,degree, period )]
    breaks    = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots     = [spl.make_knots( b,d,p )       for (b,d,p)    in zip( breaks, degree, period )]
    bsplines  = [spl.BSplines( k,d,p )         for (k,d,p)    in zip(  knots, degree, period )]
    eta_grids = [bspl.greville                 for bspl       in bsplines]

    # Compute 2D grid of processes for the two distributed dimensions in each layout
    nprocs = compute_2d_process_grid( constants.npts, mpi_size )
    
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
    grid = Grid(eta_grids,bsplines,remapper,layout,comm,dtype=dtype,allocateSaveMemory=allocateSaveMemory)
    
    if (layout=='flux_surface'):
        initialise_flux_surface(grid,constants)
    elif (layout=='v_parallel'):
        initialise_v_parallel(grid,constants)
    elif (layout=='poloidal'):
        initialise_poloidal(grid,constants)
    return grid, constants, 0

def setupFromFile(foldername, constantFile: str = None, **kwargs):
    """
    Setup using information from a previous simulation:
    
    Compulsory arguments:
    foldername -- The folder in which the results of the previous simulation
                  are stored
    
    Optional arguments:
    comm        -- MPI communicator. (default MPI.COMM_WORLD)
    plotThread  -- whether there is a thread to be used only for plotting (default False)
    drawRank    -- Thread to be used for plotting (default 0)
    
    timepoint   -- Point in time from which the simulation should resume
                   (default is latest possible)
    
    >>> setupGrid(256,512,32,128,Layout.FIELD_ALIGNED)
    """
    comm=kwargs.pop('comm',MPI.COMM_WORLD)
    
    if (constantFile==None):
        constantFile = "{0}/initParams.json".format(foldername)
    
    constants = get_constants(constantFile)
    
    for f in dir(constants):
        val=getattr(constants,f)
        if not callable(val) and f[0]!='_':
            setattr(constants,f,kwargs.pop(f,val))
    
    plotThread=kwargs.pop('plotThread',False)
    drawRank=kwargs.pop('drawRank',0)
    allocateSaveMemory=kwargs.pop('allocateSaveMemory',False)
    
    rank=comm.Get_rank()
    
    if (plotThread):
        layout_comm = comm.Split(rank==drawRank,comm.Get_rank())
    else:
        layout_comm = comm
    
    mpi_size = layout_comm.Get_size()
    
    domain = [ [constants.rMin,constants.rMax], [0,2*pi], [constants.zMin,constants.zMax], [constants.vMin, constants.vMax]]
    degree = constants.splineDegrees
    period = [False, True, True, False]
    
    # Compute breakpoints, knots, spline space and grid points
    nkts      = [n+1+d*(int(p)-1)              for (n,d,p)    in zip( constants.npts,degree, period )]
    breaks    = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots     = [spl.make_knots( b,d,p )       for (b,d,p)    in zip( breaks, degree, period )]
    bsplines  = [spl.BSplines( k,d,p )         for (k,d,p)    in zip(  knots, degree, period )]
    eta_grids = [bspl.greville                 for bspl       in bsplines]

    # Compute 2D grid of processes for the two distributed dimensions in each layout
    nprocs = compute_2d_process_grid( constants.npts, mpi_size )
    
    # Create dictionary describing layouts
    layouts = {'flux_surface': [0,3,1,2],
               'v_parallel'  : [0,2,1,3],
               'poloidal'    : [3,2,1,0]}

    # Create layout manager
    if (plotThread and rank==drawRank):
        remapper = getLayoutHandler( layout_comm, layouts, nprocs, [[],[],[],[]] )
    else:
        remapper = getLayoutHandler( layout_comm, layouts, nprocs, eta_grids )
    
    if ('timepoint' in kwargs):
        t = kwargs.pop('timepoint')
        filename = "{0}/grid_{1:06}.h5".format(foldername,t)
        assert(os.path.exists(filename))
    else:
        list_of_files = glob("{0}/grid_*".format(foldername))
        if (len(list_of_files)>1):
            filename = max(list_of_files)
            t = int(filename.split('_')[-1].split('.')[0])
        else:
            filename = None
            t = 0
    
    if (filename!=None):
        file = h5py.File(filename,'r')
        dataset=file['/dset']
        order = np.array(dataset.attrs['Layout'])
        my_layout = None
        for name, dims_order in layouts.items():
            if ((dims_order==order).all()):
                my_layout=name
        
        if (my_layout==None):
            raise ArgumentError("The stored layout is not a standard layout")
        
        # Create grid
        grid = Grid(eta_grids,bsplines,remapper,my_layout,comm,allocateSaveMemory=allocateSaveMemory)
        
        layout = grid.getLayout(my_layout)
        slices = tuple([slice(s,e) for s,e in zip(layout.starts,layout.ends)])
        grid._f[:] = dataset[slices]
        
        file.close()
        
        if ('layout' in kwargs):
            desired_layout = kwargs.pop('layout')
            if (desired_layout!=my_layout):
                grid.setLayout(desired_layout)
    else:
        # Create grid
        grid = Grid(eta_grids,bsplines,remapper,layout,comm,dtype=dtype,allocateSaveMemory=allocateSaveMemory)
        
        if (layout=='flux_surface'):
            initialise_flux_surface(grid,constants)
        elif (layout=='v_parallel'):
            initialise_v_parallel(grid,constants)
        elif (layout=='poloidal'):
            initialise_poloidal(grid,constants)
    
    for name,value in kwargs.items():
        warnings.warn("{0} is not a recognised parameter for setupFromFile".format(name))
    
    return grid, constants, t
