from mpi4py                 import MPI
import numpy                as np
import pytest
from math                   import pi

from ..model.process_grid       import compute_2d_process_grid
from ..model.layout             import LayoutSwapper, getLayoutHandler
from ..model.grid               import Grid
from ..initialisation.setups    import setupCylindricalGrid
from ..                         import splines as spl
from .poisson_solver            import PoissonSolver, DensityFinder
from ..splines.splines              import BSplines, Spline1D
from ..splines.spline_interpolators import SplineInterpolator1D

@pytest.mark.serial
@pytest.mark.parametrize( "deg,npt,eps", [(1,4,0.3),(1,32,0.01),(2,6,0.1),
                                          (2,32,0.1),(3,9,0.03),(3,32,0.02),
                                          (4,10,0.02),(4,40,0.02),(5,14,0.01),
                                          (5,64,0.01)] )
def test_BasicPoissonEquation_Dirichlet_r(deg,npt,eps):
    npts = [npt,8,4]
    domain = [[1,5],[0,2*pi],[0,1]]
    degree = [deg,3,3]
    period = [False,True,False]
    comm = MPI.COMM_WORLD
    
    # Compute breakpoints, knots, spline space and grid points
    nkts     = [n+1+d*(int(p)-1)              for (n,d,p)    in zip( npts,degree, period )]
    breaks   = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots    = [spl.make_knots( b,d,p )       for (b,d,p)    in zip( breaks, degree, period )]
    bsplines = [spl.BSplines( k,d,p )         for (k,d,p)    in zip(  knots, degree, period )]
    eta_grid = [bspl.greville                 for bspl       in bsplines]
    
    layout_poisson = {'mode_solve': [1,2,0]}
    remapper = getLayoutHandler(comm,layout_poisson,[comm.Get_size()],eta_grid)
    
    ps = PoissonSolver(eta_grid,2*deg,bsplines[0],drFactor=0,rFactor=0,ddThetaFactor=0)
    phi=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
    phi_exact=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
    rho=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
    
    r = eta_grid[0]
    
    for i,q in rho.getCoords(0):
        plane = rho.get2DSlice([i])
        plane[:]=1
        plane = phi_exact.get2DSlice([i])
        plane[:] = -0.5*r**2+0.5*(domain[0][0]+domain[0][1])*r-domain[0][0]*domain[0][1]*0.5
    
    ps.solveEquation(phi,rho)
    
    assert((np.abs(phi._f-phi_exact._f)<eps).all())

@pytest.mark.serial
@pytest.mark.parametrize( "deg,npt,eps", [(1,4,1.2),(1,32,0.02),(2,6,0.4),
                                          (2,32,0.006),(3,9,0.1),(3,32,0.004),
                                          (4,10,0.06),(4,40,0.002),(5,14,0.02),
                                          (5,64,0.0005)] )
def test_BasicPoissonEquation_lNeumann(deg,npt,eps):
    npts = [npt,8,4]
    domain = [[1,9],[0,2*pi],[0,1]]
    degree = [deg,2,2]
    period = [False,True,False]
    comm = MPI.COMM_WORLD
    
    # Compute breakpoints, knots, spline space and grid points
    nkts     = [n+1+d*(int(p)-1)              for (n,d,p)    in zip( npts,degree, period )]
    breaks   = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots    = [spl.make_knots( b,d,p )       for (b,d,p)    in zip( breaks, degree, period )]
    bsplines = [spl.BSplines( k,d,p )         for (k,d,p)    in zip(  knots, degree, period )]
    eta_grid = [bspl.greville                 for bspl       in bsplines]
    
    layout_poisson = {'mode_solve': [1,2,0]}
    remapper = getLayoutHandler(comm,layout_poisson,[comm.Get_size()],eta_grid)
    
    ps = PoissonSolver(eta_grid,2*deg,bsplines[0],lBoundary='neumann',drFactor=0,rFactor=0,ddThetaFactor=0)
    phi=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
    phi_exact=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
    rho=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
    
    r = eta_grid[0]
    
    for i,q in rho.getCoords(0):
        plane = rho.get2DSlice([i])
        plane[:]=1
        plane = phi_exact.get2DSlice([i])
        plane[:] = -0.5*r**2+domain[0][0]*r+domain[0][1]**2*0.5-domain[0][0]*domain[0][1]
    
    ps.solveEquation(phi,rho)
    
    assert((np.abs(phi._f-phi_exact._f)<eps).all())

@pytest.mark.serial
@pytest.mark.parametrize( "deg,npt,eps", [(1,4,1.2),(1,32,0.02),(2,6,0.4),
                                          (2,32,0.006),(3,9,0.1),(3,32,0.004),
                                          (4,10,0.06),(4,40,0.002),(5,14,0.02),
                                          (5,64,0.0005)] )
def test_BasicPoissonEquation_rNeumann(deg,npt,eps):
    npts = [npt,8,4]
    domain = [[1,9],[0,2*pi],[0,1]]
    degree = [deg,2,2]
    period = [False,True,False]
    comm = MPI.COMM_WORLD
    
    # Compute breakpoints, knots, spline space and grid points
    nkts     = [n+1+d*(int(p)-1)              for (n,d,p)    in zip( npts,degree, period )]
    breaks   = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots    = [spl.make_knots( b,d,p )       for (b,d,p)    in zip( breaks, degree, period )]
    bsplines = [spl.BSplines( k,d,p )         for (k,d,p)    in zip(  knots, degree, period )]
    eta_grid = [bspl.greville                 for bspl       in bsplines]
    
    layout_poisson = {'mode_solve': [1,2,0]}
    remapper = getLayoutHandler(comm,layout_poisson,[comm.Get_size()],eta_grid)
    
    ps = PoissonSolver(eta_grid,2*deg,bsplines[0],rBoundary='neumann',drFactor=0,rFactor=0,ddThetaFactor=0)
    phi=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
    phi_exact=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
    rho=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
    
    r = eta_grid[0]
    
    for i,q in rho.getCoords(0):
        plane = rho.get2DSlice([i])
        plane[:]=1
        plane = phi_exact.get2DSlice([i])
        plane[:] = -0.5*r**2+domain[0][1]*r+domain[0][0]**2*0.5-domain[0][0]*domain[0][1]
    
    ps.solveEquation(phi,rho)
    
    assert((np.abs(phi._f-phi_exact._f)<eps).all())

@pytest.mark.serial
@pytest.mark.parametrize( "deg,npt,eps", [(1,4,0.3),(1,32,0.01),(2,6,0.1),
                                          (2,32,0.1),(3,9,0.03),(3,32,0.02),
                                          (4,10,0.02),(4,40,0.02),(5,14,0.01),
                                          (5,64,0.01)] )
def test_PoissonEquation_Dirichlet(deg,npt,eps):
    npts = [npt,8,4]
    domain = [[1,5],[0,2*pi],[0,1]]
    degree = [deg,3,3]
    period = [False,True,False]
    comm = MPI.COMM_WORLD
    
    # Compute breakpoints, knots, spline space and grid points
    nkts     = [n+1+d*(int(p)-1)              for (n,d,p)    in zip( npts,degree, period )]
    breaks   = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots    = [spl.make_knots( b,d,p )       for (b,d,p)    in zip( breaks, degree, period )]
    bsplines = [spl.BSplines( k,d,p )         for (k,d,p)    in zip(  knots, degree, period )]
    eta_grid = [bspl.greville                 for bspl       in bsplines]
    
    layout_poisson = {'mode_solve': [1,2,0],
                      'v_parallel': [0,2,1]}
    remapper = getLayoutHandler(comm,layout_poisson,[comm.Get_size()],eta_grid)
    
    grid = setupCylindricalGrid(npts=[*npts,4],layout='v_parallel')
    
    ps = PoissonSolver(eta_grid,2*deg,bsplines[0],drFactor=0,rFactor=0,ddThetaFactor=-1)
    phi=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
    phi_exact=Grid(eta_grid,bsplines,remapper,'v_parallel',comm,dtype=np.complex128)
    rho=Grid(eta_grid,bsplines,remapper,'v_parallel',comm,dtype=np.complex128)
    
    q = eta_grid[1]
    
    for i,r in rho.getCoords(0):
        plane = rho.get2DSlice([i])
        plane[:]=1
        plane = phi_exact.get2DSlice([i])
        plane[:] = (-0.5*r**2+0.5*(domain[0][0]+domain[0][1])*r-domain[0][0]*domain[0][1]*0.5)
    
    r = eta_grid[0]
    
    df = DensityFinder(3,grid.getSpline(3))
    
    ps.getModes(rho)
    
    rho.setLayout('mode_solve')
    
    ps.solveEquation(phi,rho)
    
    phi.setLayout('v_parallel')
    ps.findPotential(phi)
    
    #~ print(np.max(np.abs(phi._f-phi_exact._f)))
    assert((np.abs(phi._f-phi_exact._f)<eps).all())

@pytest.mark.serial
@pytest.mark.parametrize( "deg,npt,eps", [(1,4,0.9),(1,32,0.07),(2,6,0.3),
                                          (2,32,0.05),(3,10,0.2),(3,32,0.04),
                                          (4,10,0.2),(4,40,0.03),(5,14,0.09),
                                          (5,64,0.02)] )
def test_grad(deg,npt,eps):
    npts = [npt,32,4]
    domain = [[1,5],[0,2*pi],[0,1]]
    degree = [deg,3,3]
    period = [False,True,False]
    comm = MPI.COMM_WORLD
    
    # Compute breakpoints, knots, spline space and grid points
    nkts     = [n+1+d*(int(p)-1)              for (n,d,p)    in zip( npts,degree, period )]
    breaks   = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots    = [spl.make_knots( b,d,p )       for (b,d,p)    in zip( breaks, degree, period )]
    bsplines = [spl.BSplines( k,d,p )         for (k,d,p)    in zip(  knots, degree, period )]
    eta_grid = [bspl.greville                 for bspl       in bsplines]
    
    layout_poisson = {'mode_solve': [1,2,0],
                      'v_parallel': [0,2,1]}
    remapper = getLayoutHandler(comm,layout_poisson,[comm.Get_size()],eta_grid)
    
    a=2*pi/(domain[0][1]-domain[0][0])
    
    ps = PoissonSolver(eta_grid,2*deg,bsplines[0],ddrFactor=0,drFactor=1,rFactor=0,ddThetaFactor=0)
    phi=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
    phi_exact=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
    rho=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
    
    r = eta_grid[0]
    
    for i,q in rho.getCoords(0):
        plane = rho.get2DSlice([i])
        plane[:]=a*np.cos(a*(r-domain[0][0]))
        plane = phi_exact.get2DSlice([i])
        plane[:] = np.sin(a*(r-domain[0][0]))
    
    q = eta_grid[1]
    ps.solveEquation(phi,rho)
    
    #~ print(np.max(np.abs(phi._f-phi_exact._f)))
    assert((np.abs(phi._f-phi_exact._f)<eps).all())

@pytest.mark.serial
@pytest.mark.parametrize( "deg,npt,eps", [(1,32,0.08),(1,256,0.009),(2,32,0.06),
                                          (2,256,0.006),(3,32,0.04),(3,256,0.005),
                                          (4,32,0.04),(4,256,0.004),(5,32,0.03),
                                          (5,256,0.003)] )
def test_grad_r(deg,npt,eps):
    npts = [npt,32,4]
    domain = [[1,8],[0,2*pi],[0,1]]
    degree = [deg,3,3]
    period = [False,True,False]
    comm = MPI.COMM_WORLD
    
    # Compute breakpoints, knots, spline space and grid points
    nkts     = [n+1+d*(int(p)-1)              for (n,d,p)    in zip( npts,degree, period )]
    breaks   = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots    = [spl.make_knots( b,d,p )       for (b,d,p)    in zip( breaks, degree, period )]
    bsplines = [spl.BSplines( k,d,p )         for (k,d,p)    in zip(  knots, degree, period )]
    eta_grid = [bspl.greville                 for bspl       in bsplines]
    
    layout_poisson = {'mode_solve': [1,2,0],
                      'v_parallel': [0,2,1]}
    remapper = getLayoutHandler(comm,layout_poisson,[comm.Get_size()],eta_grid)
    
    a=2*pi/(domain[0][1]-domain[0][0])
    r = eta_grid[0]
    
    ps = PoissonSolver(eta_grid,2*deg,bsplines[0],ddrFactor=0,drFactor=r,rFactor=0,ddThetaFactor=0)
    phi=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
    phi_exact=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
    rho=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
    
    for i,q in rho.getCoords(0):
        plane = rho.get2DSlice([i])
        plane[:]=a*np.cos(a*(r-domain[0][0]))*r
        plane = phi_exact.get2DSlice([i])
        plane[:] = np.sin(a*(r-domain[0][0]))
    
    q = eta_grid[1]
    ps.solveEquation(phi,rho)
    
    #~ print(np.max(np.abs(phi._f-phi_exact._f)))
    assert((np.abs(phi._f-phi_exact._f)<eps).all())

@pytest.mark.parallel
@pytest.mark.parametrize( "deg,npt,eps", [(1,32,0.07),(2,32,0.05),(3,32,0.04),
                                          (4,40,0.03),(5,64,0.02)] )
def test_grad_withFFT(deg,npt,eps):
    npts = [npt,32,4]
    domain = [[1,5],[0,2*pi],[0,1]]
    degree = [deg,3,3]
    period = [False,True,False]
    comm = MPI.COMM_WORLD
    
    # Compute breakpoints, knots, spline space and grid points
    nkts     = [n+1+d*(int(p)-1)              for (n,d,p)    in zip( npts,degree, period )]
    breaks   = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots    = [spl.make_knots( b,d,p )       for (b,d,p)    in zip( breaks, degree, period )]
    bsplines = [spl.BSplines( k,d,p )         for (k,d,p)    in zip(  knots, degree, period )]
    eta_grid = [bspl.greville                 for bspl       in bsplines]
    
    layout_poisson = {'mode_solve': [1,2,0],
                      'v_parallel': [0,2,1]}
    remapper = getLayoutHandler(comm,layout_poisson,[comm.Get_size()],eta_grid)
    
    a=2*pi/(domain[0][1]-domain[0][0])
    
    ps = PoissonSolver(eta_grid,2*deg,bsplines[0],ddrFactor=0,drFactor=1,rFactor=0,ddThetaFactor=0)
    phi=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
    phi_exact=Grid(eta_grid,bsplines,remapper,'v_parallel',comm,dtype=np.complex128)
    rho=Grid(eta_grid,bsplines,remapper,'v_parallel',comm,dtype=np.complex128)
    
    for i,r in rho.getCoords(0):
        plane = rho.get2DSlice([i])
        plane[:]=a*np.cos(a*(r-domain[0][0]))
        plane = phi_exact.get2DSlice([i])
        plane[:] = np.sin(a*(r-domain[0][0]))
    
    r = eta_grid[0]
    q = eta_grid[1]
    
    ps.getModes(rho)
    
    rho.setLayout('mode_solve')
    
    ps.solveEquation(phi,rho)
    
    phi.setLayout('v_parallel')
    ps.findPotential(phi)
    
    #~ print(np.max(np.abs(phi._f-phi_exact._f)))
    assert((np.abs(phi._f-phi_exact._f)<eps).all())

@pytest.mark.serial
@pytest.mark.parametrize( "deg,npt,eps", [(1,4,2),(1,32,0.2),(1,64,0.03),(2,6,1.1),
                                          (2,32,0.03),(3,10,0.3),(3,32,0.02),
                                          (4,10,0.3),(4,40,0.008),(5,14,0.09),
                                          (5,64,0.003)] )
def test_Sin_r_Sin_theta(deg,npt,eps):
    npts = [npt,64,4]
    domain = [[1,5],[0,2*pi],[0,1]]
    degree = [deg,3,3]
    period = [False,True,False]
    comm = MPI.COMM_WORLD
    
    # Compute breakpoints, knots, spline space and grid points
    nkts     = [n+1+d*(int(p)-1)              for (n,d,p)    in zip( npts,degree, period )]
    breaks   = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots    = [spl.make_knots( b,d,p )       for (b,d,p)    in zip( breaks, degree, period )]
    bsplines = [spl.BSplines( k,d,p )         for (k,d,p)    in zip(  knots, degree, period )]
    eta_grid = [bspl.greville                 for bspl       in bsplines]
    
    layout_poisson = {'mode_solve': [1,2,0],
                      'v_parallel': [0,2,1]}
    remapper = getLayoutHandler(comm,layout_poisson,[comm.Get_size()],eta_grid)
    
    a=2*pi/(domain[0][1]-domain[0][0])
    
    ps = PoissonSolver(eta_grid,2*deg,bsplines[0],ddrFactor=1,drFactor=1,rFactor=0,ddThetaFactor=-a*a)
    phi=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
    phi_exact=Grid(eta_grid,bsplines,remapper,'v_parallel',comm,dtype=np.complex128)
    rho=Grid(eta_grid,bsplines,remapper,'v_parallel',comm,dtype=np.complex128)
    
    q = eta_grid[1]
    
    for i,r in rho.getCoords(0):
        plane = rho.get2DSlice([i])
        plane[:] = a*np.cos(a*(r-domain[0][0]))*np.sin(q)
        plane = phi_exact.get2DSlice([i])
        plane[:] = np.sin(a*(r-domain[0][0]))*np.sin(q)
    
    r = eta_grid[0]
    
    ps.getModes(rho)
    
    rho.setLayout('mode_solve')
    
    ps.solveEquation(phi,rho)
    
    phi.setLayout('v_parallel')
    ps.findPotential(phi)
    
    #~ print(np.max(np.abs(phi._f-phi_exact._f)))
    assert((np.abs(phi._f-phi_exact._f)<eps).all())

@pytest.mark.serial
@pytest.mark.parametrize( "deg,npt", [(1,4),(1,32),(1,64),(2,6),
                                      (2,32),(3,10),(3,32),
                                      (4,10),(4,40),(5,14),
                                      (5,64)] )
def test_ddTheta(deg,npt):
    eps=1e-12
    npts = [8,npt,5]
    domain = [[1,5],[0,2*pi],[0,1]]
    degree = [deg,3,3]
    period = [False,True,False]
    comm = MPI.COMM_WORLD
    
    # Compute breakpoints, knots, spline space and grid points
    nkts     = [n+1+d*(int(p)-1)              for (n,d,p)    in zip( npts,degree, period )]
    breaks   = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots    = [spl.make_knots( b,d,p )       for (b,d,p)    in zip( breaks, degree, period )]
    bsplines = [spl.BSplines( k,d,p )         for (k,d,p)    in zip(  knots, degree, period )]
    eta_grid = [bspl.greville                 for bspl       in bsplines]
    
    layout_poisson = {'mode_solve': [1,2,0],
                      'v_parallel': [0,2,1]}
    remapper = getLayoutHandler(comm,layout_poisson,[comm.Get_size()],eta_grid)
    
    ps = PoissonSolver(eta_grid,2*deg,bsplines[0],ddrFactor=0,drFactor=0,
                        rFactor=1,ddThetaFactor=-1,lBoundary='neumann',rBoundary='neumann')
    phi=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
    phi_exact=Grid(eta_grid,bsplines,remapper,'v_parallel',comm,dtype=np.complex128)
    rho=Grid(eta_grid,bsplines,remapper,'v_parallel',comm,dtype=np.complex128)
    
    q = eta_grid[1]
    
    for i,r in rho.getCoords(0):
        plane = rho.get2DSlice([i])
        plane[:] = 2*np.sin(q)
        plane = phi_exact.get2DSlice([i])
        plane[:] = np.sin(q)
    
    r = eta_grid[0]
    
    ps.getModes(rho)
    
    rho.setLayout('mode_solve')
    
    ps.solveEquation(phi,rho)
    
    phi.setLayout('v_parallel')
    
    ps.findPotential(phi)
    
    assert((np.abs(phi._f-phi_exact._f)<eps).all())

@pytest.mark.serial
@pytest.mark.parametrize( "deg,npt", [(1,4),(1,32),(1,64),(2,6),
                                      (2,32),(3,10),(3,32),
                                      (4,10),(4,40),(5,14),
                                      (5,64)] )
def test_phi(deg,npt):
    eps=1e-12
    npts = [256,npt,4]
    domain = [[1,5],[0,2*pi],[0,1]]
    degree = [deg,3,3]
    period = [False,True,False]
    comm = MPI.COMM_WORLD
    
    # Compute breakpoints, knots, spline space and grid points
    nkts     = [n+1+d*(int(p)-1)              for (n,d,p)    in zip( npts,degree, period )]
    breaks   = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots    = [spl.make_knots( b,d,p )       for (b,d,p)    in zip( breaks, degree, period )]
    bsplines = [spl.BSplines( k,d,p )         for (k,d,p)    in zip(  knots, degree, period )]
    eta_grid = [bspl.greville                 for bspl       in bsplines]
    
    layout_poisson = {'mode_solve': [1,2,0],
                      'v_parallel': [0,2,1]}
    remapper = getLayoutHandler(comm,layout_poisson,[comm.Get_size()],eta_grid)
    
    a=2*pi/(domain[0][1]-domain[0][0])
    
    ps = PoissonSolver(eta_grid,2*deg,bsplines[0],ddrFactor=0,drFactor=0,rFactor=1,ddThetaFactor=0,
                        lBoundary='neumann',rBoundary='neumann')
    phi=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
    phi_exact=Grid(eta_grid,bsplines,remapper,'v_parallel',comm,dtype=np.complex128)
    rho=Grid(eta_grid,bsplines,remapper,'v_parallel',comm,dtype=np.complex128)
    
    q = eta_grid[1]
    
    for i,r in rho.getCoords(0):
        plane = rho.get2DSlice([i])
        plane[:] = np.sin(q)
        plane = phi_exact.get2DSlice([i])
        plane[:] = np.sin(q)
    
    r = eta_grid[0]
    
    ps.getModes(rho)
    
    rho.setLayout('mode_solve')
    
    ps.solveEquation(phi,rho)
    
    phi.setLayout('v_parallel')
    ps.findPotential(phi)
    
    #~ print(np.max(np.abs(phi._f-phi_exact._f)))
    assert((np.abs(phi._f-phi_exact._f)<eps).all())

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
    phi = Grid(grid.eta_grid[:3],grid.getSpline(slice(0,3)),remapper,'mode_solve',comm,dtype=np.complex128)
    
    df = DensityFinder(3,grid.getSpline(3))
    
    df.getRho(grid,rho)
    
    psolver = PoissonSolver(grid.eta_grid,6,rho.getSpline(0))
    
    psolver.getModes(rho)
    
    rho.setLayout('mode_solve')
    
    psolver.solveEquation(phi,rho)
    
    phi.setLayout('v_parallel')
    
    psolver.findPotential(phi)
