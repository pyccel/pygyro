from mpi4py                 import MPI
import numpy                as np
import pytest
from math                   import pi
from scipy.integrate        import trapz
from numpy.polynomial.legendre      import leggauss

import matplotlib.pyplot    as plt
from matplotlib             import rc

from ..model.process_grid           import compute_2d_process_grid
from ..model.layout                 import LayoutSwapper, getLayoutHandler
from ..model.grid                   import Grid
from ..initialisation               import mod_initialiser_funcs as initialiser
from ..initialisation.constants     import get_constants
from ..                             import splines as spl
from .poisson_solver                import DiffEqSolver, DensityFinder
from ..splines.splines              import BSplines, Spline1D
from ..splines.spline_interpolators import SplineInterpolator1D

@pytest.mark.serial
def test_QuasiNeutralityEquation_pointConverge():
    font = {'size'   : 16}

    rc('font', **font)
    rc('text', usetex=True)
    deg = 3
    npt = 128

    npts = [npt,64,4]
    domain = [[1,15],[0,2*pi],[0,1]]
    degree = [deg,3,3]
    period = [False,True,False]
    comm = MPI.COMM_WORLD

    # Compute breakpoints, knots, spline space and grid points
    nkts     = [n+1+d+d*(int(p)-1)              for (n,d,p)    in zip( npts,degree, period )]
    breaks   = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots    = [spl.make_knots( b,d,p )       for (b,d,p)    in zip( breaks, degree, period )]
    bsplines = [spl.BSplines( k,d,p )         for (k,d,p)    in zip(  knots, degree, period )]
    eta_grid = [bspl.greville                 for bspl       in bsplines]

    constants = get_constants('testSetups/iota0.json')

    layout_poisson = {'mode_solve': [1,2,0],
                      'v_parallel': [0,2,1]}
    remapper = getLayoutHandler(comm,layout_poisson,[comm.Get_size()],eta_grid)

    mVals = np.fft.fftfreq(eta_grid[1].size,1/eta_grid[1].size)

    ps = DiffEqSolver(100,bsplines[0],eta_grid[0].size,
            eta_grid[1].size,lNeumannIdx=mVals,
            ddrFactor = lambda r:-1,
            drFactor = lambda r:-( 1/r + initialiser.n0derivNormalised(r,constants.kN0,constants.rp,constants.deltaRN0) ),
            rFactor = lambda r:1/initialiser.Te(r,constants.CTe,constants.kTe,constants.deltaRTe,constants.rp),
            ddThetaFactor = lambda r:-1/r**2)
    phi=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
    phi_exact=Grid(eta_grid,bsplines,remapper,'v_parallel',comm,dtype=np.complex128)
    rho=Grid(eta_grid,bsplines,remapper,'v_parallel',comm,dtype=np.complex128)

    a=1.5*pi/(domain[0][1]-domain[0][0])
    q = eta_grid[1]

    for i,r in rho.getCoords(0):
        rArg = a*(r-domain[0][0])
        plane = rho.get2DSlice([i])
        plane[:] = -12*np.cos(rArg)**2*np.sin(rArg)**2*a*a*np.sin(q)**3 \
                   + 4*np.cos(rArg)**4                *a*a*np.sin(q)**3 \
                   + (1/r - constants.kN0*(1-np.tanh((r-constants.rp)/constants.deltaRN0)**2)) * \
                   4 * np.cos(rArg)**3*np.sin(rArg)*a*np.sin(q)**3 \
                   + np.cos(rArg)**4*np.sin(q)**3 \
                   / initialiser.Te(r,constants.CTe,constants.kTe,constants.deltaRTe,constants.rp) \
                   - 6 * np.cos(rArg)**4*np.sin(q)*np.cos(q)**2/r**2 \
                   + 3 * np.cos(rArg)**4*np.sin(q)**3/r**2
        plane = phi_exact.get2DSlice([i])
        plane[:] = np.cos(rArg)**4*np.sin(q)**3

    ps.getModes(rho)

    rho.setLayout('mode_solve')

    ps.solveEquation(phi,rho)

    phi.setLayout('v_parallel')
    ps.findPotential(phi)

    r = eta_grid[0]

    q = [*q, 2*pi]
    phiExact = np.concatenate([phi_exact._f[:,0,:],phi_exact._f[:,0,0,None]],axis=1)
    Error = np.concatenate([phi_exact._f[:,0,:]-phi._f[:,0,:],(phi_exact._f[:,0,0]-phi._f[:,0,0])[:,None]],axis=1)

    font = {'size'   : 16}
    plt.rc('font', **font)
    plt.rc('text', usetex=True)

    splitFact = 10

    plt.figure()
    ax = plt.subplot2grid((1, splitFact), (0, 0), colspan=splitFact-1, projection='polar')
    line1 = ax.pcolormesh(q,r,np.real(phiExact))

    axc = plt.subplot2grid((1, splitFact), (0, splitFact-1))

    plt.colorbar(line1, cax = axc )
    plt.tight_layout()

    fig = plt.figure()
    ax = plt.subplot2grid((1, splitFact), (0, 0), colspan=splitFact-1, projection='polar')
    line1 = plt.pcolormesh(q,r,np.real(Error))

    axc = plt.subplot2grid((1, splitFact), (0, splitFact-1))
    plt.colorbar(line1,aspect=0, cax=axc)
    plt.tight_layout()

    plt.show()
