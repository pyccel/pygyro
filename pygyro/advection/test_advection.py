from mpi4py                 import MPI
import pytest
from scipy.integrate        import trapz
import numpy                as np

from ..initialisation.setups                    import setupCylindricalGrid
from ..model.layout                             import Layout
from ..initialisation.mod_initialiser_funcs     import fEq
from .advection                                 import FluxSurfaceAdvection, PoloidalAdvection, VParallelAdvection, ParallelGradient
from ..                                         import splines as spl
from ..initialisation.constants                 import get_constants, Constants

def gauss(x):
    return np.exp(-x**2/4)


@pytest.mark.serial
@pytest.mark.parametrize( "fact,dt", [(10,1),(10,0.1), (5,1)] )
def test_fluxSurfaceAdvection(fact,dt):
    npts = [30,20]
    eta_vals = [np.linspace(0,1,4),np.linspace(0,2*np.pi,npts[0],endpoint=False),
                np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,1,4)]

    N = 10

    f_vals = np.ndarray(npts)

    domain    = [ [0,2*np.pi], [0,20] ]
    nkts      = [n+1                           for n          in npts ]
    breaks    = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots     = [spl.make_knots( b,3,True )    for b          in breaks]
    bsplines  = [spl.BSplines( k,3,True )      for k          in knots]
    eta_grids = [bspl.greville                 for bspl       in bsplines]

    c=2

    eta_vals[1]=eta_grids[0]
    eta_vals[2]=eta_grids[1]
    eta_vals[3][0]=c

    layout = Layout('flux',[1],[0,3,1,2],eta_vals,[0])

    constants = Constants()

    fluxAdv = FluxSurfaceAdvection(eta_vals, bsplines, layout, dt, constants)

    f_vals[:,:] = np.sin(eta_vals[2]*np.pi/fact)
    f_end = np.sin((eta_vals[2]-c*dt*N)*np.pi/fact)

    for n in range(N):
        fluxAdv.step(f_vals,0)

    assert(np.max(np.abs(f_vals-f_end))<1e-4)

@pytest.mark.serial
@pytest.mark.parametrize( "nptZ,dt,err", [(32,1.0,1.5e-2),(64,0.5,3e-4),(128,0.25,1e-5)] )
def test_fluxSurfaceAdvectionAligned(nptZ,dt,err):
    npts = [nptZ,nptZ]

    constants = Constants()
    constants.iotaVal=0.8
    constants.n=-11

    eta_vals = [np.linspace(0,1,4),np.linspace(0,2*np.pi,npts[0],endpoint=False),
                np.linspace(0,2*np.pi*constants.R0,npts[1],endpoint=False),np.linspace(0,1,4)]

    N = 10

    f_vals = np.ndarray(npts)

    domain    = [ [0,2*np.pi], [0,2*np.pi*constants.R0] ]
    nkts      = [n+1                           for n          in npts ]
    breaks    = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots     = [spl.make_knots( b,3,True )    for b          in breaks]
    bsplines  = [spl.BSplines( k,3,True )      for k          in knots]
    eta_grids = [bspl.greville                 for bspl       in bsplines]

    c=2

    eta_vals[1]=eta_grids[0]
    eta_vals[2]=eta_grids[1]
    eta_vals[3][0]=c

    layout = Layout('flux',[1],[0,3,1,2],eta_vals,[0])

    fluxAdv = FluxSurfaceAdvection(eta_vals, bsplines, layout, dt, constants)

    m, n = (5,-4)
    theta = eta_grids[0]
    phi = eta_grids[1]*2*np.pi/domain[1][1]
    f_vals[:,:] = np.sin( m*theta[:,None] + n*phi[None,:] )

    #~ f_vals[:,:] = np.sin(eta_vals[2]*np.pi/fact)
    f_end = f_vals.copy()

    for n in range(N):
        fluxAdv.step(f_vals,0)
    #~ print(np.max(np.abs(f_vals-f_end)))
    assert(np.max(np.abs(f_vals-f_end))<err)

@pytest.mark.serial
@pytest.mark.parametrize( "function,N", [(gauss,10),(gauss,20),(gauss,30)] )
def test_vParallelAdvection(function,N):
    npts = 50
    f = np.empty(npts)

    constants = Constants()

    dt=0.1
    c=2.0

    nkts      = npts-2
    breaks    = np.linspace( -5, 5, num=nkts )
    knots     = spl.make_knots( breaks,3,False )
    spline    = spl.BSplines( knots,3,False )
    x         = spline.greville

    r = 4
    fEdge = fEq(r,x[0],constants.CN0,constants.kN0,constants.deltaRN0,
                constants.rp,constants.CTi,constants.kTi,constants.deltaRTi)
    assert(fEq(r,x[0],constants.CN0,constants.kN0,constants.deltaRN0,
                constants.rp,constants.CTi,constants.kTi,constants.deltaRTi)
        == fEq(r,x[-1],constants.CN0,constants.kN0,constants.deltaRN0,
                constants.rp,constants.CTi,constants.kTi,constants.deltaRTi))

    f = function(x)+fEdge

    vParAdv = VParallelAdvection([0,0,0,x], spline, constants, 'null')

    for i in range(N):
        vParAdv.step(f,dt,c,r)

    fEnd = np.empty(npts)

    for i in range(npts):
        if ((x[i]-c*dt*N)<x[0]):
            fEnd[i]=fEq(r,(x[i]-c*dt*N),constants.CN0,constants.kN0,constants.deltaRN0,
                constants.rp,constants.CTi,constants.kTi,constants.deltaRTi)
        else:
            fEnd[i]=fEdge+function(x[i]-c*dt*N)

    assert(max(abs(f-fEnd))<2e-3)

@pytest.mark.serial
@pytest.mark.parametrize( "N", [10,20,30] )
def test_vParallelAdvectionPeriodic(N):
    npts = 50
    f = np.empty(npts)

    dt=0.1
    c=2.0

    nkts      = npts-2
    breaks    = np.linspace( -5, 5, num=nkts )
    knots     = spl.make_knots( breaks,3,False )
    spline    = spl.BSplines( knots,3,False )
    x         = spline.greville

    r = 4

    f = gauss(x)

    constants = Constants()

    vParAdv = VParallelAdvection([0,0,0,x], spline, constants, 'periodic')

    for i in range(N):
        vParAdv.step(f,dt,c,r)

    fEnd = np.empty(npts)

    for i in range(npts):
        fEnd[i]=gauss((x[i]-c*dt*N+5)%10 -5)

    assert(max(abs(f-fEnd))<2e-3)

def Phi(r,theta, omega, xc, yc):
    return omega * (r*r/2 - r*np.sin(theta)*yc - r * np.cos(theta)*xc)

def initConditions(r,theta):
    a=2
    factor = np.pi/a/2
    r=np.sqrt((r-8)**2+8*(theta-np.pi)**2)

    if (r<=a):
        return np.cos(r*factor)**4
    else:
        return 0.0

initConds = np.vectorize(initConditions, otypes=[np.float])

@pytest.mark.serial
@pytest.mark.parametrize( "dt,v,xc,yc", [(1,-5,0,0), (0.1,5,1,0), (0.1, 2, 2, 1)])
def test_poloidalAdvection(dt,v,xc,yc):

    npts = [64,64]
    eta_vals = [np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,2*np.pi,npts[0],endpoint=False),
                np.linspace(0,1,4),np.linspace(0,1,4)]

    N = int(1/dt)

    f_vals = np.ndarray([npts[1],npts[0]])
    final_f_vals = np.ndarray([npts[1],npts[0]])

    deg = 3
    omega = 1

    domain    = [ [1,14.5], [0,2*np.pi] ]
    periodic  = [ False, True ]
    nkts      = [n+1+deg*(int(p)-1)            for (n,p)      in zip( npts, periodic )]
    breaks    = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots     = [spl.make_knots( b,deg,p )     for b,p        in zip(breaks,periodic)]
    bsplines  = [spl.BSplines( k,deg,p )       for k,p        in zip(knots,periodic)]
    eta_grids = [bspl.greville                 for bspl       in bsplines]

    eta_vals[0]=eta_grids[0]
    eta_vals[1]=eta_grids[1]

    constants = Constants()

    polAdv = PoloidalAdvection(eta_vals, bsplines[::-1], constants, True)

    phi = spl.Spline2D(bsplines[1],bsplines[0])
    phiVals = np.empty([npts[1],npts[0]])
    phiVals[:] = Phi(eta_vals[0],np.atleast_2d(eta_vals[1]).T, omega, xc, yc)
    interp = spl.SplineInterpolator2D(bsplines[1],bsplines[0])

    interp.compute_interpolant(phiVals,phi)

    f_vals[:,:] = initConds(eta_vals[0],np.atleast_2d(eta_vals[1]).T)

    for n in range(N):
        polAdv.step(f_vals[:,:],dt,phi,v)

    x0 = polAdv._points[1] * np.cos(polAdv._shapedQ)
    y0 = polAdv._points[1] * np.sin(polAdv._shapedQ)

    x = xc + (x0 - xc) * np.cos(omega * -dt * N) - (y0 - yc) * np.sin(omega * -dt * N)
    y = yc + (x0 - xc) * np.sin(omega * -dt * N) + (y0 - yc) * np.cos(omega * -dt * N)

    finalPts = ( np.ndarray([npts[1],npts[0]]), np.ndarray([npts[1],npts[0]]))
    finalPts[0][:] = np.mod(np.arctan2(y, x), 2 * np.pi)
    finalPts[1][:] = np.sqrt(x * x + y * y)
    final_f_vals[:,:] = initConds(finalPts[1],finalPts[0])

    l2=np.sqrt(trapz(trapz((f_vals-final_f_vals)**2,eta_grids[1],axis=0)*eta_grids[0],eta_grids[0]))
    assert(l2<0.2)

@pytest.mark.serial
@pytest.mark.long
@pytest.mark.parametrize( "dt,v,xc,yc", [(1,-5,0,0), (0.1,5,1,0), (0.1, 2, 2, 1)])
def test_poloidalAdvectionImplicit(dt,v,xc,yc):

    npts = [64,64]
    eta_vals = [np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,2*np.pi,npts[0],endpoint=False),
                np.linspace(0,1,4),np.linspace(0,1,4)]

    N = int(1/dt)

    f_vals = np.ndarray([npts[1],npts[0]])
    final_f_vals = np.ndarray([npts[1],npts[0]])

    deg = 3
    omega = 1

    domain    = [ [1,14.5], [0,2*np.pi] ]
    periodic  = [ False, True ]
    nkts      = [n+1+deg*(int(p)-1)            for (n,p)      in zip( npts, periodic )]
    breaks    = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots     = [spl.make_knots( b,deg,p )     for b,p        in zip(breaks,periodic)]
    bsplines  = [spl.BSplines( k,deg,p )       for k,p        in zip(knots,periodic)]
    eta_grids = [bspl.greville                 for bspl       in bsplines]

    eta_vals[0]=eta_grids[0]
    eta_vals[1]=eta_grids[1]

    constants = Constants()

    polAdv = PoloidalAdvection(eta_vals, bsplines[::-1], constants, True, False, 1e-10)

    phi = spl.Spline2D(bsplines[1],bsplines[0])
    phiVals = np.empty([npts[1],npts[0]])
    phiVals[:] = Phi(eta_vals[0],np.atleast_2d(eta_vals[1]).T, omega, xc, yc)
    interp = spl.SplineInterpolator2D(bsplines[1],bsplines[0])

    interp.compute_interpolant(phiVals,phi)

    f_vals[:,:] = initConds(eta_vals[0],np.atleast_2d(eta_vals[1]).T)

    for n in range(N):
        polAdv.step(f_vals[:,:],dt,phi,v)

    x0 = polAdv._points[1] * np.cos(polAdv._shapedQ)
    y0 = polAdv._points[1] * np.sin(polAdv._shapedQ)

    x = xc + (x0 - xc) * np.cos(omega * -dt * N) - (y0 - yc) * np.sin(omega * -dt * N)
    y = yc + (x0 - xc) * np.sin(omega * -dt * N) + (y0 - yc) * np.cos(omega * -dt * N)

    finalPts = ( np.ndarray([npts[1],npts[0]]), np.ndarray([npts[1],npts[0]]))
    finalPts[0][:] = np.mod(np.arctan2(y, x), 2 * np.pi)
    finalPts[1][:] = np.sqrt(x * x + y * y)
    final_f_vals[:,:] = initConds(finalPts[1],finalPts[0])

    l2=np.sqrt(trapz(trapz((f_vals-final_f_vals)**2,eta_grids[1],axis=0)*eta_grids[0],eta_grids[0]))
    assert(l2<0.2)

@pytest.mark.serial
def test_fluxSurfaceAdvection_gridIntegration():
    npts = [10,20,10,10]
    grid,constants,t = setupCylindricalGrid(npts   = npts,
                                layout = 'flux_surface')

    dt=0.1

    fluxAdv = FluxSurfaceAdvection(grid.eta_grid, grid.get2DSpline(),
                                    grid.getLayout('flux_surface'),dt,constants)

    for i,r in grid.getCoords(0):
        for j,v in grid.getCoords(1):
            fluxAdv.step(grid.get2DSlice([i,j]),j)

@pytest.mark.serial
def test_vParallelAdvection_gridIntegration():
    npts = [4,4,4,100]
    grid,constants,t = setupCylindricalGrid(npts   = npts,
                                layout = 'v_parallel')

    dt=0.1
    c=0

    old_f=grid._f.copy()

    vParAdv = VParallelAdvection(grid.eta_grid, grid.get1DSpline(),constants)

    for i,r in grid.getCoords(0):
        for j,z in grid.getCoords(1):
            for k,q in grid.getCoords(2):
                vParAdv.step(grid.get1DSlice([i,j,k]),dt,c,r)

    assert(np.allclose(old_f,grid._f))

@pytest.mark.serial
def test_poloidalAdvection_gridIntegration():
    npts = [10,20,10,10]
    grid,constants,t = setupCylindricalGrid(npts   = npts,
                                layout = 'poloidal')

    basis = grid.get2DSpline()

    polAdv = PoloidalAdvection(grid.eta_grid, basis,constants)

    phi = spl.Spline2D(basis[0],basis[1])
    phiVals = np.full((npts[1],npts[0]),2)
    interp = spl.SplineInterpolator2D(basis[0],basis[1])

    interp.compute_interpolant(phiVals,phi)

    dt=0.1

    for i,z in grid.getCoords(0):
        for j,v in grid.getCoords(1):
            polAdv.step(grid.get2DSlice([i,j]),dt,phi,v)

"""
# Tests are too slow
@pytest.mark.parallel
def test_equilibrium():
    comm = MPI.COMM_WORLD

    npts = [20,20,10,8]
    grid,constants,t = setupCylindricalGrid(npts   = npts,
                                layout = 'flux_surface',
                                eps    = 0,
                                comm   = comm)

    startVals = grid._f.copy()

    N=10

    dt=0.1
    halfStep = dt*0.5

    fluxAdv = FluxSurfaceAdvection(grid.eta_grid, grid.get2DSpline(),grid.getLayout('flux_surface'),halfStep)
    vParAdv = VParallelAdvection(grid.eta_grid, grid.getSpline(3))
    polAdv = PoloidalAdvection(grid.eta_grid, grid.getSpline(slice(1,None,-1)))

    phi = spl.Spline2D(grid.getSpline(1),grid.getSpline(0))
    phiVals = np.empty([npts[1],npts[0]])
    phiVals[:]=3*grid.eta_grid[0]**2
    interp = spl.SplineInterpolator2D(grid.getSpline(1),grid.getSpline(0))

    interp.compute_interpolant(phiVals,phi)

    for n in range(N):
        for i,r in grid.getCoords(0):
            for j,v in grid.getCoords(1):
                fluxAdv.step(grid.get2DSlice([i,j]),j)

        grid.setLayout('v_parallel')

        for i,r in grid.getCoords(0):
            for j,z in grid.getCoords(1):
                for k,q in grid.getCoords(2):
                    vParAdv.step(grid.get1DSlice([i,j,k]),halfStep,0,r)

        grid.setLayout('poloidal')

        for i,v in grid.getCoords(0):
            for j,z in grid.getCoords(1):
                polAdv.step(grid.get2DSlice([i,j]),dt,phi,v)

        grid.setLayout('v_parallel')

        for i,r in grid.getCoords(0):
            for j,z in grid.getCoords(1):
                for k,q in grid.getCoords(2):
                    vParAdv.step(grid.get1DSlice([i,j,k]),halfStep,0,r)

        grid.setLayout('flux_surface')

        for i,r in grid.getCoords(0):
            for j,v in grid.getCoords(1):
                fluxAdv.step(grid.get2DSlice([i,j]),j)

    print(np.max(startVals-grid._f))
    assert(np.max(startVals-grid._f)<1e-8)

@pytest.mark.parallel
def test_perturbedEquilibrium():
    comm = MPI.COMM_WORLD

    npts = [20,20,10,8]
    grid,constants,t = setupCylindricalGrid(npts   = npts,
                                layout = 'flux_surface',
                                comm   = comm)

    startVals = grid._f.copy()

    N=10

    dt=0.1
    halfStep = dt*0.5

    fluxAdv = FluxSurfaceAdvection(grid.eta_grid, grid.get2DSpline(),grid.getLayout('flux_surface'),halfStep)
    vParAdv = VParallelAdvection(grid.eta_grid, grid.getSpline(3))
    polAdv = PoloidalAdvection(grid.eta_grid, grid.getSpline(slice(1,None,-1)))

    phi = spl.Spline2D(grid.getSpline(1),grid.getSpline(0))
    phiVals = np.empty([npts[1],npts[0]])
    phiVals[:]=3*grid.eta_grid[0]**2
    interp = spl.SplineInterpolator2D(grid.getSpline(1),grid.getSpline(0))

    interp.compute_interpolant(phiVals,phi)

    for n in range(N):
        for i,r in grid.getCoords(0):
            for j,v in grid.getCoords(1):
                fluxAdv.step(grid.get2DSlice([i,j]),j)

        grid.setLayout('v_parallel')

        for i,r in grid.getCoords(0):
            for j,z in grid.getCoords(1):
                for k,q in grid.getCoords(2):
                    vParAdv.step(grid.get1DSlice([i,j,k]),halfStep,0,r)

        grid.setLayout('poloidal')

        for i,v in grid.getCoords(0):
            for j,z in grid.getCoords(1):
                polAdv.step(grid.get2DSlice([i,j]),dt,phi,v)

        grid.setLayout('v_parallel')

        for i,r in grid.getCoords(0):
            for j,z in grid.getCoords(1):
                for k,q in grid.getCoords(2):
                    vParAdv.step(grid.get1DSlice([i,j,k]),halfStep,0,r)

        grid.setLayout('flux_surface')

        for i,r in grid.getCoords(0):
            for j,v in grid.getCoords(1):
                fluxAdv.step(grid.get2DSlice([i,j]),j)

    print(np.max(startVals-grid._f))
    assert(np.max(startVals-grid._f)>1e-8)
"""

@pytest.mark.serial
def test_vParGradAligned():
    comm = MPI.COMM_WORLD

    npts = [20,80,20,8]
    grid,constants,t = setupCylindricalGrid(npts   = npts,
                                layout  = 'flux_surface',
                                eps     = 0,
                                iotaVal = 0.8,
                                n       = -11,
                                comm    = comm)

    N=10

    pG = ParallelGradient(grid.getSpline(1),grid.eta_grid,grid.getLayout(grid.currentLayout),constants)

    phiVals = np.empty([npts[2],npts[1]])

    m, n = (5,-4)
    theta = grid.eta_grid[1]
    phi = grid.eta_grid[2]*2*np.pi/constants.zMax
    phiVals[:,:] = np.sin( m*theta[None,:] + n*phi[:,None] )

    der = np.empty([npts[2],npts[1]])
    pG.parallel_gradient(phiVals,0,der)
    assert(np.isfinite(der).all())
    assert((np.abs(der)<1e-7).all())

@pytest.mark.serial
def test_vParGrad():
    comm = MPI.COMM_WORLD

    npts = [20,20,10,8]
    grid,constants,t = setupCylindricalGrid(npts   = npts,
                                layout = 'flux_surface',
                                eps    = 0,
                                comm   = comm)

    N=10

    pG = ParallelGradient(grid.getSpline(1),grid.eta_grid,grid.getLayout(grid.currentLayout),constants)

    phiVals = np.empty([npts[2],npts[1]])
    phiVals[:]=3

    der = np.empty([npts[2],npts[1]])
    pG.parallel_gradient(phiVals,0,der)
    assert(np.isfinite(der).all())
    assert((np.abs(der)<1e-12).all())

def pg_Phi(theta,z):
    #return np.cos(z*np.pi*0.1) + np.sin(theta)
    return np.sin(z*np.pi*0.1)**2 + np.cos(theta)**2

def pg_dPhi(r,theta,z,btheta,bz):
    #return -np.sin(z*np.pi*0.1)*np.pi*0.1*bz + np.cos(theta)*btheta
    return 2*np.sin(z*np.pi*0.1)*np.cos(z*np.pi*0.1)*np.pi*0.1*bz - 2*np.cos(theta)*np.sin(theta)*btheta/r

@pytest.mark.serial
@pytest.mark.long
@pytest.mark.parametrize( "phiOrder,zOrder", [(3,3),(3,4),(3,5),(4,4)] )
def test_Phi_deriv_dz(phiOrder,zOrder):
    nconvpts = 2
    npts = [1,64,64]

    l2=np.empty(nconvpts)
    linf=np.empty(nconvpts)

    constants = Constants()
    constants.iotaVal=0.8
    constants.n=-11

    for i in range(nconvpts):
        breaks_theta = np.linspace(0,2*np.pi,npts[1]+1)
        spline_theta = spl.BSplines(spl.make_knots(breaks_theta,phiOrder,True),phiOrder,True)
        breaks_z = np.linspace(0,20,npts[2]+1)
        spline_z = spl.BSplines(spl.make_knots(breaks_z,3,True),3,True)

        eta_grid = [np.array([1]), spline_theta.greville, spline_z.greville]

        dz = eta_grid[2][2]-eta_grid[2][1]
        dtheta = constants.iota()*dz/constants.R0

        bz = 1 / np.sqrt(1+( constants.iota(1)/constants.R0)**2)
        btheta =  constants.iota(1)/constants.R0 / np.sqrt(1+( constants.iota(1)/constants.R0)**2)
        # ~ bz = dz/np.sqrt(dz**2+dtheta**2)
        # ~ btheta = dtheta/np.sqrt(dz**2+dtheta**2)

        phiVals = np.empty([npts[2],npts[1]])
        phiVals[:] = pg_Phi(eta_grid[1][None,:],eta_grid[2][:,None])

        layout = Layout('par_grad',[1],[0,2,1],eta_grid,[0])

        pGrad = ParallelGradient(spline_theta,eta_grid,layout,constants,zOrder)

        approxGrad = np.empty([npts[2],npts[1]])
        pGrad.parallel_gradient(phiVals,0,approxGrad)
        exactGrad = pg_dPhi(1,eta_grid[1][None,:],eta_grid[2][:,None],btheta,bz)

        err = approxGrad-exactGrad

        l2[i]=np.sqrt(np.trapz(np.trapz(err**2,dx=dz),dx=dtheta))
        linf[i]=np.linalg.norm(err.flatten(),np.inf)

        npts[1]*=2
        npts[2]*=2

    print("linf:",linf)
    print("l2:",l2)

    linfOrder = np.log2(linf[0]/linf[1])
    print("linfordre:",np.log2(linf[:-1]/linf[1:]))
    print("l2ordre:",np.log2(l2[:-1]/l2[1:]))

    assert(abs(linfOrder-zOrder)<0.1)

