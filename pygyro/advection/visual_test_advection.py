import pytest
import numpy                as np
from matplotlib             import rc        as pltFont
import matplotlib.pyplot    as plt
import matplotlib.colors    as colors
from math                 import pi

from ..                                     import splines as spl
from ..initialisation.setups                import setupCylindricalGrid
from ..initialisation.constants             import get_constants
from ..initialisation.mod_initialiser_funcs import fEq
from ..model.layout                         import Layout
from .advection                             import FluxSurfaceAdvection, PoloidalAdvection, VParallelAdvection, ParallelGradient

@pytest.mark.serial
def test_fluxSurfaceAdvection():
    npts = [30,20]
    eta_vals = [np.linspace(0,1,4),np.linspace(0,2*pi,npts[0],endpoint=False),
                np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,1,4)]

    N = 100

    dt=0.1

    c=2

    f_vals = np.ndarray([N+1,npts[0],npts[1]])

    domain    = [ [0,2*pi], [0,20] ]
    nkts      = [n+1                           for n          in npts ]
    breaks    = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots     = [spl.make_knots( b,3,True )    for b          in breaks]
    bsplines  = [spl.BSplines( k,3,True )      for k          in knots]
    eta_grids = [bspl.greville                 for bspl       in bsplines]

    eta_vals[1]=eta_grids[0]
    eta_vals[2]=eta_grids[1]
    eta_vals[3][0]=c

    layout = Layout('flux',[1],[0,3,1,2],eta_vals,[0])

    constants = get_constants('testSetups/iota0.json')

    fluxAdv = FluxSurfaceAdvection(eta_vals, bsplines, layout, dt, constants)

    f_vals[0,:,:]=np.sin(eta_vals[2]*pi/10)

    for n in range(N):
        f_vals[n+1,:,:]=f_vals[n,:,:]
        fluxAdv.step(f_vals[n+1,:,:],0)

    x,y = np.meshgrid(eta_vals[2], eta_vals[1])

    f_min = np.min(f_vals)
    f_max = np.max(f_vals)

    plt.ion()

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.25, 0.7, 0.7],)
    colorbarax2 = fig.add_axes([0.85, 0.1, 0.03, 0.8],)

    line1 = ax.pcolormesh(x,y,f_vals[0,:,:],vmin=f_min,vmax=f_max)
    fig.canvas.draw()
    fig.canvas.flush_events()

    fig.colorbar(line1, cax = colorbarax2)

    for n in range(1,N):
        del line1
        line1 = ax.pcolormesh(x,y,f_vals[n,:,:],vmin=f_min,vmax=f_max)
        fig.canvas.draw()
        fig.canvas.flush_events()

    print(np.max(f_vals[N,:,:]-f_vals[0,:,:]))

@pytest.mark.serial
def test_poloidalAdvection_invariantPhi():
    npts = [30,20]
    eta_vals = [np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,2*pi,npts[0],endpoint=False),
                np.linspace(0,1,4),np.linspace(0,1,4)]

    N = 200
    dt=0.1

    v=0

    f_vals = np.ndarray([N+1,npts[1],npts[0]])

    deg = 3

    domain    = [ [0.1,14.5], [0,2*pi] ]
    periodic  = [ False, True ]
    nkts      = [n+1+deg*(int(p)-1)            for (n,p)      in zip( npts, periodic )]
    breaks    = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots     = [spl.make_knots( b,deg,p )     for b,p        in zip(breaks,periodic)]
    bsplines  = [spl.BSplines( k,deg,p )       for k,p        in zip(knots,periodic)]
    eta_grids = [bspl.greville                 for bspl       in bsplines]

    eta_vals[0]=eta_grids[0]
    eta_vals[1]=eta_grids[1]

    constants = get_constants('testSetups/iota0.json')

    polAdv = PoloidalAdvection(eta_vals, bsplines[::-1],constants)

    phi = spl.Spline2D(bsplines[1],bsplines[0])
    phiVals = np.empty([npts[1],npts[0]])
    phiVals[:]=3*eta_vals[0]**2 * (1+ 1e-1 * np.cos(np.atleast_2d(eta_vals[1]).T*2))
    #phiVals[:]=10*eta_vals[0]
    interp = spl.SplineInterpolator2D(bsplines[1],bsplines[0])

    interp.compute_interpolant(phiVals,phi)

    #~ f_vals[0,:,:] = np.exp(-np.atleast_2d((eta_vals[1]-pi)**2).T - (eta_vals[0]-7)**2)/4 \
                        #~ + fEq(0.1,v,constants.CN0,constants.kN0,
                                            #~ constants.deltaRN0,constants.rp,
                                            #~ constants.CTi,constants.kTi,
                                            #~ constants.deltaRTi)
    f_vals[0,:,:] = phiVals + fEq(0.1,v,constants.CN0,constants.kN0,
                                            constants.deltaRN0,constants.rp,
                                            constants.CTi,constants.kTi,
                                            constants.deltaRTi)

    for n in range(N):
        f_vals[n+1,:,:]=f_vals[n,:,:]
        polAdv.step(f_vals[n+1,:,:],dt,phi,v)

    f_min = np.min(f_vals)
    f_max = np.max(f_vals)

    plt.ion()

    fig = plt.figure()
    ax = plt.subplot(111, projection='polar')
    #ax = fig.add_axes([0.1, 0.25, 0.7, 0.7],)
    colorbarax2 = fig.add_axes([0.85, 0.1, 0.03, 0.8],)

    plotParams = {'vmin':f_min,'vmax':f_max, 'cmap':"jet"}

    line1 = ax.contourf(eta_vals[1],eta_vals[0],f_vals[0,:,:].T,20,**plotParams)
    fig.canvas.draw()
    fig.canvas.flush_events()

    fig.colorbar(line1, cax = colorbarax2)

    for n in range(1,N):
        for coll in line1.collections:
            coll.remove()
        del line1
        line1 = ax.contourf(eta_vals[1],eta_vals[0],f_vals[n,:,:].T,20,**plotParams)
        fig.canvas.draw()
        fig.canvas.flush_events()

@pytest.mark.serial
def test_poloidalAdvection_vortex():
    npts = [30,20]
    eta_vals = [np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,2*pi,npts[0],endpoint=False),
                np.linspace(0,1,4),np.linspace(0,1,4)]

    N = 200
    dt=0.1

    v=0

    f_vals = np.ndarray([N+1,npts[1],npts[0]])

    deg = 3

    domain    = [ [0.1,14.5], [0,2*pi] ]
    periodic  = [ False, True ]
    nkts      = [n+1+deg*(int(p)-1)            for (n,p)      in zip( npts, periodic )]
    breaks    = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots     = [spl.make_knots( b,deg,p )     for b,p        in zip(breaks,periodic)]
    bsplines  = [spl.BSplines( k,deg,p )       for k,p        in zip(knots,periodic)]
    eta_grids = [bspl.greville                 for bspl       in bsplines]

    eta_vals[0]=eta_grids[0]
    eta_vals[1]=eta_grids[1]

    constants = get_constants('testSetups/iota0.json')

    polAdv = PoloidalAdvection(eta_vals, bsplines[::-1],constants)

    phi = spl.Spline2D(bsplines[1],bsplines[0])
    phiVals = np.empty([npts[1],npts[0]])
    phiVals[:]=10*eta_vals[0]
    interp = spl.SplineInterpolator2D(bsplines[1],bsplines[0])

    interp.compute_interpolant(phiVals,phi)

    f_vals[0,:,:] = np.exp(-np.atleast_2d((eta_vals[1]-pi)**2).T - (eta_vals[0]-7)**2)/4 \
                        + fEq(0.1,v,constants.CN0,constants.kN0,
                                            constants.deltaRN0,constants.rp,
                                            constants.CTi,constants.kTi,
                                            constants.deltaRTi)

    for n in range(N):
        f_vals[n+1,:,:]=f_vals[n,:,:]
        polAdv.step(f_vals[n+1,:,:],dt,phi,v)

    f_min = np.min(f_vals)
    f_max = np.max(f_vals)

    plt.ion()

    fig = plt.figure()
    ax = plt.subplot(111, projection='polar')
    #ax = fig.add_axes([0.1, 0.25, 0.7, 0.7],)
    colorbarax2 = fig.add_axes([0.85, 0.1, 0.03, 0.8],)

    plotParams = {'vmin':f_min,'vmax':f_max, 'cmap':"jet"}

    line1 = ax.contourf(eta_vals[1],eta_vals[0],f_vals[0,:,:].T,20,**plotParams)
    fig.canvas.draw()
    fig.canvas.flush_events()

    fig.colorbar(line1, cax = colorbarax2)

    for n in range(1,N):
        for coll in line1.collections:
            coll.remove()
        del line1
        line1 = ax.contourf(eta_vals[1],eta_vals[0],f_vals[n,:,:].T,20,**plotParams)
        fig.canvas.draw()
        fig.canvas.flush_events()

@pytest.mark.serial
def test_poloidalAdvection_constantAdv():
    npts = [30,20]
    eta_vals = [np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,2*pi,npts[0],endpoint=False),
                np.linspace(0,1,4),np.linspace(0,1,4)]

    N = 200
    dt=0.1

    v=0

    f_vals = np.ndarray([N+1,npts[1],npts[0]])

    deg = 3

    domain    = [ [0.1,14.5], [0,2*pi] ]
    periodic  = [ False, True ]
    nkts      = [n+1+deg*(int(p)-1)            for (n,p)      in zip( npts, periodic )]
    breaks    = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots     = [spl.make_knots( b,deg,p )     for b,p        in zip(breaks,periodic)]
    bsplines  = [spl.BSplines( k,deg,p )       for k,p        in zip(knots,periodic)]
    eta_grids = [bspl.greville                 for bspl       in bsplines]

    eta_vals[0]=eta_grids[0]
    eta_vals[1]=eta_grids[1]

    constants = get_constants('testSetups/iota0.json')

    polAdv = PoloidalAdvection(eta_vals, bsplines[::-1],constants)

    phi = spl.Spline2D(bsplines[1],bsplines[0])
    phiVals = np.empty([npts[1],npts[0]])
    phiVals[:]=3*eta_vals[0]**2
    interp = spl.SplineInterpolator2D(bsplines[1],bsplines[0])

    interp.compute_interpolant(phiVals,phi)

    f_vals[0,:,:] = np.exp(-np.atleast_2d((eta_vals[1]-pi)**2).T - (eta_vals[0]-7)**2)/4 \
                        + fEq(0.1,v,constants.CN0,constants.kN0,
                                            constants.deltaRN0,constants.rp,
                                            constants.CTi,constants.kTi,
                                            constants.deltaRTi)

    for n in range(N):
        f_vals[n+1,:,:]=f_vals[n,:,:]
        polAdv.step(f_vals[n+1,:,:],dt,phi,v)

    f_min = np.min(f_vals)
    f_max = np.max(f_vals)

    plt.ion()

    fig = plt.figure()
    ax = plt.subplot(111, projection='polar')
    #ax = fig.add_axes([0.1, 0.25, 0.7, 0.7],)
    colorbarax2 = fig.add_axes([0.85, 0.1, 0.03, 0.8],)

    plotParams = {'vmin':f_min,'vmax':f_max, 'cmap':"jet"}

    line1 = ax.contourf(eta_vals[1],eta_vals[0],f_vals[0,:,:].T,20,**plotParams)
    fig.canvas.draw()
    fig.canvas.flush_events()

    fig.colorbar(line1, cax = colorbarax2)

    for n in range(1,N):
        for coll in line1.collections:
            coll.remove()
        del line1
        line1 = ax.contourf(eta_vals[1],eta_vals[0],f_vals[n,:,:].T,20,**plotParams)
        fig.canvas.draw()
        fig.canvas.flush_events()

@pytest.mark.serial
def test_vParallelAdvection():
    npts = [4,4,4,100]
    grid,constants = setupCylindricalGrid(constantFile='testSetups/iota0.json',
                                npts   = npts,
                                layout = 'v_parallel')

    N = 100

    dt=0.1

    c = 1.0

    f_vals = np.ndarray([npts[3],N])

    vParAdv = VParallelAdvection(grid.eta_grid, grid.get1DSpline(),constants,'periodic')

    for n in range(N):
        for i,r in grid.getCoords(0):
            vParAdv.step(grid.get1DSlice([i,0,0]),dt,c,r)
        f_vals[:,n]=grid.get1DSlice([0,0,0])

    plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(grid.eta_grid[3], f_vals[:,0]) # Returns a tuple of line objects, thus the comma

    for n in range(1,N):
        line1.set_ydata(f_vals[:,n])
        fig.canvas.draw()
        fig.canvas.flush_events()

def positional_Phi(r,theta,a,b,c,d):
    return - a * (r-b)**2 + c*np.sin(d*theta)

"""
def initConditions(r,theta):
    a=6
    factor = pi/a/2
    x=r*np.cos(theta)
    y=r*np.sin(theta)
    R1=np.sqrt((x+7)**2+8*y**2)
    R2=np.sqrt(4*(x+7)**2+0.5*y**2)
    result=0.0
    if (R1<=a):
        result+=0.5*np.cos(R1*factor)**4
    if (R2<=a):
        result+=0.5*np.cos(R2*factor)**4
    return result
"""

def initConditions(r,theta):
    a=4
    factor = pi/a/2
    r=np.sqrt((r-7)**2+2*(theta-pi)**2)

    if (r<=a):
        return np.cos(r*factor)**4
    else:
        return 0.0

initConds = np.vectorize(initConditions, otypes=[np.float])

@pytest.mark.serial
def test_poloidalAdvection():
    #~ npts = [128,128]
    npts = [16,16]

    print(npts)
    eta_vals = [np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,2*pi,npts[0],endpoint=False),
                np.linspace(0,1,4),np.linspace(0,1,4)]

    N = 100
    dt=0.01

    v=0

    f_vals = np.ndarray([N+1,npts[1]+1,npts[0]])
    #~ f_vals = np.ndarray([N+1,npts[1],npts[0]])

    deg = 3

    domain    = [ [1,13], [0,2*pi] ]
    periodic  = [ False, True ]
    nkts      = [n+1+deg*(int(p)-1)            for (n,p)      in zip( npts, periodic )]
    breaks    = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots     = [spl.make_knots( b,deg,p )     for b,p        in zip(breaks,periodic)]
    bsplines  = [spl.BSplines( k,deg,p )       for k,p        in zip(knots,periodic)]
    eta_grids = [bspl.greville                 for bspl       in bsplines]

    eta_vals[0]=eta_grids[0]
    eta_vals[1]=eta_grids[1]

    constants = get_constants('testSetups/iota0.json')

    polAdv = PoloidalAdvection(eta_vals, bsplines[::-1],constants,True)

    phi = spl.Spline2D(bsplines[1],bsplines[0])
    phiVals = np.empty([npts[1],npts[0]])
    a=5
    b=0
    c=40
    d=1
    phiVals[:] = positional_Phi(eta_vals[0],np.atleast_2d(eta_vals[1]).T,a,b,c,d)
    interp = spl.SplineInterpolator2D(bsplines[1],bsplines[0])

    interp.compute_interpolant(phiVals,phi)

    f_vals[0,:-1,:] = initConds(eta_vals[0],np.atleast_2d(eta_vals[1]).T)
    f_vals[0,-1,:] = f_vals[0,0,:]
    #f_vals[0,:,:] = initConds(eta_vals[0],np.atleast_2d(eta_vals[1]).T)

    endPts = ( np.ndarray([npts[1],npts[0]]), np.ndarray([npts[1],npts[0]]))
    endPts[0][:] = polAdv._shapedQ   +     2*a*dt/constants.B0
    endPts[1][:] = np.sqrt(polAdv._points[1]**2-c*np.sin(d*polAdv._shapedQ)/a/constants.B0 \
                    + c*np.sin(d*endPts[0])/a/constants.B0)

    for n in range(N):
        f_vals[n+1,:-1,:]=f_vals[n,:-1,:]
        #f_vals[:,:,n+1]=f_vals[:,:,n]
        polAdv.exact_step(f_vals[n+1,:-1,:],endPts,v)
        polAdv.step(f_vals[n+1,:-1,:],dt,phi,v)
        #polAdv.step(f_vals[:,:,n+1],dt,phi,v)
        #polAdv.exact_step(f_vals[:,:,n+1],endPts,v)

    f_vals[:,-1,:]=f_vals[:,0,:]
    f_min = np.min(f_vals)
    f_max = np.max(f_vals)

    print(f_min,f_max)

    theta=np.append(eta_vals[1],eta_vals[1][0])
    #theta=eta_vals[1]

    plt.ion()

    font = {'size'   : 16}

    pltFont('font', **font)

    fig = plt.figure()
    ax = plt.subplot(111, projection='polar')
    ax.set_rlim(0,13)
    colorbarax2 = fig.add_axes([0.85, 0.1, 0.03, 0.8],)

    norm = colors.BoundaryNorm(boundaries=np.linspace(-1,1,41), ncolors=256,clip=True)
    plotParams = {'vmin':-1,'vmax':1, 'norm':norm, 'cmap':"jet"}

    line1 = ax.contourf(theta,eta_vals[0],f_vals[0,:,:].T,20,**plotParams)
    fig.canvas.draw()
    fig.canvas.flush_events()

    fig.colorbar(line1, cax = colorbarax2)

    for n in range(1,N+1):
        for coll in line1.collections:
            coll.remove()
        del line1
        line1 = ax.contourf(theta,eta_vals[0],f_vals[n,:,:].T,20,**plotParams)
        print(f_vals[n,:,:])
        fig.canvas.draw()
        fig.canvas.flush_events()

def initConditionsFlux(theta,z):
    a=4
    factor = pi/a/2
    r=np.sqrt((z-10)**2+2*(theta-4)**2)
    if (r<=4):
        return np.cos(r*factor)**6
    else:
        return 0.0

initCondsF = np.vectorize(initConditionsFlux, otypes=[np.float])

def iota0(r = 6.0):
    return np.full_like(r,0.0,dtype=float)

def iota8(r = 6.0):
    return np.full_like(r,0.8,dtype=float)

@pytest.mark.serial
def test_fluxAdvection_dz():
    dt=0.1
    npts = [64,64]

    CFL = dt*(npts[0]+npts[1])

    N = 100

    v=0

    eta_vals = [np.linspace(0,1,4),np.linspace(0,2*pi,npts[0],endpoint=False),
            np.linspace(0,20,npts[1],endpoint=False),np.linspace(0,1,4)]

    c=2

    f_vals = np.ndarray([N+1,npts[0],npts[1]])

    domain    = [ [0,2*pi], [0,20] ]
    nkts      = [n+1                           for n          in npts ]
    breaks    = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots     = [spl.make_knots( b,3,True )    for b          in breaks]
    bsplines  = [spl.BSplines( k,3,True )      for k          in knots]
    eta_grids = [bspl.greville                 for bspl       in bsplines]

    eta_vals[1]=eta_grids[0]
    eta_vals[2]=eta_grids[1]
    eta_vals[3][0]=c

    constants = get_constants('testSetups/iota0.json')

    layout = Layout('flux',[1],[0,3,1,2],eta_vals,[0])
    fluxAdv = FluxSurfaceAdvection(eta_vals, bsplines, layout, dt, constants)

    dz = eta_vals[2][1]-eta_vals[2][0]
    dtheta = iota0()*dz/constants.R0

    f_vals[0,:,:] = initCondsF(np.atleast_2d(eta_vals[1]).T,eta_vals[2])

    for n in range(1,N+1):
        f_vals[n,:,:]=f_vals[n-1,:,:]
        fluxAdv.step(f_vals[n,:,:],0)

    x,y = np.meshgrid(eta_vals[2],eta_vals[1])

    f_min = np.min(f_vals)
    f_max = np.max(f_vals)

    plt.ion()

    fig = plt.figure()
    #~ ax = plt.subplot(111, projection='polar')
    ax = fig.add_axes([0.1, 0.25, 0.7, 0.7],)
    colorbarax2 = fig.add_axes([0.85, 0.1, 0.03, 0.8],)

    #~ line1 = ax.pcolormesh(y,x,f_vals[0,:,:],vmin=f_min,vmax=f_max)
    line1 = ax.pcolormesh(x,y,f_vals[0,:,:],vmin=f_min,vmax=f_max)
    ax.set_title('End of Calculation')
    fig.canvas.draw()
    fig.canvas.flush_events()

    fig.colorbar(line1, cax = colorbarax2)

    #~ plt.show()

    for n in range(1,N+1):
        del line1
        line1 = ax.pcolormesh(x,y,f_vals[n,:,:],vmin=f_min,vmax=f_max)
        #~ line1 = ax.pcolormesh(y,x,f_vals[n,:,:],vmin=f_min,vmax=f_max)
        fig.canvas.draw()
        fig.canvas.flush_events()

def test_flux_aligned():
    dt=0.1
    npts = [64,64]

    constants = get_constants('testSetups/iota8.json')

    CFL = dt*(npts[0]+npts[1])

    N = 100

    eta_vals = [np.linspace(0,1,4),np.linspace(0,2*pi,npts[0],endpoint=False),
            np.linspace(0,2*pi*constants.R0,npts[1],endpoint=False),np.linspace(0,1,4)]

    c=2

    f_vals = np.ndarray([N+1,npts[0],npts[1]])

    domain    = [ [0,2*pi], [0,2*pi*constants.R0] ]
    nkts      = [n+1                           for n          in npts ]
    breaks    = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
    knots     = [spl.make_knots( b,3,True )    for b          in breaks]
    bsplines  = [spl.BSplines( k,3,True )      for k          in knots]
    eta_grids = [bspl.greville                 for bspl       in bsplines]

    eta_vals[1]=eta_grids[0]
    eta_vals[2]=eta_grids[1]
    eta_vals[3][0]=c

    layout = Layout('flux',[1],[0,3,1,2],eta_vals,[0])
    fluxAdv = FluxSurfaceAdvection(eta_vals, bsplines, layout, dt, constants)

    m, n = (5, 4)
    theta = eta_grids[0]
    phi = eta_grids[1]*2*pi/domain[1][1]
    f_vals[0,:,:] = 0.5 +  0.5 * np.sin( m*theta[:,None] - n*phi[None,:] )

    #~ m, n = (4, 5)
    # ~ m, n = (0, 5)
    #~ theta = eta_grids[0]
    #~ phi = eta_grids[1]*2*pi/domain[1][1]
    #~ f_vals[:,:,0] = 0.5+ 0.5*np.sin( m*theta[:,None] + n*phi[None,:] )

    for n in range(1,N+1):
        f_vals[n,:,:]=f_vals[n-1,:,:]
        fluxAdv.step(f_vals[n,:,:],0)

    f_min = np.min(f_vals)
    f_max = np.max(f_vals)
    e_min = np.min(f_vals-f_vals[0,None,:,:])
    e_max = np.max(f_vals-f_vals[0,None,:,:])

    print(f_min,f_max)

    #~ f_min = 0
    #~ f_max = 1

    plt.ion()

    fig = plt.figure()
    #~ ax = plt.subplot(111, projection='polar')
    ax = fig.add_axes([0.1, 0.25, 0.7, 0.7],)
    colorbarax1 = fig.add_axes([0.85, 0.1, 0.03, 0.8],)

    fig2 = plt.figure()
    #~ ax = plt.subplot(111, projection='polar')
    ax2 = fig2.add_axes([0.1, 0.25, 0.7, 0.7],)
    colorbarax2 = fig2.add_axes([0.85, 0.1, 0.03, 0.8],)

    #~ line1 = ax.pcolormesh(phi,theta,f_vals[:,:,0],vmin=f_min,vmax=f_max)
    #~ line1 = ax.pcolormesh(theta,phi,f_vals[:,:,0].T,vmin=f_min,vmax=f_max)
    line1 = ax.pcolormesh(phi,theta,f_vals[0,:,:]-f_vals[0,:,:],vmin=e_min,vmax=e_max)
    line2 = ax2.pcolormesh(phi,theta,f_vals[0,:,:],vmin=f_min,vmax=f_max)
    #~ line1 = ax.pcolormesh(phi,theta,f_vals[:,:,0],vmin=f_min,vmax=f_max)
    ax.set_title('Error')
    ax2.set_title('Values')
    fig.canvas.draw()
    fig.canvas.flush_events()

    fig.colorbar(line1, cax = colorbarax1)
    fig.colorbar(line2, cax = colorbarax2)

    #~ plt.show()

    for n in range(1,N+1):
        del line1
        del line2
        #~ line1 = ax.pcolormesh(phi,theta,f_vals[:,:,n],vmin=f_min,vmax=f_max)
        #~ line1 = ax.pcolormesh(phi,theta,f_vals[:,:,n],vmin=f_min,vmax=f_max)
        #~ line1 = ax.pcolormesh(theta,phi,f_vals[:,:,n].T,vmin=f_min,vmax=f_max)
        line1 = ax.pcolormesh(phi,theta,f_vals[n,:,:]-f_vals[0,:,:],vmin=e_min,vmax=e_max)
        line2 = ax2.pcolormesh(phi,theta,f_vals[n,:,:],vmin=f_min,vmax=f_max)
        fig.canvas.draw()
        fig.canvas.flush_events()
        fig2.canvas.draw()
        fig2.canvas.flush_events()

def Phi(theta,z):
    #return np.cos(z*pi*0.1) + np.sin(theta)
    # ~ return np.sin(z*pi*0.1)**2 + np.cos(theta)**2
    m, n = (5, 4)
    phi = z*2*np.pi/20
    return 0.5+ 0.5*np.sin( m*theta - n*phi )

def dPhi(r,theta,z,btheta,bz):
    #return -np.sin(z*pi*0.1)*pi*0.1*bz + np.cos(theta)*btheta
    # ~ return 2*np.sin(z*pi*0.1)*np.cos(z*pi*0.1)*pi*0.1*bz - 2*np.cos(theta)*np.sin(theta)*btheta/r
    m, n = (5, 4)
    phi = z*2*np.pi/20
    return 0.5*np.cos( m*theta - n*phi )*(m*btheta/r - bz*n*2*np.pi/20)

@pytest.mark.serial
def test_Phi_deriv():
    npts = [1,64,128]

    constants = get_constants('testSetups/iota8.json')

    breaks_theta = np.linspace(0,2*pi,npts[1]+1)
    spline_theta = spl.BSplines(spl.make_knots(breaks_theta,3,True),3,True)
    breaks_z = np.linspace(0,20,npts[2]+1)
    spline_z = spl.BSplines(spl.make_knots(breaks_z,3,True),3,True)

    eta_grid = [np.array([1]), spline_theta.greville, spline_z.greville]

    dz = eta_grid[2][1]-eta_grid[2][0]
    dtheta = iota8()*dz/constants.R0

    r = eta_grid[0]

    bz = 1 / np.sqrt(1+(r * iota8(r)/constants.R0)**2)
    btheta = r * iota8(r)/constants.R0 / np.sqrt(1+(r * iota8(r)/constants.R0)**2)
    #bz = dz/np.sqrt(dz**2+dtheta**2)
    #btheta = dtheta/np.sqrt(dz**2+r*dtheta**2)

    phiVals = np.empty([npts[2],npts[1]])
    phiVals[:] = Phi(eta_grid[1][None,:],eta_grid[2][:,None])

    theLayout = Layout('full',[1,1,1],[0,2,1],eta_grid,[0,0,0])

    pGrad = ParallelGradient(spline_theta,eta_grid,theLayout,constants)

    approxGrad = np.empty([npts[2],npts[1]])
    pGrad.parallel_gradient(phiVals,0,approxGrad)
    exactGrad = dPhi(eta_grid[0][:,None,None],eta_grid[1][None,:],eta_grid[2][:,None],btheta,bz)

    err = np.squeeze(approxGrad-exactGrad)

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.25, 0.7, 0.7],)
    colorbarax1 = fig.add_axes([0.85, 0.1, 0.03, 0.8],)

    fig2 = plt.figure()
    ax2 = fig2.add_axes([0.1, 0.25, 0.7, 0.7],)
    colorbarax2 = fig2.add_axes([0.85, 0.1, 0.03, 0.8],)

    fig3 = plt.figure()
    ax3 = fig3.add_axes([0.1, 0.25, 0.7, 0.7],)
    colorbarax3 = fig3.add_axes([0.85, 0.1, 0.03, 0.8],)

    line1 = ax.pcolormesh(eta_grid[1],eta_grid[2],err)
    line2 = ax2.pcolormesh(eta_grid[1],eta_grid[2],approxGrad)
    line3 = ax3.pcolormesh(eta_grid[1],eta_grid[2],np.squeeze(exactGrad))
    ax.set_title('Error')
    ax2.set_title('Values')
    ax3.set_title('Expected Values')

    fig.colorbar(line1, cax = colorbarax1)
    fig2.colorbar(line2, cax = colorbarax2)
    fig3.colorbar(line3, cax = colorbarax3)

    plt.show()
