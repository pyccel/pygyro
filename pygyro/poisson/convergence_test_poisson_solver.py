from mpi4py                 import MPI
import numpy                as np
import pytest
from math                   import pi
from scipy.integrate        import trapz

from ..model.process_grid       import compute_2d_process_grid
from ..model.layout             import LayoutSwapper, getLayoutHandler
from ..model.grid               import Grid
from ..initialisation.setups    import setupCylindricalGrid
from ..                         import splines as spl
from .poisson_solver            import PoissonSolver, DensityFinder
from ..splines.splines              import BSplines, Spline1D
from ..splines.spline_interpolators import SplineInterpolator1D

@pytest.mark.serial
@pytest.mark.parametrize( "deg", [1,2,3,4,5] )
def test_BasicPoissonEquation_pointConverge(deg):
    npt = 8
    nconvpts = 10
    
    npts = [npt,8,4]
    domain = [[1,15],[0,2*pi],[0,1]]
    degree = [deg,3,3]
    period = [False,True,False]
    comm = MPI.COMM_WORLD
    
    l2 = np.empty(nconvpts)
    lInf = np.empty(nconvpts)
    
    for c in range(nconvpts):
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
        
        err=(phi._f-phi_exact._f)[0,0]
        l2[c] = np.sqrt(trapz(np.real(err*err.conj()),r))
        lInf[c] = np.max(np.abs(phi._f-phi_exact._f))
        
        npts[0]*=2
    
    l2Order = np.log2(l2[:-1]/l2[1:])
    lInfOrder = np.log2(lInf[:-1]/lInf[1:])
    
    print(" ")
    print(deg," & & ",npt,"    & & $",end=' ')
    mag2Order = np.floor(np.log10(l2[0]))
    magInfOrder = np.floor(np.log10(lInf[0]))
    print(str.format('{0:.2f}',l2[0]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
    print("}$ &       & $",str.format('{0:.2f}',lInf[0]*10**-magInfOrder),"\\cdot 10^{", str.format('{0:n}',magInfOrder),end=' ')
    print("}$ &  \\\\")
    print("\\hline")
    for i in range(nconvpts-1):
        n=npt*2**(i+1)
        mag2Order = np.floor(np.log10(l2[i+1]))
        magInfOrder = np.floor(np.log10(lInf[i+1]))
        print(deg," & & ",n,"    & & $",end=' ')
        print(str.format('{0:.2f}',l2[i+1]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
        print("}$ & ",str.format('{0:.2f}',l2Order[i])," & $",end=' ')
        print(str.format('{0:.2f}',lInf[i+1]*10**-magInfOrder),"\\cdot 10^{", str.format('{0:n}',magInfOrder),end=' ')
        print("}$ & ",str.format('{0:.2f}',lInfOrder[i])," \\\\")
        print("\\hline")

@pytest.mark.serial
def test_BasicPoissonEquation_degreeConverge():
    deg = 1
    # splev only works for splines of degree <=5
    nconvpts = 3
    
    npts = [512,8,4]
    domain = [[1,15],[0,2*pi],[0,1]]
    degree = [deg,3,3]
    period = [False,True,False]
    comm = MPI.COMM_WORLD
    
    l2 = np.empty(nconvpts)
    lInf = np.empty(nconvpts)
    
    for c in range(nconvpts):
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
        
        err=(phi._f-phi_exact._f)[0,0]
        l2[c] = np.sqrt(trapz(np.real(err*err.conj()),r))
        lInf[c] = np.max(np.abs(phi._f-phi_exact._f))
        
        degree[0]*=2
    
    print(l2)
    print(lInf)
    
    l2Order = np.log2(l2[:-1]/l2[1:])
    lInfOrder = np.log2(lInf[:-1]/lInf[1:])
    
    print(l2Order)
    print(lInfOrder)
    
    print(" ")
    print(deg," & & ",npts[0],"    & & $",end=' ')
    mag2Order = np.floor(np.log10(l2[0]))
    magInfOrder = np.floor(np.log10(lInf[0]))
    print(str.format('{0:.2f}',l2[0]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
    print("}$ &       & $",str.format('{0:.2f}',lInf[0]*10**-magInfOrder),"\\cdot 10^{", str.format('{0:n}',magInfOrder),end=' ')
    print("}$ &  \\\\")
    print("\\hline")
    for i in range(nconvpts-1):
        n=deg*2**(i+1)
        mag2Order = np.floor(np.log10(l2[i+1]))
        magInfOrder = np.floor(np.log10(lInf[i+1]))
        print(n," & & ",npts[0],"    & & $",end=' ')
        print(str.format('{0:.2f}',l2[i+1]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
        print("}$ & ",str.format('{0:.2f}',l2Order[i])," & $",end=' ')
        print(str.format('{0:.2f}',lInf[i+1]*10**-magInfOrder),"\\cdot 10^{", str.format('{0:n}',magInfOrder),end=' ')
        print("}$ & ",str.format('{0:.2f}',lInfOrder[i])," \\\\")
        print("\\hline")

@pytest.mark.serial
@pytest.mark.parametrize( "deg", [1,2,3,4,5] )
def test_grad_pointConverge(deg):
    npt = 8
    nconvpts = 10
    
    npts = [npt,8,4]
    domain = [[1,9],[0,2*pi],[0,1]]
    degree = [deg,3,3]
    period = [False,True,False]
    comm = MPI.COMM_WORLD
    
    l2 = np.empty(nconvpts)
    lInf = np.empty(nconvpts)
    
    for c in range(nconvpts):
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
        
        err=(phi._f-phi_exact._f)[0,0]
        l2[c] = np.sqrt(trapz(np.real(err*err.conj()),r))
        lInf[c] = np.max(np.abs(phi._f-phi_exact._f))
        
        npts[0]*=2
    
    l2Order = np.log2(l2[:-1]/l2[1:])
    lInfOrder = np.log2(lInf[:-1]/lInf[1:])
    
    print(" ")
    print(deg," & & ",npt,"    & & $",end=' ')
    mag2Order = np.floor(np.log10(l2[0]))
    magInfOrder = np.floor(np.log10(lInf[0]))
    print(str.format('{0:.2f}',l2[0]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
    print("}$ &       & $",str.format('{0:.2f}',lInf[0]*10**-magInfOrder),"\\cdot 10^{", str.format('{0:n}',magInfOrder),end=' ')
    print("}$ &  \\\\")
    print("\\hline")
    for i in range(nconvpts-1):
        n=npt*2**(i+1)
        mag2Order = np.floor(np.log10(l2[i+1]))
        magInfOrder = np.floor(np.log10(lInf[i+1]))
        print(deg," & & ",n,"    & & $",end=' ')
        print(str.format('{0:.2f}',l2[i+1]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
        print("}$ & ",str.format('{0:.2f}',l2Order[i])," & $",end=' ')
        print(str.format('{0:.2f}',lInf[i+1]*10**-magInfOrder),"\\cdot 10^{", str.format('{0:n}',magInfOrder),end=' ')
        print("}$ & ",str.format('{0:.2f}',lInfOrder[i])," \\\\")
        print("\\hline")

@pytest.mark.serial
def test_grad_degreeConverge():
    deg = 1
    # splev only works for splines of degree <=5
    nconvpts = 3
    
    npts = [512,32,4]
    domain = [[1,7],[0,2*pi],[0,1]]
    degree = [deg,3,3]
    period = [False,True,False]
    comm = MPI.COMM_WORLD
    
    l2 = np.empty(nconvpts)
    lInf = np.empty(nconvpts)
    
    for c in range(nconvpts):
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
        
        ps.solveEquation(phi,rho)
        
        err=(phi._f-phi_exact._f)[0,0]
        l2[c] = np.sqrt(trapz(np.real(err*err.conj()),r))
        lInf[c] = np.max(np.abs(phi._f-phi_exact._f))
        
        degree[0]*=2
    
    print(l2)
    print(lInf)
    
    l2Order = np.log2(l2[:-1]/l2[1:])
    lInfOrder = np.log2(lInf[:-1]/lInf[1:])
    
    print(l2Order)
    print(lInfOrder)
    
    print(" ")
    print(deg," & & ",npts[0],"    & & $",end=' ')
    mag2Order = np.floor(np.log10(l2[0]))
    magInfOrder = np.floor(np.log10(lInf[0]))
    print(str.format('{0:.2f}',l2[0]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
    print("}$ &       & $",str.format('{0:.2f}',lInf[0]*10**-magInfOrder),"\\cdot 10^{", str.format('{0:n}',magInfOrder),end=' ')
    print("}$ &  \\\\")
    print("\\hline")
    for i in range(nconvpts-1):
        n=deg*2**(i+1)
        mag2Order = np.floor(np.log10(l2[i+1]))
        magInfOrder = np.floor(np.log10(lInf[i+1]))
        print(n," & & ",npts[0],"    & & $",end=' ')
        print(str.format('{0:.2f}',l2[i+1]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
        print("}$ & ",str.format('{0:.2f}',l2Order[i])," & $",end=' ')
        print(str.format('{0:.2f}',lInf[i+1]*10**-magInfOrder),"\\cdot 10^{", str.format('{0:n}',magInfOrder),end=' ')
        print("}$ & ",str.format('{0:.2f}',lInfOrder[i])," \\\\")
        print("\\hline")

@pytest.mark.serial
@pytest.mark.parametrize( "deg", [1,2,3,4,5] )
def test_grad_r_pointConverge(deg):
    npt = 8
    nconvpts = 7
    
    npts = [npt,8,4]
    domain = [[1,9],[0,2*pi],[0,1]]
    degree = [deg,3,3]
    period = [False,True,False]
    comm = MPI.COMM_WORLD
    
    l2 = np.empty(nconvpts)
    lInf = np.empty(nconvpts)
    
    for c in range(nconvpts):
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
        
        ps = PoissonSolver(eta_grid,2*deg,bsplines[0],ddrFactor=0,drFactor=1/r,rFactor=0,ddThetaFactor=0)
        phi=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
        phi_exact=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
        rho=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
        
        for i,q in rho.getCoords(0):
            plane = rho.get2DSlice([i])
            plane[:]=a*np.cos(a*(r-domain[0][0]))/r
            plane = phi_exact.get2DSlice([i])
            plane[:] = np.sin(a*(r-domain[0][0]))
        
        q = eta_grid[1]
        ps.solveEquation(phi,rho)
        
        err=(phi._f-phi_exact._f)[0,0]
        l2[c] = np.sqrt(trapz(np.real(err*err.conj()),r))
        lInf[c] = np.max(np.abs(phi._f-phi_exact._f))
        
        npts[0]*=2
    
    l2Order = np.log2(l2[:-1]/l2[1:])
    lInfOrder = np.log2(lInf[:-1]/lInf[1:])
    
    print(" ")
    print(deg," & & ",npt,"    & & $",end=' ')
    mag2Order = np.floor(np.log10(l2[0]))
    magInfOrder = np.floor(np.log10(lInf[0]))
    print(str.format('{0:.2f}',l2[0]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
    print("}$ &       & $",str.format('{0:.2f}',lInf[0]*10**-magInfOrder),"\\cdot 10^{", str.format('{0:n}',magInfOrder),end=' ')
    print("}$ &  \\\\")
    print("\\hline")
    for i in range(nconvpts-1):
        n=npt*2**(i+1)
        mag2Order = np.floor(np.log10(l2[i+1]))
        magInfOrder = np.floor(np.log10(lInf[i+1]))
        print(deg," & & ",n,"    & & $",end=' ')
        print(str.format('{0:.2f}',l2[i+1]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
        print("}$ & ",str.format('{0:.2f}',l2Order[i])," & $",end=' ')
        print(str.format('{0:.2f}',lInf[i+1]*10**-magInfOrder),"\\cdot 10^{", str.format('{0:n}',magInfOrder),end=' ')
        print("}$ & ",str.format('{0:.2f}',lInfOrder[i])," \\\\")
        print("\\hline")

@pytest.mark.serial
def test_grad_r_degreeConverge():
    deg = 1
    # splev only works for splines of degree <=5
    nconvpts = 3
    
    npts = [512,32,4]
    domain = [[1,7],[0,2*pi],[0,1]]
    degree = [deg,3,3]
    period = [False,True,False]
    comm = MPI.COMM_WORLD
    
    l2 = np.empty(nconvpts)
    lInf = np.empty(nconvpts)
    
    for c in range(nconvpts):
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
        
        ps = PoissonSolver(eta_grid,2*deg,bsplines[0],ddrFactor=0,drFactor=1/r,rFactor=0,ddThetaFactor=0)
        phi=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
        phi_exact=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
        rho=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
        
        for i,q in rho.getCoords(0):
            plane = rho.get2DSlice([i])
            plane[:]=a*np.cos(a*(r-domain[0][0]))/r
            plane = phi_exact.get2DSlice([i])
            plane[:] = np.sin(a*(r-domain[0][0]))
        
        ps.solveEquation(phi,rho)
        
        err=(phi._f-phi_exact._f)[0,0]
        l2[c] = np.sqrt(trapz(np.real(err*err.conj()),r))
        lInf[c] = np.max(np.abs(phi._f-phi_exact._f))
        
        degree[0]*=2
    
    print(l2)
    print(lInf)
    
    l2Order = np.log2(l2[:-1]/l2[1:])
    lInfOrder = np.log2(lInf[:-1]/lInf[1:])
    
    print(l2Order)
    print(lInfOrder)
    
    print(" ")
    print(deg," & & ",npts[0],"    & & $",end=' ')
    mag2Order = np.floor(np.log10(l2[0]))
    magInfOrder = np.floor(np.log10(lInf[0]))
    print(str.format('{0:.2f}',l2[0]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
    print("}$ &       & $",str.format('{0:.2f}',lInf[0]*10**-magInfOrder),"\\cdot 10^{", str.format('{0:n}',magInfOrder),end=' ')
    print("}$ &  \\\\")
    print("\\hline")
    for i in range(nconvpts-1):
        n=deg*2**(i+1)
        mag2Order = np.floor(np.log10(l2[i+1]))
        magInfOrder = np.floor(np.log10(lInf[i+1]))
        print(n," & & ",npts[0],"    & & $",end=' ')
        print(str.format('{0:.2f}',l2[i+1]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
        print("}$ & ",str.format('{0:.2f}',l2Order[i])," & $",end=' ')
        print(str.format('{0:.2f}',lInf[i+1]*10**-magInfOrder),"\\cdot 10^{", str.format('{0:n}',magInfOrder),end=' ')
        print("}$ & ",str.format('{0:.2f}',lInfOrder[i])," \\\\")
        print("\\hline")

@pytest.mark.serial
@pytest.mark.parametrize( "deg", [1,2,3,4,5] )
def test_ddTheta(deg):
    npt = 8
    nconvpts = 7
    
    npts = [8,npt,4]
    domain = [[1,8],[0,2*pi],[0,1]]
    degree = [deg,3,3]
    period = [False,True,False]
    comm = MPI.COMM_WORLD
    
    l2 = np.empty(nconvpts)
    lInf = np.empty(nconvpts)
    
    for c in range(nconvpts):
        # Compute breakpoints, knots, spline space and grid points
        nkts     = [n+1+d*(int(p)-1)              for (n,d,p)    in zip( npts,degree, period )]
        breaks   = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
        knots    = [spl.make_knots( b,d,p )       for (b,d,p)    in zip( breaks, degree, period )]
        bsplines = [spl.BSplines( k,d,p )         for (k,d,p)    in zip(  knots, degree, period )]
        eta_grid = [bspl.greville                 for bspl       in bsplines]
        
        layout_poisson = {'mode_solve': [1,2,0],
                          'v_parallel': [0,2,1]}
        remapper = getLayoutHandler(comm,layout_poisson,[comm.Get_size()],eta_grid)
        
        ps = PoissonSolver(eta_grid,2*deg,bsplines[0],ddrFactor=0,drFactor=0,rFactor=1,ddThetaFactor=-1,lBoundary='neumann',rBoundary='neumann')
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
        
        #~ phi._f=phi._f*2
        
        err=(phi._f-phi_exact._f)[0,0]
        l2[c] = np.sqrt(trapz(np.real(err)**2,q))
        lInf[c] = np.max(np.abs(np.real(phi._f-phi_exact._f)))
        
        npts[1]*=2
    
    l2Order = np.log2(l2[:-1]/l2[1:])
    lInfOrder = np.log2(lInf[:-1]/lInf[1:])
    
    print(" ")
    print(deg," & & ",npt,"    & & $",end=' ')
    mag2Order = np.floor(np.log10(l2[0]))
    magInfOrder = np.floor(np.log10(lInf[0]))
    print(str.format('{0:.2f}',l2[0]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
    print("}$ &       & $",str.format('{0:.2f}',lInf[0]*10**-magInfOrder),"\\cdot 10^{", str.format('{0:n}',magInfOrder),end=' ')
    print("}$ &  \\\\")
    print("\\hline")
    for i in range(nconvpts-1):
        n=npt*2**(i+1)
        mag2Order = np.floor(np.log10(l2[i+1]))
        magInfOrder = np.floor(np.log10(lInf[i+1]))
        print(deg," & & ",n,"    & & $",end=' ')
        print(str.format('{0:.2f}',l2[i+1]*10**-mag2Order),"\\cdot 10^{", str.format('{0:n}',mag2Order),end=' ')
        print("}$ & ",str.format('{0:.2f}',l2Order[i])," & $",end=' ')
        print(str.format('{0:.2f}',lInf[i+1]*10**-magInfOrder),"\\cdot 10^{", str.format('{0:n}',magInfOrder),end=' ')
        print("}$ & ",str.format('{0:.2f}',lInfOrder[i])," \\\\")
