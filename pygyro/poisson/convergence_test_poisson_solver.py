from mpi4py                 import MPI
import numpy                as np
import pytest
from math                   import pi
from scipy.integrate        import trapz
from numpy.polynomial.legendre      import leggauss

import matplotlib.pyplot    as plt
from matplotlib             import rc

from ..model.layout                 import getLayoutHandler
from ..model.grid                   import Grid
from ..initialisation               import mod_initialiser_funcs as initialiser
from ..initialisation.constants     import get_constants
from ..                             import splines as spl
from .poisson_solver                import DiffEqSolver
from ..splines.splines              import Spline1D
from ..splines.spline_interpolators import SplineInterpolator1D

@pytest.mark.serial
@pytest.mark.parametrize( "deg", [1,2,3,4,5] )
def test_BasicPoissonEquation_pointConverge(deg):
    npt = 8
    nconvpts = 8

    npts = [npt,8,4]
    domain = [[0,1],[0,2*pi],[0,1]]
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

        ps = DiffEqSolver(2*deg+1,bsplines[0],eta_grid[0].size,
                            eta_grid[1].size,drFactor=lambda r:0,
                            rFactor=lambda r:0,ddThetaFactor=lambda r:0)

        phi=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
        phi_exact=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
        rho=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)

        r = eta_grid[0]

        a=2*pi/(domain[0][1]-domain[0][0])

        for i,q in rho.getCoords(0):
            plane = rho.get2DSlice([i])
            plane[:]= np.sin(a*(r-domain[0][0]))*a*a
            plane = phi_exact.get2DSlice([i])
            plane[:] = np.sin(a*(r-domain[0][0]))

        ps.solveEquation(phi,rho)
        #~ ps.solveEquationForFunction(phi,lambda r: a*a*np.sin(a*(r-domain[0][0])))

        rspline=bsplines[0]

        points,weights = leggauss(deg+1)
        multFactor = (rspline.breaks[1]-rspline.breaks[0])*0.5
        startPoints = (rspline.breaks[1:]+rspline.breaks[:-1])*0.5
        evalPts = (startPoints[:,None]+points[None,:]*multFactor).flatten()
        weights = np.tile(weights,startPoints.size)

        approxSpline = Spline1D(rspline)
        exactSpline = Spline1D(rspline)
        interp = SplineInterpolator1D(rspline)

        assert(np.max(np.abs(np.imag(phi.get1DSlice([0,0]))))<1e-16)

        interp.compute_interpolant(np.real(phi.get1DSlice([0,0])),approxSpline)
        interp.compute_interpolant(np.real(phi_exact.get1DSlice([0,0])),exactSpline)

        l2[c] = np.sqrt(np.sum((approxSpline.eval(evalPts)-np.sin(a*(evalPts-domain[0][0])))**2 \
                        * multFactor*weights))
        lInf[c] = np.max(np.abs(approxSpline.eval(evalPts)-np.sin(a*(evalPts-domain[0][0]))))

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
    a=2*pi/(domain[0][1]-domain[0][0])

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

        ps = DiffEqSolver(2*degree[0]+1,bsplines[0],eta_grid[0].size,
                            eta_grid[1].size,drFactor=lambda r:0,
                            rFactor=lambda r:0,ddThetaFactor=lambda r:0)

        phi=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
        phi_exact=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
        rho=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)

        r = eta_grid[0]

        for i,q in rho.getCoords(0):
            plane = rho.get2DSlice([i])
            plane[:]= np.sin(a*(r-domain[0][0]))*a*a
            plane = phi_exact.get2DSlice([i])
            plane[:] = np.sin(a*(r-domain[0][0]))

        ps.solveEquation(phi,rho)

        rspline=bsplines[0]

        points,weights = leggauss(deg+1)
        multFactor = (rspline.breaks[1]-rspline.breaks[0])*0.5
        startPoints = (rspline.breaks[1:]+rspline.breaks[:-1])*0.5
        evalPts = (startPoints[:,None]+points[None,:]*multFactor).flatten()
        weights = np.tile(weights,startPoints.size)

        approxSpline = Spline1D(rspline)
        exactSpline = Spline1D(rspline)
        interp = SplineInterpolator1D(rspline)

        assert(np.max(np.abs(np.imag(phi.get1DSlice([0,0]))))<1e-16)

        interp.compute_interpolant(np.real(phi.get1DSlice([0,0])),approxSpline)
        interp.compute_interpolant(np.real(phi_exact.get1DSlice([0,0])),exactSpline)

        l2[c] = np.sqrt(np.sum((approxSpline.eval(evalPts)-np.sin(a*(evalPts-domain[0][0])))**2 \
                        * multFactor*weights))
        lInf[c] = np.max(np.abs(approxSpline.eval(evalPts)-np.sin(a*(evalPts-domain[0][0]))))

        degree[0]*=2

    l2Order = np.log2(l2[:-1]/l2[1:])
    lInfOrder = np.log2(lInf[:-1]/lInf[1:])

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
    nconvpts = 8

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

        ps = DiffEqSolver(2*deg+1,bsplines[0],eta_grid[0].size,
                        eta_grid[1].size,ddrFactor=lambda r:0,
                        drFactor=lambda r:1,rFactor=lambda r:1,ddThetaFactor=lambda r:0)
        phi=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
        phi_exact=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
        rho=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)

        r = eta_grid[0]

        for i,q in rho.getCoords(0):
            plane = rho.get2DSlice([i])
            plane[:]=a*np.cos(a*(r-domain[0][0]))+np.sin(a*(r-domain[0][0]))
            plane = phi_exact.get2DSlice([i])
            plane[:] = np.sin(a*(r-domain[0][0]))

        q = eta_grid[1]
        ps.solveEquation(phi,rho)

        rspline=bsplines[0]

        points,weights = leggauss(deg+1)
        multFactor = (rspline.breaks[1]-rspline.breaks[0])*0.5
        startPoints = (rspline.breaks[1:]+rspline.breaks[:-1])*0.5
        evalPts = (startPoints[:,None]+points[None,:]*multFactor).flatten()
        weights = np.tile(weights,startPoints.size)

        approxSpline = Spline1D(rspline)
        exactSpline = Spline1D(rspline)
        interp = SplineInterpolator1D(rspline)

        assert(np.max(np.abs(np.imag(phi.get1DSlice([0,0]))))<1e-16)

        interp.compute_interpolant(np.real(phi.get1DSlice([0,0])),approxSpline)
        interp.compute_interpolant(np.real(phi_exact.get1DSlice([0,0])),exactSpline)

        l2[c] = np.sqrt(np.sum((approxSpline.eval(evalPts)-np.sin(a*(evalPts-domain[0][0])))**2 \
                        * multFactor*weights))
        lInf[c] = np.max(np.abs(approxSpline.eval(evalPts)-np.sin(a*(evalPts-domain[0][0]))))

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

        ps = DiffEqSolver(2*degree[0]+1,bsplines[0],eta_grid[0].size,
                eta_grid[1].size,ddrFactor=lambda r:0,
                drFactor=lambda r:1,rFactor=lambda r:1,ddThetaFactor=lambda r:0)
        phi=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
        phi_exact=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
        rho=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)

        r = eta_grid[0]

        for i,q in rho.getCoords(0):
            plane = rho.get2DSlice([i])
            plane[:]=a*np.cos(a*(r-domain[0][0]))+np.sin(a*(r-domain[0][0]))
            plane = phi_exact.get2DSlice([i])
            plane[:] = np.sin(a*(r-domain[0][0]))

        ps.solveEquation(phi,rho)

        rspline=bsplines[0]

        points,weights = leggauss(deg+1)
        multFactor = (rspline.breaks[1]-rspline.breaks[0])*0.5
        startPoints = (rspline.breaks[1:]+rspline.breaks[:-1])*0.5
        evalPts = (startPoints[:,None]+points[None,:]*multFactor).flatten()
        weights = np.tile(weights,startPoints.size)

        approxSpline = Spline1D(rspline)
        exactSpline = Spline1D(rspline)
        interp = SplineInterpolator1D(rspline)

        assert(np.max(np.abs(np.imag(phi.get1DSlice([0,0]))))<1e-16)

        interp.compute_interpolant(np.real(phi.get1DSlice([0,0])),approxSpline)
        interp.compute_interpolant(np.real(phi_exact.get1DSlice([0,0])),exactSpline)

        l2[c] = np.sqrt(np.sum((approxSpline.eval(evalPts)-np.sin(a*(evalPts-domain[0][0])))**2 \
                        * multFactor*weights))
        lInf[c] = np.max(np.abs(approxSpline.eval(evalPts)-np.sin(a*(evalPts-domain[0][0]))))

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

        ps = DiffEqSolver(2*deg+1,bsplines[0],eta_grid[0].size,
                eta_grid[1].size,ddrFactor=lambda r:0,
                drFactor=lambda r:1/r,rFactor=lambda r:1,ddThetaFactor=lambda r:0)
        phi=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
        phi_exact=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
        rho=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)

        for i,_ in rho.getCoords(0): # q
            plane = rho.get2DSlice([i])
            plane[:]=a*np.cos(a*(r-domain[0][0]))/r+np.sin(a*(r-domain[0][0]))
            plane = phi_exact.get2DSlice([i])
            plane[:] = np.sin(a*(r-domain[0][0]))

        ps.solveEquation(phi,rho)

        rspline=bsplines[0]

        points,weights = leggauss(deg+1)
        multFactor = (rspline.breaks[1]-rspline.breaks[0])*0.5
        startPoints = (rspline.breaks[1:]+rspline.breaks[:-1])*0.5
        evalPts = (startPoints[:,None]+points[None,:]*multFactor).flatten()
        weights = np.tile(weights,startPoints.size)

        approxSpline = Spline1D(rspline)
        exactSpline = Spline1D(rspline)
        interp = SplineInterpolator1D(rspline)

        assert(np.max(np.abs(np.imag(phi.get1DSlice([0,0]))))<1e-16)

        interp.compute_interpolant(np.real(phi.get1DSlice([0,0])),approxSpline)
        interp.compute_interpolant(np.real(phi_exact.get1DSlice([0,0])),exactSpline)

        l2[c] = np.sqrt(np.sum((approxSpline.eval(evalPts)-np.sin(a*(evalPts-domain[0][0])))**2 \
                        * multFactor*weights))
        lInf[c] = np.max(np.abs(approxSpline.eval(evalPts)-np.sin(a*(evalPts-domain[0][0]))))

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

        ps = DiffEqSolver(2*degree[0]+1,bsplines[0],eta_grid[0].size,
                eta_grid[1].size,ddrFactor=lambda r:0,
                drFactor=lambda r:1/r,rFactor=lambda r:1,ddThetaFactor=lambda r:0)
        phi=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
        phi_exact=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
        rho=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)

        for i,_ in rho.getCoords(0):
            plane = rho.get2DSlice([i])
            plane[:]=a*np.cos(a*(r-domain[0][0]))/r+np.sin(a*(r-domain[0][0]))
            plane = phi_exact.get2DSlice([i])
            plane[:] = np.sin(a*(r-domain[0][0]))

        ps.solveEquation(phi,rho)

        rspline=bsplines[0]

        points,weights = leggauss(deg+1)
        multFactor = (rspline.breaks[1]-rspline.breaks[0])*0.5
        startPoints = (rspline.breaks[1:]+rspline.breaks[:-1])*0.5
        evalPts = (startPoints[:,None]+points[None,:]*multFactor).flatten()
        weights = np.tile(weights,startPoints.size)

        approxSpline = Spline1D(rspline)
        exactSpline = Spline1D(rspline)
        interp = SplineInterpolator1D(rspline)

        assert(np.max(np.abs(np.imag(phi.get1DSlice([0,0]))))<1e-16)

        interp.compute_interpolant(np.real(phi.get1DSlice([0,0])),approxSpline)
        interp.compute_interpolant(np.real(phi_exact.get1DSlice([0,0])),exactSpline)

        l2[c] = np.sqrt(np.sum((approxSpline.eval(evalPts)-np.sin(a*(evalPts-domain[0][0])))**2 \
                        * multFactor*weights))
        lInf[c] = np.max(np.abs(approxSpline.eval(evalPts)-np.sin(a*(evalPts-domain[0][0]))))

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

        mVals = np.fft.fftfreq(eta_grid[1].size,1/eta_grid[1].size)

        ps = DiffEqSolver(2*degree[0]+1,bsplines[0],eta_grid[0].size,
                            eta_grid[1].size,ddrFactor=lambda r:0,
                            drFactor=lambda r:0,rFactor=lambda r:1,
                            ddThetaFactor=lambda r:-1,lNeumannIdx=mVals,uNeumannIdx=mVals)
        phi=Grid(eta_grid,bsplines,remapper,'mode_solve',comm,dtype=np.complex128)
        phi_exact=Grid(eta_grid,bsplines,remapper,'v_parallel',comm,dtype=np.complex128)
        rho=Grid(eta_grid,bsplines,remapper,'v_parallel',comm,dtype=np.complex128)

        q = eta_grid[1]

        for i,_ in rho.getCoords(0):
            plane = rho.get2DSlice([i])
            plane[:] = -6*np.sin(q)*np.cos(q)**2+4*np.sin(q)**3
            plane = phi_exact.get2DSlice([i])
            plane[:] = np.sin(q)**3

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

@pytest.mark.serial
#@pytest.mark.parametrize( "deg", [1,2,3,4,5] )
#def test_QuasiNeutralityEquation_pointConverge(deg):
def test_QuasiNeutralityEquation_pointConverge():
    font = {'size'   : 16}

    constants = get_constants('testSetups/iota0.json')

    rc('font', **font)
    rc('text', usetex=True)
    for deg,leg,nconvpts in zip(range(1,5),('1st degree','2nd degree','3rd degree','4th degree'),(8,8,8,7)):
        npt = 16

        npts = [npt,8,4]
        domain = [[1,15],[0,2*pi],[0,1]]
        degree = [deg,3,3]
        period = [False,True,False]
        comm = MPI.COMM_WORLD

        l2 = np.empty(nconvpts)
        lInf = np.empty(nconvpts)

        for c in range(nconvpts):
            # Compute breakpoints, knots, spline space and grid points
            nkts     = [n+1+d+d*(int(p)-1)              for (n,d,p)    in zip( npts,degree, period )]
            breaks   = [np.linspace( *lims, num=num ) for (lims,num) in zip( domain, nkts )]
            knots    = [spl.make_knots( b,d,p )       for (b,d,p)    in zip( breaks, degree, period )]
            bsplines = [spl.BSplines( k,d,p )         for (k,d,p)    in zip(  knots, degree, period )]
            eta_grid = [bspl.greville                 for bspl       in bsplines]

            layout_poisson = {'mode_solve': [1,2,0],
                              'v_parallel': [0,2,1]}
            remapper = getLayoutHandler(comm,layout_poisson,[comm.Get_size()],eta_grid)

            mVals = np.fft.fftfreq(eta_grid[1].size,1/eta_grid[1].size)

            ps = DiffEqSolver(100,bsplines[0],eta_grid[0].size,
                    eta_grid[1].size,lNeumannIdx=mVals,
                    ddrFactor = lambda r:-1,
                    drFactor = lambda r:-( 1/r + initialiser.n0derivNormalised(r,constants.kN0,constants.rp,constants.deltaRN0)),
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
                #plane[:] = -12*np.cos(rArg)**2*np.sin(rArg)**2*a*a \
                #           + 4*np.cos(rArg)**4                *a*a \
                #           + (1/r - constants.kN0*(1-np.tanh((r-constants.rp)/constants.deltaRN0)**2)) * \
                #           4 * np.cos(rArg)**3*np.sin(rArg)*a \
                #           + np.cos(rArg)**4 / initialiser.Te(r,constants.CTe,constants.kTe,constants.deltaRTe,constants.rp)
                #plane = phi_exact.get2DSlice([i])
                #plane[:] = np.cos(rArg)**4

            ps.getModes(rho)

            rho.setLayout('mode_solve')

            ps.solveEquation(phi,rho)

            phi.setLayout('v_parallel')
            ps.findPotential(phi)

            r = eta_grid[0]

            rspline=bsplines[0]
            points,weights = leggauss(deg+1)
            multFactor = (rspline.breaks[1]-rspline.breaks[0])*0.5
            startPoints = (rspline.breaks[1:]+rspline.breaks[:-1])*0.5
            evalPts = (startPoints[:,None]+points[None,:]*multFactor).flatten()
            weights = np.tile(weights,startPoints.size)

            approxSpline = Spline1D(rspline)
            interp = SplineInterpolator1D(rspline)

            assert(np.max(np.abs(np.imag(phi.get1DSlice([0,0]))))<1e-16)

            phi.setLayout('mode_solve')

            rArg = a*(evalPts-domain[0][0])

            lI=0
            l2Q = np.zeros(eta_grid[1].size)
            for i,q in enumerate(eta_grid[1]):
                interp.compute_interpolant(np.real(phi.get1DSlice([i,0])),approxSpline)
                m = np.max(np.abs(approxSpline.eval(evalPts)-np.cos(rArg)**4*np.sin(q)**3))
                #~ m = np.max(np.abs(approxSpline.eval(evalPts)-np.cos(rArg)**4))
                if (m>lI):
                    lI=m
                l2Q[i]=np.sum((approxSpline.eval(evalPts)-np.cos(rArg)**4*np.sin(q)**3)**2 \
                            * multFactor*weights)
                #~ l2Q[i]=np.sum((approxSpline.eval(evalPts)-np.cos(rArg)**4)**2 \
                            #~ * multFactor*weights)

            l2[c] = np.sqrt(trapz(l2Q,eta_grid[1]))
            lInf[c] = lI

            npts[0]*=2

        plt.figure(1)
        plt.loglog(npt*2**np.arange(0,nconvpts),l2,'x',label=leg,markersize=10)
        if (degree[0]!=4):
            plt.loglog(npt*2**np.arange(0,nconvpts),l2[-1]*np.power(
                        degree[0]+1,2*np.arange(nconvpts-1,-1,-1)),
                        label='order {0}'.format(degree[0]+1),markersize=10)
        else:
            plt.loglog(npt*2**np.arange(0,nconvpts),l2[0]/np.power(
                        degree[0]+1,2*np.arange(0,nconvpts)),
                        label='order {0}'.format(degree[0]+1),markersize=10)
        plt.figure(2)
        plt.loglog(npt*2**np.arange(0,nconvpts),lInf,'x',label=leg,markersize=10)
        if (degree[0]!=4):
            plt.loglog(npt*2**np.arange(0,nconvpts),lInf[-1]*np.power(
                        degree[0]+1,2*np.arange(nconvpts-1,-1,-1)),
                        label='order {0}'.format(degree[0]+1),markersize=10)
        else:
            plt.loglog(npt*2**np.arange(0,nconvpts),lInf[0]/np.power(
                        degree[0]+1,2*np.arange(0,nconvpts)),
                        label='order {0}'.format(degree[0]+1),markersize=10)
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

    plt.figure(1)
    plt.legend()
    plt.xlabel('N')
    plt.ylabel('l$_2$ norm')

    plt.figure(2)
    plt.legend()
    plt.xlabel('N')
    plt.ylabel('l$_{\inf}$ norm')
    plt.show()

@pytest.mark.serial
def test_QuasiNeutralityEquation_degreeConverge():
    deg = 1
    # splev only works for splines of degree <=5
    nconvpts = 3

    constants = get_constants('testSetups/iota0.json')

    npts = [512,8,4]
    domain = [[1,15],[0,2*pi],[0,1]]
    degree = [deg,3,3]
    comm = MPI.COMM_WORLD
    period = [False,True,False]

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

        rspline=bsplines[0]
        points,weights = leggauss(deg+1)
        multFactor = (rspline.breaks[1]-rspline.breaks[0])*0.5
        startPoints = (rspline.breaks[1:]+rspline.breaks[:-1])*0.5
        evalPts = (startPoints[:,None]+points[None,:]*multFactor).flatten()
        weights = np.tile(weights,startPoints.size)

        approxSpline = Spline1D(rspline)
        interp = SplineInterpolator1D(rspline)

        assert(np.max(np.abs(np.imag(phi.get1DSlice([0,0]))))<1e-16)

        phi.setLayout('mode_solve')

        rArg = a*(evalPts-domain[0][0])

        lI=0
        l2Q = np.zeros(npts[1])
        for i,q in enumerate(eta_grid[1]):
            interp.compute_interpolant(np.real(phi.get1DSlice([i,0])),approxSpline)
            #~ m = np.max(np.abs(approxSpline.eval(evalPts)-np.cos(rArg)**4*np.sin(q)**3))
            m = np.max(np.abs(approxSpline.eval(evalPts)-np.cos(rArg)**4))
            if (m>lI):
                lI=m
            #~ plt.figure()
            #~ plt.plot(evalPts,approxSpline.eval(evalPts),label='approx')
            #~ plt.plot(evalPts,np.cos(rArg)**4,label='exact')
            #~ #~ plt.plot(evalPts,np.cos(rArg)**4*np.sin(q)**3,label='exact')
            #~ plt.legend()
            #~ plt.show()
            #~ l2Q[i]=np.sum((approxSpline.eval(evalPts)-np.cos(rArg)**4*np.sin(q)**3)**2 \
                        #~ * multFactor*weights)
            l2Q[i]=np.sum((approxSpline.eval(evalPts)-np.cos(rArg)**4)**2 \
                        * multFactor*weights)

        l2[c] = np.sqrt(trapz(l2Q,eta_grid[1]))
        lInf[c] = lI

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
