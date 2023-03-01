import numpy as np
from pygyro.initialisation.constants import Constants
from pygyro.initialisation.initialiser_funcs import init_f_vpar
from pygyro import splines as spl
import time

constants = Constants()

npts = [1,1,256,64]

domain = [[constants.rMin, constants.rMax], [0, 2*np.pi],
          [constants.zMin, constants.zMax], [constants.vMin, constants.vMax]]
degree = constants.splineDegrees
period = [False, True, True, False]

# Compute breakpoints, knots, spline space and grid points
nkts = [n+1+d*(int(p)-1)
        for (n, d, p) in zip(npts, degree, period)]
breaks = [(constants.rMax+constants.rMin)/2, constants.zMax/2] + [np.linspace(*lims, num=num) for (lims, num) in zip(domain[2:], nkts[2:])]
knots = [spl.make_knots(b, d, p)
         for (b, d, p) in zip(breaks[2:], degree[2:], period[2:])]
bsplines = [spl.BSplines(k, d, p, True)
        for (k, d, p) in zip(knots, degree[2:], period[2:])]
eta_grids = [bspl.greville for bspl in bsplines]

spl_interp = spl.SplineInterpolator1D(bsplines[-1])
spline = spl.Spline1D(bsplines[-1])

f = np.empty(npts)
init_f_vpar(f[0,0], breaks[0], eta_grids[0], breaks[1], eta_grids[1],
                        constants.m, constants.n, constants.eps,
                        constants.CN0, constants.kN0, constants.deltaRN0,
                        constants.rp, constants.CTi, constants.kTi,
                        constants.deltaRTi, constants.deltaR, constants.R0)

c = np.empty((1,1,npts[2],spline._coeffs.size))

start = time.time()
c[:], spl_interp._sinfo = spl_interp._solveFunc(
        spl_interp._bmat, spl_interp._l, spl_interp._u, f[0,0,:,:], spl_interp._ipiv)
end = time.time()

multi_time = end-start

start = time.time()
for i in range(npts[2]):
    spl_interp.compute_interpolant(f[0,0,i,:], spline)
end = time.time()
spl_time = end-start

print(multi_time, spl_time)
