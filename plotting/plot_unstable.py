import argparse
import glob
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import os
from fourier2d import Fourier2D

from pygyro import splines as spl
from pygyro.initialisation.constants import get_constants
from pygyro.model.process_grid import compute_2d_process_grid
from pygyro.model.grid import Grid
from pygyro.model.layout import LayoutSwapper


parser = argparse.ArgumentParser(
    description='Plot the most unstable modes of the simulation')
parser.add_argument('foldername', type=str,
                    help='The folders whose results should be plotted')
foldername = args.foldername

###############################################################
#                   Setup to handle layouts
###############################################################

comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()

filename = os.path.join(foldername, "initParams.json")
constants = get_constants(filename)

npts = constants.npts

degree = constants.splineDegrees[:-1]
period = [False, True, True]
domain = [[constants.rMin, constants.rMax], [
    0, 2*np.pi], [constants.zMin, constants.zMax]]

nkts = [n+1+d*(int(p)-1) for (n, d, p) in zip(npts, degree, period)]
breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
knots = [spl.make_knots(b, d, p)
         for (b, d, p) in zip(breaks, degree, period)]
bsplines = [spl.BSplines(k, d, p)
            for (k, d, p) in zip(knots, degree, period)]
eta_grids = [bspl.greville for bspl in bsplines]

layout_poisson = {'v_parallel_2d': [0, 2, 1],
                  'mode_solve': [1, 2, 0]}
layout_vpar = {'v_parallel_1d': [0, 2, 1]}
layout_poloidal = {'poloidal': [2, 1, 0]}

nprocs = compute_2d_process_grid(npts, mpi_size)

remapperPhi = LayoutSwapper(comm, [layout_poisson, layout_vpar, layout_poloidal],
                            [nprocs, nprocs[0], nprocs[1]], eta_grids,
                            'v_parallel_2d')

phi = Grid(eta_grids, bsplines, remapperPhi,
           'v_parallel_2d', comm, dtype=np.complex128)

r_idx = npts[0]//2


###############################################################

theta = phi.eta_grid[1]
z = phi.eta_grid[2]

Ntheta = theta.size
Nz = z.size

phi_files = glob.glob(os.path.join(foldername, 'phi_*.h5'))
phi_files = [os.path.basename(f) for f in phi_files]
time_grid = np.array([int(f[4:-3]) for f in phi_files])
n_time_steps = len(time_grid)
assert n_time_steps > 0

time_grid.sort()

m_max = 4
nb_mn_unstable = 7
ktheta0 = int(Ntheta//2)
kphi0 = int(Nz//2)

Abs_modes_m0 = np.zeros((m_max+1, n_time_steps))
Abs_modes_mn_unstable = np.zeros((nb_mn_unstable+1, n_time_steps))

phi.loadFromFile(foldername, 0, "phi")
assert r_idx in phi.getGlobalIdxVals(0)
Phi_FluxSurface = phi.get2DSlice(r_idx)
[TFPhi_mn, m2d, n2d] = Fourier2D(
    Phi_FluxSurface, z, theta)

# m0 labels
dic_kthetam = {}
dic_strm0 = {}
str_n0 = str(int(n2d[kphi0]))
for im in range(m_max+1):
    kthetam = ktheta0 + im
    dic_kthetam[im] = kthetam
    dic_strm0[im] = '(m,n) = {m},0'.format(m=int(m2d[kthetam]))
# end for

# Unstable labels
# --> The search of maximum is only done on half the spectrum
# -->  (due to symmetry)
# --> iphi_min excludes modes such that: |n| < iphi_min
iphi_min = 1
Abs_TFPhi = np.abs(TFPhi_mn[0:Nz//2+1-iphi_min, :])
max_abs_TFPhi = np.sort(np.amax(Abs_TFPhi, axis=0))

dic_ktheta_unstable = {}
dic_kphi_unstable = {}
dic_str_mn_unstable = {}
nb_max_found = 0
for imax in range(nb_mn_unstable+1):
    max_imax = max_abs_TFPhi[-1-nb_max_found]
    k_imax = np.nonzero(Abs_TFPhi == max_imax)
    max_found = len(k_imax[0])
    nb_max_found = nb_max_found + max_found
    ktheta_imax = k_imax[1][-1]
    kphi_imax = k_imax[0][-1]
    dic_ktheta_unstable[imax] = ktheta_imax
    dic_kphi_unstable[imax] = kphi_imax
    dic_str_mn_unstable[imax] = '(m,n) = {m},{n}'.format(
        m=int(m2d[ktheta_imax]),
        n=int(n2d[kphi_imax]))
# end for

for it, time in enumerate(time_grid[1:], 1):
    phi.loadFromFile(foldername, time, "phi")
    assert r_idx in phi.getGlobalIdxVals(0)
    Phi_FluxSurface = phi.get2DSlice(r_idx)
    [TFPhi_mn, m2d, n2d] = Fourier2D(
        Phi_FluxSurface, z, theta)
    for im in range(m_max+1):
        Abs_modes_m0[im, it] = np.abs(TFPhi_mn[kphi0, dic_kthetam[im]])
    for imax in range(nb_mn_unstable+1):
        Abs_modes_mn_unstable[imax, it] = np.abs(
            TFPhi_mn[dic_kphi_unstable[imax], dic_ktheta_unstable[imax]])
    # end for
# end for

fig = plt.figure(figsize=(8, 8))
color_str = ['k', 'k--', 'r', 'b', 'g', 'm', 'k', 'y', 'c', 'r--', 'b--']
# --> Plot (m,n)=(0,0)
if (np.min(Abs_modes_m0[0, :]) > 0.):
    plt.semilogy(time_grid[0:n_time_steps], Abs_modes_m0[0, :],
                 color_str[0], label=dic_strm0[0])
# --> Plot (m,n)=(1,0)
if (np.min(Abs_modes_m0[1, :]) > 0.):
    plt.semilogy(time_grid[0:n_time_steps], Abs_modes_m0[1, :],
                 color_str[1], label=dic_strm0[1])
# --> Plot the nb_mn_unstable most unstable (m,n) modes with n different from 0
for imax in np.arange(0, nb_mn_unstable+1):
    str_legend_mn = dic_str_mn_unstable[imax]
    plt.semilogy(time_grid[0:n_time_steps], Abs_modes_mn_unstable[imax, :],
                 color_str[2+imax], label=str_legend_mn)
# end for
plt.xlabel('time')
plt.ylabel(r'$abs(\Phi_{mn})(r)$')
plt.legend(loc=4)
#ax = plt.subplot(111)
#
#clevels = np.linspace( data.min(), data.max(), 101)
#im = ax.contourf( theta, z, data.T, clevels, cmap='jet' )
# for c in im.collections:
#    c.set_edgecolor('face')
#plt.colorbar( im )
#plt.title(f"t = {t}")
# plt.xlabel('$\\theta$')
# plt.ylabel('z')
plt.savefig(os.path.join(foldername, 'unstable_modes.png'))
