import os
import h5py
import numpy as np
from mpi4py import MPI

import matplotlib.pyplot as plt
from matplotlib import rc as pltFont

from pygyro import splines as spl
from pygyro.initialisation.setups import setupFromFile
from pygyro.initialisation.constants import get_constants
from pygyro.initialisation import initialiser_funcs as init
from pygyro.model.layout import LayoutSwapper
from pygyro.model.process_grid import compute_2d_process_grid
from pygyro.model.grid import Grid
from pygyro.initialisation.initialiser_funcs import n0, f_eq

def calc_energy_from_4d(foldername, conservations, ind, comm, z=None, v=None):

    mpi_size = comm.Get_size()
    tEnd = int(conservations[0, ind])
    distribFunc, constants, _ = setupFromFile(foldername,
                                              timepoint=tEnd, comm=comm)

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
    phi.loadFromFile(foldername, tEnd, "phi")

    
    distribFunc.setLayout('v_parallel')

    KEclass = KineticEnergy(
            distribFunc.eta_grid, distribFunc.getLayout('v_parallel'), constants)
    PEclass = PotentialEnergy(
            distribFunc.eta_grid, distribFunc.getLayout('v_parallel'), constants)

    conservations[1, ind] += KEclass.getKE(distribFunc)
    conservations[2, ind] += PEclass.getPE(distribFunc, phi)

def doall(foldername, z=None, v=None):
    filelist = os.listdir(foldername)

    t = []

    for fname in filelist:
        if fname.find("grid_") != -1:
            t_str = fname[fname.rfind('_')+1:fname.rfind('.')]
            t.append(int(t_str))

    # time, int f, int f^2, int f phi, ekin, epot
    conservations = np.zeros((3, len(t)))
    conservations[0, :] = t
    ekin = np.zeros(len(t))
    epot = np.zeros(len(t))

    comm = MPI.COMM_WORLD

    for i in range(len(t)):
        calc_energy_from_4d(foldername, conservations, i, comm, z=None, v=None)
        
    comm.Reduce(conservations[1, :],
                        ekin, op=MPI.SUM, root=0)
    comm.Reduce(conservations[2, :],
                        epot, op=MPI.SUM, root=0)

    if (comm.rank == 0):
        if (not os.path.isdir(foldername+"plots/")):
            os.mkdir(foldername+"plots/")
        if (not os.path.isdir(foldername+"plots_f/")):
            os.mkdir(foldername+"plots_f/")
        if (not os.path.isdir(foldername+"plots_phi/")):
            os.mkdir(foldername+"plots_phi/")

        with open(foldername+"plots/energies.txt", 'w') as savefile:
            savefile.write(
                    "t\t\t\tpotential energie\t\tkinetic energy\t\t\ttotal energy\n")

        perm = np.array(t).argsort()
        t_f = conservations[0, perm]
        ekin = ekin[perm]
        epot = epot[perm]
        en = ekin + epot 

        with open(foldername+"plots/energies.txt", 'a') as savefile:
            for i in range(len(t_f)):
                savefile.write(
                    format(t_f[i]) + "\t" + format(epot[i], '.15E')+ "\t" + format(ekin[i], '.15E') + "\t" + format(en[i], '.15E') + "\n")


        _, ax = plt.subplots(1)
        ax.set_title('Energies')
        ax.set_xlabel("t")
        ax.plot(t_f, ekin, label="Kinetic energy")
        ax.plot(t_f, epot, label="Potential energy")
        ax.plot(t_f, en, label="Total energy")
        plt.legend()
        plt.savefig(foldername+'plots/energies.png')
        plt.close()
    
        _, ax = plt.subplots(1)
        ax.set_title('Energies')
        ax.set_xlabel("t")
        dt = 2
        nsave = 5
        n = int(4000/(5*16))
        ax.plot(t_f[:n], ekin[:n], label="Kinetic energy")
        ax.plot(t_f[:n], epot[:n], label="Potential energy")
        ax.plot(t_f[:n], en[:n], label="Total energy")
        plt.legend()
        plt.savefig(foldername+'plots/energies_cut.png')
        plt.close()

class KineticEnergy:
    """
    TODO
    """

    def __init__(self, eta_grid: list, layout, constants):
        idx_r = layout.inv_dims_order[0]
        idx_v = layout.inv_dims_order[3]

        my_r = eta_grid[0][layout.starts[idx_r]:layout.ends[idx_r]]
        my_v = eta_grid[3][layout.starts[idx_v]:layout.ends[idx_v]]

        r = eta_grid[0]
        q = eta_grid[1]
        z = eta_grid[2]
        v = eta_grid[3]

        dr = r[1:] - r[:-1]
        dv = v[1:] - v[:-1]
        
        shape = [1, 1, 1, 1]
        shape[idx_r] = my_r.size
        shape[idx_v] = my_v.size
        self._my_feq = np.empty(shape)
        if (idx_r < idx_v):
            my_feq = [f_eq(r, v, constants.CN0, constants.kN0, constants.deltaRN0, constants.rp, 
                        constants.CTi, constants.kTi,constants.deltaRTi) 
                        for r in my_r for v in my_v]
        else:
            my_feq = [f_eq(r, v, constants.CN0, constants.kN0, constants.deltaRN0, constants.rp, 
                        constants.CTi, constants.kTi,constants.deltaRTi) 
                        for v in my_v for r in my_r]
        self._my_feq.flat = my_feq
        
        drMult = np.array(
            [dr[0] * 0.5, *((dr[1:] + dr[:-1]) * 0.5), dr[-1] * 0.5])
        dvMult = np.array(
            [dv[0] * 0.5, *((dv[1:] + dv[:-1]) * 0.5), dv[-1] * 0.5])

        mydrMult = drMult[layout.starts[idx_r]:layout.ends[idx_r]]
        mydvMult = dvMult[layout.starts[idx_v]:layout.ends[idx_v]]

        shape = [1, 1, 1, 1]
        shape[idx_r] = mydrMult.size
        shape[idx_v] = mydvMult.size

        self._factor1 = np.empty(shape)
        if (idx_r < idx_v):
            self._factor1.flat = (
                (mydrMult * my_r)[:, None] * (mydvMult * my_v**2)[None, :]).flat
        else:
            self._factor1.flat = (
                (mydrMult * my_r)[None, :] * (mydvMult * my_v**2)[:, None]).flat

        self._layout = layout.name

        dq = q[2] - q[1]
        assert dq * eta_grid[1].size - 2 * np.pi < 1e-7

        dz = z[2] - z[1]
        assert dq > 0
        assert dz > 0

        self._factor2 = 0.5 * dq * dz

    def getKE(self, grid):
        """
        TODO
        """
        assert self._layout == grid.currentLayout

        points = (grid._f - self._my_feq) * self._factor1

        return np.sum(points) * self._factor2

class PotentialEnergy:
    """
    TODO
    """

    def __init__(self, eta_grid: list, layout, constants):
        idx_r = layout.inv_dims_order[0]
        idx_q = layout.inv_dims_order[1]
        idx_z = layout.inv_dims_order[2]
        idx_v = layout.inv_dims_order[3]

        my_r = eta_grid[0][layout.starts[idx_r]:layout.ends[idx_r]]
        my_q = eta_grid[1][layout.starts[idx_q]:layout.ends[idx_q]]
        my_z = eta_grid[2][layout.starts[idx_z]:layout.ends[idx_z]]
        my_v = eta_grid[3][layout.starts[idx_v]:layout.ends[idx_v]]

        shape = [1, 1, 1, 1]
        shape[idx_r] = my_r.size
        shape[idx_q] = my_q.size
        shape[idx_z] = my_z.size
        self._shape_phi = shape

        r = eta_grid[0]
        q = eta_grid[1]
        z = eta_grid[2]
        v = eta_grid[3]

        dr = r[1:] - r[:-1]
        dv = v[1:] - v[:-1]

        shape = [1, 1, 1]
        shape[idx_r] = my_r.size
        self._myn0 = np.empty(shape)
        my_n0 = [n0(r, constants.CN0, constants.kN0, constants.deltaRN0, constants.rp) 
                    for r in my_r]
        self._myn0.flat = my_n0[:]

        drMult = np.array(
            [dr[0] * 0.5, *((dr[1:] + dr[:-1]) * 0.5), dr[-1] * 0.5])
        dvMult = np.array(
            [dv[0] * 0.5, *((dv[1:] + dv[:-1]) * 0.5), dv[-1] * 0.5])

        mydrMult = drMult[layout.starts[idx_r]:layout.ends[idx_r]]
        mydvMult = dvMult[layout.starts[idx_v]:layout.ends[idx_v]]

        shape = [1, 1, 1, 1]
        shape[idx_r] = mydrMult.size
        shape[idx_v] = mydvMult.size

        self._factor1 = np.empty(shape)
        if (idx_r < idx_v):
            self._factor1.flat = (
                (mydrMult * my_r)[:, None] * (mydvMult)[None, :]).flat
        else:
            self._factor1.flat = (
                (mydrMult * my_r)[None, :] * (mydvMult)[:, None]).flat
        
        shape = [1, 1, 1]
        shape[idx_r] = mydrMult.size 
        self._factor12 = np.empty(shape)
        self._factor12.flat = mydrMult * my_r

        self._layout = layout.name

        dq = q[2] - q[1]
        assert dq * eta_grid[1].size - 2 * np.pi < 1e-7

        dz = z[2] - z[1]
        assert dq > 0
        assert dz > 0

        self._factor2 = 0.5 * dq * dz

    def getPE(self, f, phi):
        """
        TODO
        """
        assert self._layout == f.currentLayout
        phi_grid = np.empty(self._shape_phi)
        phi_grid.flat = np.real(phi._f).flat

        points = np.real(f._f) * phi_grid * self._factor1
        points2 = -self._myn0 * np.real(phi._f) * self._factor12

        return (np.sum(points2)+np.sum(points)) * self._factor2

if __name__ == "__main__":
    foldername = "cobra_simulation_0/"
    doall(foldername)
