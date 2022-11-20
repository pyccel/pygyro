import os
import numpy as np
from mpi4py import MPI
import argparse

from pygyro import splines as spl
from pygyro.initialisation.setups import setupFromFile
from pygyro.model.layout import LayoutSwapper
from pygyro.model.process_grid import compute_2d_process_grid
from pygyro.model.grid import Grid
from plotting.energy import KineticEnergy_v2, PotentialEnergy_v2, PotentialEnergy_fresch, Mass_f, L2_f, L2_phi
from pygyro.diagnostics.norms import l2


def calc_consv(foldername, diagnostics, ind, comm, classes):
    """
    TODO
    """
    size = comm.Get_size()
    time = int(diagnostics[0, ind])
    distribFunc, constants, _ = setupFromFile(foldername,
                                              timepoint=time,
                                              comm=comm)

    npts = constants.npts

    degree = constants.splineDegrees[:-1]
    period = [False, True, True]
    domain = [[constants.rMin, constants.rMax],
              [0, 2*np.pi],
              [constants.zMin, constants.zMax]]

    nkts = [n + 1 + d * (int(p) - 1)
            for (n, d, p) in zip(npts, degree, period)]
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

    nprocs = compute_2d_process_grid(npts, size)

    nprocs = distribFunc.getLayout(distribFunc.currentLayout).nprocs[:2]

    remapperPhi = LayoutSwapper(comm, [layout_poisson, layout_vpar, layout_poloidal],
                                [nprocs, nprocs[0], nprocs[1]], eta_grids,
                                'v_parallel_2d')

    phi = Grid(eta_grids, bsplines, remapperPhi,
               'v_parallel_2d', comm, dtype=np.complex128)

    phi.loadFromFile(foldername, time, "phi")

    distribFunc.setLayout('poloidal')
    phi.setLayout('poloidal')

    for k, myclass in enumerate(classes):
        if k == 0:
            diagnostics[k + 1, ind] = myclass.getMASSF(distribFunc)
        elif k == 1:
            diagnostics[k + 1, ind] = myclass.getL2F(distribFunc)
        elif k == 2:
            diagnostics[k + 1, ind] = myclass.getL2Phi(phi)
            # diagnostics[k + 1, ind] = myclass.l2NormSquared(phi)
        elif k == 3:
            diagnostics[k + 1, ind] = myclass.getKE(distribFunc)
        elif k == 4:
            if isinstance(myclass, PotentialEnergy_fresch):
                distribFunc.setLayout('v_parallel')
                phi.setLayout('v_parallel_2d')
            diagnostics[k + 1, ind] = myclass.getPE(distribFunc, phi)
        else:
            raise ValueError


def do_all(foldername):
    """
    quantities saved are:
        'time', 'mass_f', 'l2_f', 'l2_phi', 'en_kin', 'en_pot'
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    distribFunc, constants, _ = setupFromFile(foldername,
                                              timepoint=0,
                                              comm=comm)

    # which quantities should be extracted
    quantities = ['mass_f', 'l2_f', 'l2_phi', 'en_kin', 'en_pot']
    method = constants.poloidalAdvectionMethod[0]

    # create classes for computing quantities
    MASSFclass = Mass_f(
        distribFunc.eta_grid, distribFunc.getLayout('poloidal'))
    L2Fclass = L2_f(
        distribFunc.eta_grid, distribFunc.getLayout('poloidal'))
    # L2PHIclass = l2(
    #     distribFunc.eta_grid, distribFunc.getLayout('poloidal'))
    L2PHIclass = L2_phi(
        distribFunc.eta_grid, distribFunc.getLayout('poloidal'))
    KEclass = KineticEnergy_v2(
        distribFunc.eta_grid, distribFunc.getLayout('poloidal'), constants)
    # PEclass = PotentialEnergy_v2(
    #     distribFunc.eta_grid, distribFunc.getLayout('poloidal'), constants)
    PEclass = PotentialEnergy_fresch(
        distribFunc.eta_grid, distribFunc.getLayout('v_parallel'), constants)

    classes = [MASSFclass, L2Fclass, L2PHIclass, KEclass, PEclass]

    t = []
    filelist = os.listdir(foldername)

    for fname in filelist:
        if fname.find("grid_") != -1:
            t_str = fname[fname.rfind('_') + 1: fname.rfind('.')]
            t.append(int(t_str))

    t = np.array(t)

    advection_savefile = foldername + method + '_consv.txt'

    if rank == 0:
        # Create savefile or overwrite old one
        with open(advection_savefile, 'w') as savefile:
            savefile.write('time')
            savefile.write("\t")
            for quantity in quantities:
                savefile.write(quantity)
                savefile.write("\t\t\t\t\t")

            savefile.write("\n")

    diagnostics = np.zeros((len(quantities) + 1, len(t)))
    diagnostics_val = np.zeros((len(quantities) + 1, len(t)))
    diagnostics[0, :] = t

    for i in range(len(t)):
        calc_consv(foldername, diagnostics, i, comm, classes)

    for k in range(1, len(diagnostics)):
        comm.Reduce(diagnostics[k, :],
                    diagnostics_val[k, :], op=MPI.SUM, root=0)

    index_l2_f = quantities.index('l2_f') + 1
    index_l2_phi = quantities.index('l2_phi') + 1

    diagnostics_val[index_l2_f, :] = np.sqrt(diagnostics_val[index_l2_f, :])
    diagnostics_val[index_l2_phi, :] = np.sqrt(diagnostics_val[index_l2_phi, :])

    if rank == 0:
        with open(advection_savefile, 'a') as savefile:
            for i, time in enumerate(t):
                savefile.write(format(time) + '\t')
                if len(str(time)) < 4:
                    savefile.write('\t')
                for k in range(1, len(diagnostics)):
                    savefile.write(
                        format(diagnostics_val[k, i], '.15E') + "\t")
                savefile.write('\n')


def main():
    """
    TODO
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', metavar='foldername',
                        nargs='*', type=int)

    args = parser.parse_args()
    k = args.k[0]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    while True:
        foldername = 'simulation_' + str(k) + '/'
        if os.path.exists(foldername):
            if rank == 0:
                print(f'Now computing conservation for {foldername}')
            do_all(foldername)
            # k += 1
            break
        else:
            break


if __name__ == '__main__':
    main()
