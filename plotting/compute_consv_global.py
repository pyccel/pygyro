import json
import os
import numpy as np
import argparse
from mpi4py import MPI

from pygyro.initialisation.setups import setupFromFile
from pygyro.model.process_grid import compute_2d_process_grid
from pygyro.model.grid import Grid
from pygyro.model.layout import LayoutSwapper
from pygyro.diagnostics.diagnostic_collector import AdvectionDiagnostics
import pygyro.splines.splines as spl


def get_time(k):
    """
    TODO
    """
    t = str(k)

    while len(t) < 6:
        t = "0" + t

    return t


def main():
    """
    TODO
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('foldername', metavar='foldername',
                        nargs='*', type=str)

    args = parser.parse_args()
    foldername = args.foldername[0]

    assert os.path.exists(foldername), "The indicated folder does not exist!"

    # Get save time-step
    k = 1
    while True:
        if not os.path.exists(os.path.join(foldername, "grid_" + get_time(k) + ".h5")):
            k += 1
        else:
            break

    s = k

    constants_filename = os.path.join(foldername, "initParams.json")

    with open(constants_filename) as file:
        dt = json.load(file)["dt"]
    with open(constants_filename) as file:
        method = json.load(file)["poloidalAdvectionMethod"][0]

    quantities = ["mass_f", "l2_f", "en_tot"]

    rank = MPI.COMM_WORLD.Get_rank()

    savefile = os.path.join(foldername, method + "_all_consv.txt")

    if rank == 0:
        with open(savefile, 'w') as file:
            for quantity in quantities:
                file.write(quantity)
                file.write("\t\t\t\t\t")

            file.write("\n")

    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()

    distribFunc, constants, _ = setupFromFile(foldername,
                                              timepoint=0, comm=comm)

    npts = constants.npts

    degree = constants.splineDegrees[:-1]
    period = [False, True, True]
    domain = [[constants.rMin, constants.rMax],
              [0, 2*np.pi],
              [constants.zMin, constants.zMax]]

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
    phi.loadFromFile(foldername, 0, "phi")

    collector = AdvectionDiagnostics(comm, dt, distribFunc, constants)

    k = 0
    while True:
        t = k * s
        if not os.path.exists(os.path.join(foldername, "grid_" + get_time(t) + ".h5")):
            break

        distribFunc.setLayout('v_parallel')
        phi.setLayout('v_parallel_2d')

        distribFunc.loadFromFile(foldername, t, "grid")
        phi.loadFromFile(foldername, t, "phi")

        distribFunc.setLayout('poloidal')
        phi.setLayout('poloidal')

        collector.collect(distribFunc, phi)
        collector.reduce()

        mass_f = collector.diagnostics_val[0][0]
        l2_f = collector.diagnostics_val[1][0]
        en_kin = collector.diagnostics_val[3][0]
        en_pot = collector.diagnostics_val[4][0]
        en_tot = en_kin + en_pot

        if (rank == 0):
            with open(savefile, 'a') as file:
                file.write(format(mass_f, '.15E') + "\t")
                file.write(format(l2_f, '.15E') + "\t")
                file.write(format(en_tot, '.15E') + "\t")
                file.write("\n")
        k += 1


if __name__ == '__main__':
    main()
