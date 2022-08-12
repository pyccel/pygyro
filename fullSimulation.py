def main():
    import os
    import argparse
    import numpy as np
    from mpi4py import MPI
    import time

    from pygyro.diagnostics.diagnostic_collector import DiagnosticCollector
    from pygyro.utilities.savingTools import setupSave
    from pygyro.poisson.poisson_solver import DensityFinder, QuasiNeutralitySolver
    from pygyro.advection.advection import FluxSurfaceAdvection, VParallelAdvection, PoloidalAdvection, PoloidalAdvectionArakawa, ParallelGradient
    from pygyro.initialisation.setups import setupCylindricalGrid, setupFromFile
    from pygyro.model.grid import Grid
    from pygyro.model.layout import LayoutSwapper, getLayoutHandler

    setup_time_start = time.time()
    profilingOn = False

    if (profilingOn):
        import cProfile
        import pstats
        import io

    diagnostic_start = 0
    diagnostic_time = 0
    output_start = 0
    output_time = 0
    full_loop_start = 0
    full_loop_time = 0

    parser = argparse.ArgumentParser(description='Process foldername')
    parser.add_argument('tEnd', metavar='tEnd', nargs=1, type=int,
                        help='end time')
    parser.add_argument('tMax', metavar='tMax', nargs=1, type=int,
                        help='Maximum runtime in seconds')
    parser.add_argument('-c', dest='constantFile', nargs=1, type=str,
                        default=[""],
                        help='File describing the constants')
    parser.add_argument('-f', dest='foldername', nargs=1, type=str,
                        default=[""],
                        help='the name of the folder from which to load and in which to save')
    parser.add_argument('-s', dest='saveStep', nargs=1, type=int,
                        default=[5],
                        help='Number of time steps between writing output')
    parser.add_argument('--nosave', action='store_true')

    def my_print(rank, nosave, *args, **kwargs):
        if (rank == 0) and not nosave:
            print(time.time()-setup_time_start, *args, **kwargs,
                  file=open("out{}.txt".format(MPI.COMM_WORLD.Get_size()), "a"))

    args = parser.parse_args()
    foldername = args.foldername[0]
    constantFile = args.constantFile[0]
    nosave = args.nosave

    loadable = False

    saveStep = args.saveStep[0]
    saveStepCut = saveStep - 1

    tEnd = args.tEnd[0]

    stopTime = args.tMax[0]

    if (len(foldername) > 0):
        print("To load from ", foldername)
        if (os.path.isdir(foldername) and os.path.exists("{0}/initParams.json".format(foldername))):
            loadable = True
    else:
        foldername = None

    if (len(constantFile) == 0):
        constantFile = None

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if (loadable):
        my_print(rank, nosave, "ready to setup from loadable")

        distribFunc, constants, t = setupFromFile(foldername,
                                                  constantFile, comm=comm,
                                                  allocateSaveMemory=True,
                                                  layout='v_parallel')
    else:
        assert (constantFile is not None)

        my_print(rank, nosave, "ready to setup new")

        distribFunc, constants, t = setupCylindricalGrid(constantFile=constantFile,
                                                         layout='v_parallel',
                                                         comm=comm,
                                                         allocateSaveMemory=True)

        if nosave:
            if rank == 0:
                print('Not saving results')
        else:
            my_print(rank, nosave, "setup done, saving initParams")

            foldername = setupSave(constants, foldername)

            if rank == 0:
                print("Saving in", foldername)

    ti = t//constants.dt
    tN = int(tEnd//constants.dt)

    # --------------------------
    halfStep = constants.dt*0.5
    fullStep = constants.dt
    # --------------------------

    # Set method for poloidal advection step
    poloidal_method = constants.poloidalAdvectionMethod[0]

    print(f'using method {poloidal_method}')

    my_print(rank, nosave, "conditional setup done")

    fluxAdv = FluxSurfaceAdvection(distribFunc.eta_grid, distribFunc.get2DSpline(),
                                   distribFunc.getLayout('flux_surface'), halfStep, constants)
    my_print(rank, nosave, "flux adv init done")

    vParAdv = VParallelAdvection(
        distribFunc.eta_grid, distribFunc.getSpline(3), constants)
    my_print(rank, nosave, "v par adv init done")

    # Poloidal Advection step: Semi-Lagrangian method or Arakawa scheme
    if poloidal_method == 'sl':
        polAdv = PoloidalAdvection(
            distribFunc.eta_grid, distribFunc.getSpline(slice(1, None, -1)), constants)
        my_print(rank, nosave, "pol adv sl init done")

    elif poloidal_method == 'akw':
        polAdv = PoloidalAdvectionArakawa(distribFunc.eta_grid, constants)
        my_print(rank, nosave, "pol adv akw init done")

    else:
        raise NotImplementedError(
            f"{poloidal_method} is an unknown option for the poloidal advection step!")

    parGradVals = np.empty([distribFunc.getLayout(
        distribFunc.currentLayout).shape[0], constants.npts[2], constants.npts[1]])
    my_print(rank, nosave, "par grad vals done")

    layout_poisson = {'v_parallel_2d': [0, 2, 1],
                      'mode_solve': [1, 2, 0]}
    layout_vpar = {'v_parallel_1d': [0, 2, 1]}
    layout_poloidal = {'poloidal': [2, 1, 0]}

    nprocs = distribFunc.getLayout(distribFunc.currentLayout).nprocs[:2]
    my_print(rank, nosave, "layout params ready")

    remapperPhi = LayoutSwapper(comm, [layout_poisson, layout_vpar, layout_poloidal],
                                [nprocs, nprocs[0], nprocs[1]
                                 ], distribFunc.eta_grid[:3],
                                'mode_solve')
    my_print(rank, nosave, "remapper1 done")

    remapperRho = getLayoutHandler(
        comm, layout_poisson, nprocs, distribFunc.eta_grid[:3])
    my_print(rank, nosave, "remappers done")

    phi = Grid(distribFunc.eta_grid[:3], distribFunc.getSpline(slice(0, 3)),
               remapperPhi, 'mode_solve', comm, dtype=np.complex128)
    my_print(rank, nosave, "phi done")

    rho = Grid(distribFunc.eta_grid[:3], distribFunc.getSpline(slice(0, 3)),
               remapperRho, 'v_parallel_2d', comm, dtype=np.complex128)
    my_print(rank, nosave, "rho done")

    density = DensityFinder(6, distribFunc.getSpline(3),
                            distribFunc.eta_grid, constants)
    my_print(rank, nosave, "df ready")

    QNSolver = QuasiNeutralitySolver(distribFunc.eta_grid[:3], 7, distribFunc.getSpline(0),
                                     constants, chi=0)
    my_print(rank, nosave, "QN ready")

    parGrad = ParallelGradient(distribFunc.getSpline(1), distribFunc.eta_grid,
                               remapperPhi.getLayout('v_parallel_1d'), constants)
    my_print(rank, nosave, "par grad ready")

    diagnostics = DiagnosticCollector(
        comm, saveStep, fullStep, distribFunc, phi)
    my_print(rank, nosave, "diagnostics ready")

    diagnostic_filename = "{0}/phiDat.txt".format(foldername)

    if (profilingOn):
        # Setup profiling tools
        pr = cProfile.Profile()
        pr.enable()

    my_print(rank, nosave, "ready for setup")

    setup_time = time.time() - setup_time_start

    # ============================================
    # ==== Find phi from f^n by solving QN eq ====
    # ============================================
    distribFunc.setLayout('v_parallel')
    density.getPerturbedRho(distribFunc, rho)
    my_print(rank, nosave, "pert rho")

    QNSolver.getModes(rho)
    my_print(rank, nosave, "got modes")

    rho.setLayout('mode_solve')
    phi.setLayout('mode_solve')
    my_print(rank, nosave, "ready to solve")

    QNSolver.solveEquation(phi, rho)
    my_print(rank, nosave, "solved")

    phi.setLayout('v_parallel_2d')
    rho.setLayout('v_parallel_2d')
    my_print(rank, nosave, "ready to inv fourier")

    QNSolver.findPotential(phi)
    my_print(rank, nosave, "got phi")

    diagnostic_start = time.time()
    # Calculate diagnostic quantities
    diagnostics.collect(distribFunc, phi, t)

    diagnostic_time += (time.time() - diagnostic_start)

    if (not loadable) and (not nosave):
        my_print(rank, nosave, "save time", t)

        if rank == 0:
            print(rank, "save time", t)

        output_start = time.time()

        distribFunc.writeH5Dataset(foldername, t)

        my_print(rank, nosave, "grid printed")
        phi.writeH5Dataset(foldername, t, "phi")
        my_print(rank, nosave, "phi printed")

        diagnostics.reduce()
        if (rank == 0):
            diagnosticFile = open(diagnostic_filename, "a")
            print(diagnostics.getLine(0), file=diagnosticFile)
            diagnosticFile.close()
        output_time += (time.time() - output_start)

    nLoops = 0
    average_loop = 0
    average_output = 0
    startPrint = max(0, ti % saveStep)
    timeForLoop = True

    while (ti < tN and timeForLoop):
        full_loop_start = time.time()

        t += fullStep
        my_print(rank, nosave, "t=", t)

        # =============================================
        # ==== Compute f^n+1/2 using Lie splitting ====
        # =============================================
        distribFunc.setLayout('flux_surface')
        distribFunc.saveGridValues()
        fluxAdv.gridStep(distribFunc)

        distribFunc.setLayout('v_parallel')
        phi.setLayout('v_parallel_1d')
        vParAdv.gridStep(distribFunc, phi, parGrad, parGradVals, halfStep)

        distribFunc.setLayout('poloidal')
        phi.setLayout('poloidal')
        polAdv.gridStep(distribFunc, phi, halfStep)

        # ======================================================
        # ==== Find phi from f^n+1/2 by solving QN eq again ====
        # ======================================================
        distribFunc.setLayout('v_parallel')
        density.getPerturbedRho(distribFunc, rho)
        QNSolver.getModes(rho)

        rho.setLayout('mode_solve')
        phi.setLayout('mode_solve')
        QNSolver.solveEquation(phi, rho)

        phi.setLayout('v_parallel_2d')
        rho.setLayout('v_parallel_2d')
        QNSolver.findPotential(phi)

        # ==============================================
        # ==== Compute f^n+1 using strang splitting ====
        # ==============================================
        distribFunc.restoreGridValues()  # restored from flux_surface layout
        fluxAdv.gridStep(distribFunc)

        distribFunc.setLayout('v_parallel')
        phi.setLayout('v_parallel_1d')
        vParAdv.gridStep(distribFunc, phi, parGrad, parGradVals, halfStep)

        distribFunc.setLayout('poloidal')
        phi.setLayout('poloidal')
        polAdv.gridStep(distribFunc, phi, fullStep)

        distribFunc.setLayout('v_parallel')
        vParAdv.gridStepKeepGradient(distribFunc, parGradVals, halfStep)
        distribFunc.setLayout('flux_surface')
        fluxAdv.gridStep(distribFunc)

        # ============================================
        # ==== Find phi from f^n by solving QN eq ====
        # ============================================
        distribFunc.setLayout('v_parallel')
        density.getPerturbedRho(distribFunc, rho)
        QNSolver.getModes(rho)

        rho.setLayout('mode_solve')
        phi.setLayout('mode_solve')
        QNSolver.solveEquation(phi, rho)

        phi.setLayout('v_parallel_2d')
        rho.setLayout('v_parallel_2d')
        QNSolver.findPotential(phi)

        # =====================
        # ==== Diagnostics ====
        # =====================
        diagnostic_start = time.time()
        # Calculate diagnostic quantities
        diagnostics.collect(distribFunc, phi, t)

        diagnostic_time += (time.time() - diagnostic_start)

        if (ti % saveStep == saveStepCut) and (not nosave):
            my_print(rank, nosave, "save time", t)
            output_start = time.time()
            distribFunc.writeH5Dataset(foldername, t)
            phi.writeH5Dataset(foldername,t,"phi")

            diagnostics.reduce()
            if (rank == 0):
                diagnosticFile = open(diagnostic_filename, "a")
                for i in range(startPrint, min(saveStep, ti+1)):
                    print(diagnostics.getLine(i), file=diagnosticFile)
                diagnosticFile.close()
            startPrint = 0
            output_time += (time.time() - output_start)
            average_output = output_time*saveStep/nLoops

        nLoops += 1
        ti += 1
        full_loop_time += (time.time() - full_loop_start)
        average_loop = full_loop_time/nLoops
        timeForLoop = comm.allreduce((time.time(
        ) - setup_time_start + 2*average_loop + 2*average_output) < stopTime, op=MPI.LAND)

    full_loop_time += (time.time() - full_loop_start)

    output_start = time.time()

    if (ti % saveStep != 0) and (not nosave):

        diagnostics.reduce()
        if (rank == 0):
            diagnosticFile = open(diagnostic_filename, "a")
            for i in range(ti % saveStep):
                print(diagnostics.getLine(i), file=diagnosticFile)
            diagnosticFile.close()

        distribFunc.writeH5Dataset(foldername, t)
        phi.writeH5Dataset(foldername,t,"phi")
        output_time += (time.time() - output_start)

    # End profiling and print results
    if (profilingOn):
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('time')
        ps.print_stats()
        print(s.getvalue(), file=open("profile/l2Test{}.txt".format(rank), "w"))

    if (rank == 0):
        if (not os.path.isdir("timing")):
            os.mkdir("timing")

    MPI.COMM_WORLD.Barrier()

    print("{loop:16.10e}   {output:16.10e}   {setup:16.10e}   {diagnostic:16.10e}".
          format(loop=full_loop_time, output=output_time, setup=setup_time,
                 diagnostic=diagnostic_time),
          file=open("timing/{}_l2Test{}.txt".format(MPI.COMM_WORLD.Get_size(), rank), "w"))


if __name__ == "__main__":
    main()
