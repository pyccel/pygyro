from pygyro.initialisation.setups import setupFromFile
import h5py
from mpi4py import MPI
import time
setup_time_start = time.clock()

# import cProfile, pstats, io


def get_grid_slice(foldername, tEnd):
    """
    TODO
    """

    assert(len(foldername) > 0)

    comm = MPI.COMM_WORLD

    distribFunc = setupFromFile(foldername, comm=comm,
                                allocateSaveMemory=True,
                                timepoint=tEnd)

    distribFunc.setLayout('poloidal')

    nv = distribFunc.eta_grid[3].size

    if (nv//2 in distribFunc.getGlobalIdxVals(0) and 0 in distribFunc.getGlobalIdxVals(1)):
        filename = "{0}/Slice_{1}.h5".format(foldername, tEnd)
        file = h5py.File(filename, 'w')
        dset = file.create_dataset(
            "dset", [distribFunc.eta_grid[1].size, distribFunc.eta_grid[0].size])
        i = nv//2 - distribFunc.getLayout(distribFunc.currentLayout).starts[0]
        dset[:] = distribFunc.get2DSlice([i, 0])
        file.close()
