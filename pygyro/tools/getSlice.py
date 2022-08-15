from pygyro.initialisation.setups import setupFromFile
import h5py
from mpi4py import MPI
import os


def get_grid_slice(foldername, tEnd, save_foldername=None, z_idx=None, v_idx=None):
    """
    Extract a poloidal slice of the distribution function at time tEnd from the specified folder
    and save the slice into a file named Slice_tEnd.h5 . The data is saved with theta
    in the first dimension, and r in the second dimension

    Parameters
    ----------
    foldername : str
                 The folder containing the simulation
    tEnd       : int
                 The time which should be examined to obtain the slice
    save_foldername : str
                 The folder where to save it to
    z_idx      : int
                 The index of the slice in the z direction
                 Default : 0
    v_idx      : int
                 The index of the slice in the v direction
                 Default : nv//2
    """

    assert (len(foldername) > 0)

    comm = MPI.COMM_WORLD

    distribFunc, _, _ = setupFromFile(foldername, comm=comm,
                                      allocateSaveMemory=True,
                                      timepoint=tEnd)

    distribFunc.setLayout('poloidal')

    if save_foldername is None:
        save_foldername = "Slices"

    if z_idx is None:
        z_idx = 0
    else:
        assert z_idx < distribFunc.eta_grid[2].size

    if v_idx is None:
        nv = distribFunc.eta_grid[3].size
        v_idx = nv//2
    else:
        assert v_idx < distribFunc.eta_grid[3].size

    if (v_idx in distribFunc.getGlobalIdxVals(0) and z_idx in distribFunc.getGlobalIdxVals(1)):
        dirname = "{0}/{1}/".format(foldername, save_foldername)
        if (not os.path.isdir(dirname)):
            os.mkdir(dirname)
        filename = dirname+"Slice_{:06}.h5".format(tEnd)

        file = h5py.File(filename, 'w')
        dset = file.create_dataset(
            "dset", [distribFunc.eta_grid[1].size, distribFunc.eta_grid[0].size])
        starts = distribFunc.getLayout(distribFunc.currentLayout).starts
        i = v_idx - starts[0]
        j = z_idx - starts[1]
        dset[:] = distribFunc.get2DSlice(i, j)
        file.close()


def get_flux_surface_grid_slice(foldername, tEnd, save_foldername=None, r_idx=None, v_idx=None):
    """
    Extract a flux surface slice of the distribution function at time tEnd from the
    specified folder and save the slice into a file named FluxSlice_tEnd.h5 . The data is
    saved with theta in the first dimension, and z in the second dimension

    Parameters
    ----------
    foldername : str
                 The folder containing the simulation
    tEnd       : int
                 The time which should be examined to obtain the slice
    save_foldername : str
                 The folder where to save it to
    r_idx      : int
                 The index of the slice in the r direction
                 Default : nr//2
    v_idx      : int
                 The index of the slice in the v direction
                 Default : nv//2
    """

    assert (len(foldername) > 0)

    comm = MPI.COMM_WORLD

    distribFunc, _, _ = setupFromFile(foldername, comm=comm,
                                      allocateSaveMemory=True,
                                      timepoint=tEnd)

    distribFunc.setLayout('flux_surface')

    if save_foldername is None:
        save_foldername = "FluxSlices"

    if r_idx is None:
        nr = distribFunc.eta_grid[0].size
        r_idx = nr//2
    else:
        assert r_idx < distribFunc.eta_grid[0].size

    if v_idx is None:
        nv = distribFunc.eta_grid[3].size
        v_idx = nv//2
    else:
        assert v_idx < distribFunc.eta_grid[3].size

    if (r_idx in distribFunc.getGlobalIdxVals(0) and v_idx in distribFunc.getGlobalIdxVals(1)):
        dirname = "{0}/{1}/".format(foldername, save_foldername)
        if (not os.path.isdir(dirname)):
            os.mkdir(dirname)
        filename = dirname+"FluxSlice_{:06}.h5".format(tEnd)

        file = h5py.File(filename, 'w')
        dset = file.create_dataset(
            "dset", [distribFunc.eta_grid[1].size, distribFunc.eta_grid[2].size])
        starts = distribFunc.getLayout(distribFunc.currentLayout).starts
        i = r_idx - starts[0]
        j = v_idx - starts[1]
        dset[:] = distribFunc.get2DSlice(i, j)
        file.close()
