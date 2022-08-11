from mpi4py import MPI
import os
import pytest

from ..initialisation.constants import Constants
from .savingTools import setupSave


@pytest.mark.serial
def test_Save_s():
    """
    TODO
    """
    npts = [10, 20, 10, 10]

    constants = Constants()
    constants.npts = npts

    n1 = setupSave(constants)
    n2 = setupSave(constants)
    n3 = setupSave(constants, "testFilename")

    assert os.path.isdir(n1)
    assert os.path.isdir(n2)
    assert os.path.isdir(n3)

    os.remove("{0}/initParams.json".format(n1))
    os.remove("{0}/initParams.json".format(n2))
    os.remove("{0}/initParams.json".format(n3))

    os.rmdir(n1)
    os.rmdir(n2)
    os.rmdir(n3)


@pytest.mark.parallel
def test_Save_p():
    """
    TODO
    """
    npts = [10, 20, 10, 10]

    constants = Constants()
    constants.npts = npts

    n1 = setupSave(constants)
    n2 = setupSave(constants)
    n3 = setupSave(constants, "testFilename")

    if (MPI.COMM_WORLD.Get_rank() == 0):
        assert os.path.isdir(n1)
        assert os.path.isdir(n2)
        assert os.path.isdir(n3)

        os.remove("{0}/initParams.json".format(n1))
        os.remove("{0}/initParams.json".format(n2))
        os.remove("{0}/initParams.json".format(n3))

        os.rmdir(n1)
        os.rmdir(n2)
        os.rmdir(n3)
