from mpi4py import MPI
import numpy as np
from functools import reduce
import pytest

from  .setups                 import setupCylindricalGrid

@pytest.mark.serial
def test_FluxSurface_setup():
    nr=10
    ntheta=20
    nz=10
    nv=10
    setupCylindricalGrid(nr, ntheta, nz, nv, 'flux_surface')

@pytest.mark.serial
def test_Poloidal_setup():
    nr=10
    ntheta=20
    nz=10
    nv=10
    setupCylindricalGrid(nr, ntheta, nz, nv, 'poloidal')

@pytest.mark.serial
def test_vParallel_setup():
    nr=10
    ntheta=20
    nz=10
    nv=10
    setupCylindricalGrid(nr, ntheta, nz, nv, 'v_parallel')

@pytest.mark.parallel
@pytest.mark.parametrize("splitN", [1,2,3,4,5,6])
def test_FluxSurface_setup_parallel(splitN):
    nr=10
    ntheta=20
    nz=10
    nv=10
    size=MPI.COMM_WORLD.Get_size()
    n1=max(size//splitN,1)
    n2=size//n1
    if (n1*n2!=size):
        return
    
    setupCylindricalGrid(nr, ntheta, nz, nv, 'flux_surface',nProcEta1=n1,nProcEta4=n2)

@pytest.mark.parallel
@pytest.mark.parametrize("splitN", [1,2,3,4,5,6])
def test_Poloidal_setup_parallel(splitN):
    nr=10
    ntheta=20
    nz=10
    nv=10
    size=MPI.COMM_WORLD.Get_size()
    n1=max(size//splitN,1)
    n2=size//n1
    if (n1*n2!=size):
        return
    
    setupCylindricalGrid(nr, ntheta, nz, nv, 'poloidal',nProcEta3=n1,nProcEta4=n2)

@pytest.mark.parallel
@pytest.mark.parametrize("splitN", [1,2,3,4,5,6])
def test_vParallel_setup_parallel(splitN):
    nr=10
    ntheta=20
    nz=10
    nv=10
    size=MPI.COMM_WORLD.Get_size()
    n1=max(size//splitN,1)
    n2=size//n1
    if (n1*n2!=size):
        return
    
    setupCylindricalGrid(nr, ntheta, nz, nv, 'v_parallel',nProcEta1=n1,nProcEta3=n2)
