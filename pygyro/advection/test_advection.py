import pytest

from ..model.grid               import Layout
from ..initialisation.setups    import setupCylindricalGrid
from .advection                 import *

@pytest.mark.serial
def test_fluxSurfaceAdvection():
    print("todo")

@pytest.mark.serial
def test_vParallelAdvection():
    print("todo")

@pytest.mark.serial
def test_poloidalAdvection():
    print("todo")

@pytest.mark.serial
def test_fluxSurfaceAdvection_gridIntegration():
    nr=10
    ntheta=20
    nz=10
    nv=10
    
    grid=setupCylindricalGrid(nr, ntheta, nz, nv, Layout.FIELD_ALIGNED)
    for j,r in grid.getEta1Coords():
        for k,v in grid.getEta4Coords():
            fluxSurfaceAdv(grid.getFieldAlignedSlice(j,k))

@pytest.mark.serial
def test_vParallelAdvection_gridIntegration():
    nr=10
    ntheta=20
    nz=10
    nv=10
    
    grid=setupCylindricalGrid(nr, ntheta, nz, nv, Layout.V_PARALLEL)
    for i,r in grid.getEta1Coords():
        for j,z in grid.getEta3Coords():
            for k,q in grid.getEta2Coords():
                vParallelAdv(grid.getEta4_Slice(i,j,k))

@pytest.mark.serial
def test_poloidalAdvection_gridIntegration():
    nr=10
    ntheta=20
    nz=10
    nv=10
    
    grid=setupCylindricalGrid(nr, ntheta, nz, nv, Layout.POLOIDAL)
    for j,z in grid.getEta3Coords():
        for k,v in grid.getEta4Coords():
            fluxSurfaceAdv(grid.getPoloidalSlice(j,k))
