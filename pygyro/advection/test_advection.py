import pytest

from ..model.grid               import Layout
from ..initialisation.setups    import setupGrid

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
    
    setupGrid(nr, ntheta, nz, nv, Layout.FIELD_ALIGNED)
    for j,r in grid.getRCoords():
        for k,v in grid.getVCoords():
            fluxSurfaceAdv(grid.getFieldAlignedSlice(j,k))

@pytest.mark.serial
def test_vParallelAdvection_gridIntegration():
    nr=10
    ntheta=20
    nz=10
    nv=10
    
    setupGrid(nr, ntheta, nz, nv, Layout.V_PARALLEL)
    for i,q in grid.getThetaCoords():
        for j,r in grid.getRCoords():
            for k,z in grid.getZCoords():
                vParallelAdv(grid.getVParSlice(i,j,k))

@pytest.mark.serial
def test_poloidalAdvection_gridIntegration():
    nr=10
    ntheta=20
    nz=10
    nv=10
    
    setupGrid(nr, ntheta, nz, nv, Layout.POLOIDAL)
    for j,z in grid.getZCoords():
        for k,v in grid.getVCoords():
            fluxSurfaceAdv(grid.getPoloidalSlice(j,k))
