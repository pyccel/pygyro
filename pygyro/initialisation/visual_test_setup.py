import pytest

from  .setups                   import setupCylindricalGrid
from  .initialiser_funcs        import f_eq, perturbation
from ..utilities.grid_plotter   import SlicePlotterNd

@pytest.mark.serial
def test_Perturbation_FluxSurface():
    npts = [10,10,10,20]
    m = 15
    n = 20
    grid,constants,tStart = setupCylindricalGrid(npts   = npts,
                                layout = 'flux_surface',
                                m      = m,
                                n      = n)
    for i,r in grid.getCoords(0):
        for j,_ in grid.getCoords(1): # v
            # Get surface
            FluxSurface = grid.get2DSlice([i,j])
            # Get coordinate values
            theta = grid.getCoordVals(2)
            z = grid.getCoordVals(3)

            # transpose theta to use ufuncs
            theta = theta.reshape(theta.size,1)
            FluxSurface[:]=perturbation(r,theta,z,m,n,constants.rp,constants.deltaR,constants.R0)

    p = SlicePlotterNd(grid,0,3,polar=False)
    p.setLabels('r','v')
    p.show()

@pytest.mark.serial
def test_Perturbation_vPar():
    npts = [10,20,10,10]
    m = 15
    n = 20
    grid,constants,tStart = setupCylindricalGrid(npts   = npts,
                                layout = 'v_parallel',
                                m      = m,
                                n      = n)
    for i,r in grid.getCoords(0):
        for j,z in grid.getCoords(1):
            # Get surface
            Surface = grid.get2DSlice([i,j])
            # Get coordinate values
            theta = grid.getCoordVals(2)
            v = grid.getCoordVals(3)

            # transpose theta to use ufuncs
            theta = theta.reshape(theta.size,1)
            Surface[:]=perturbation(r,theta,z,m,n,constants.rp,constants.deltaR,constants.R0)

    p = SlicePlotterNd(grid,0,3,False)
    p.setLabels('r','v')
    p.show()

@pytest.mark.serial
def test_Perturbation_Poloidal():
    npts = [10,20,10,10]
    m = 15
    n = 20
    grid,constants,tStart = setupCylindricalGrid(npts   = npts,
                                layout = 'poloidal',
                                m      = m,
                                n      = n)
    for i,_ in grid.getCoords(0): # v
        for j,z in grid.getCoords(1):
            # Get surface
            PoloidalSurface = grid.get2DSlice([i,j])
            # Get coordinate values
            theta = grid.getCoordVals(2)
            r = grid.getCoordVals(3)

            # transpose theta to use ufuncs
            theta = theta.reshape(theta.size,1)
            PoloidalSurface[:]=perturbation(r,theta,z,m,n,constants.rp,constants.deltaR,constants.R0)

    p = SlicePlotterNd(grid,0,3,False)
    p.setLabels('r','v')
    p.show()

@pytest.mark.serial
def test_FieldPlot_FluxSurface():
    npts = [10,20,10,10]
    m = 15
    n = -11
    iotaVal = 0.8

    grid,constants,tStart = setupCylindricalGrid(npts   = npts,
                                layout  = 'flux_surface',
                                m       = m,
                                n       = n,
                                iotaVal = iotaVal)
    for i,r in grid.getCoords(0):
        for j,_ in grid.getCoords(1): # v
            # Get surface
            FluxSurface = grid.get2DSlice([i,j])
            # Get coordinate values
            theta = grid.getCoordVals(2)
            z = grid.getCoordVals(3)

            # transpose theta to use ufuncs
            theta = theta.reshape(theta.size,1)
            FluxSurface[:]=perturbation(r,theta,z,m,n,constants.rp,constants.deltaR,constants.R0)

    p = SlicePlotterNd(grid,1,2,False)
    p.setLabels('q','z')
    p.show()

@pytest.mark.serial
def test_FieldPlot_vPar():
    npts = [10,20,10,10]
    m = 15
    n = -11
    iotaVal = 0.8
    grid,constants,tStart = setupCylindricalGrid(npts   = npts,
                                layout = 'v_parallel',
                                m      = m,
                                n      = n,
                                iotaVal = iotaVal)
    for i,r in grid.getCoords(0):
        for j,z in grid.getCoords(1):
            # Get surface
            Surface = grid.get2DSlice([i,j])
            # Get coordinate values
            theta = grid.getCoordVals(2)
            v = grid.getCoordVals(3)

            # transpose theta to use ufuncs
            theta = theta.reshape(theta.size,1)
            Surface[:]=perturbation(r,theta,z,m,n,constants.rp,constants.deltaR,constants.R0)

    p = SlicePlotterNd(grid,1,2,False)
    p.setLabels('q','z')
    p.show()

@pytest.mark.serial
def test_FieldPlot_Poloidal():
    npts = [10,20,10,10]
    m = 15
    n = -11
    iotaVal = 0.8
    grid,constants,tStart = setupCylindricalGrid(npts   = npts,
                                layout = 'poloidal',
                                m      = m,
                                n      = n,
                                iotaVal = iotaVal)
    for i,_ in grid.getCoords(0): # v
        for j,z in grid.getCoords(1):
            # Get surface
            PoloidalSurface = grid.get2DSlice([i,j])
            # Get coordinate values
            theta = grid.getCoordVals(2)
            r = grid.getCoordVals(3)

            # transpose theta to use ufuncs
            theta = theta.reshape(theta.size,1)
            PoloidalSurface[:]=perturbation(r,theta,z,m,n,constants.rp,constants.deltaR,constants.R0)

    p = SlicePlotterNd(grid,1,2,False)
    p.setLabels('q','z')
    p.show()

@pytest.mark.serial
def test_Equilibrium_FluxSurface():
    npts = [10,20,10,10]
    m = 15
    n = 20
    grid,constants,tStart = setupCylindricalGrid(npts   = npts,
                                layout = 'flux_surface',
                                m      = m,
                                n      = n)
    for i,r in grid.getCoords(0):
        for j,v in grid.getCoords(1):
            # Get surface
            FluxSurface = grid.get2DSlice([i,j])
            # Get coordinate values
            theta = grid.getCoordVals(2)
            z = grid.getCoordVals(3)

            # transpose theta to use ufuncs
            theta = theta.reshape(theta.size,1)
            FluxSurface[:]=f_eq(r,v,constants.CN0,constants.kN0,
                                constants.deltaRN0,constants.rp,
                                constants.CTi,constants.kTi,
                                constants.deltaRTi)

    p = SlicePlotterNd(grid,0,3,False)
    p.setLabels('r','v')
    p.show()

@pytest.mark.serial
def test_Equilibrium_vPar():
    npts = [10,20,10,10]
    m = 15
    n = 20
    grid,constants,tStart = setupCylindricalGrid(npts   = npts,
                                layout = 'v_parallel',
                                m      = m,
                                n      = n)
    for i,r in grid.getCoords(0):
        for j,_ in grid.getCoords(1): # z
            # Get surface
            Surface = grid.get2DSlice([i,j])
            # Get coordinate values
            theta = grid.getCoordVals(2)
            v = grid.getCoordVals(3)

            # transpose theta to use ufuncs
            theta = theta.reshape(theta.size,1)
            Surface[:]=f_eq(r,v,constants.CN0,constants.kN0,
                            constants.deltaRN0,constants.rp,
                            constants.CTi,constants.kTi,
                            constants.deltaRTi)

    p = SlicePlotterNd(grid,0,3,False)
    p.setLabels('r','v')
    p.show()

@pytest.mark.serial
def test_Equilibrium_Poloidal():
    npts = [10,20,10,10]
    m = 15
    n = 20
    grid,constants,tStart = setupCylindricalGrid(npts   = npts,
                                layout = 'poloidal',
                                m      = m,
                                n      = n)
    for i,v in grid.getCoords(0):
        for j,_ in grid.getCoords(1): # z
            # Get surface
            PoloidalSurface = grid.get2DSlice([i,j])
            # Get coordinate values
            theta = grid.getCoordVals(2)
            r = grid.getCoordVals(3)

            # transpose theta to use ufuncs
            theta = theta.reshape(theta.size,1)
            PoloidalSurface[:]=f_eq(r,v,constants.CN0,constants.kN0,
                                    constants.deltaRN0,constants.rp,
                                    constants.CTi,constants.kTi,
                                    constants.deltaRTi)

    p = SlicePlotterNd(grid,0,3,False)
    p.setLabels('r','v')
    p.show()

@pytest.mark.serial
def test_FluxSurface():
    npts = [10,10,20,20]
    grid,constants,tStart = setupCylindricalGrid(npts   = npts,
                                layout = 'flux_surface')
    p = SlicePlotterNd(grid,1,2,False, sliderDimensions=[0,3], sliderNames=['r','v'])
    p.setLabels('q','z')
    p.show()

@pytest.mark.serial
def test_vParallel():
    npts = [10,10,20,20]
    grid,constants,tStart = setupCylindricalGrid(npts   = npts,
                                layout = 'v_parallel')
    p = SlicePlotterNd(grid,0,3,False, sliderDimensions=[1,2], sliderNames=['q','z'])
    p.setLabels('r','v')
    p.show()

@pytest.mark.serial
def test_Poloidal():
    npts = [10,10,20,20]
    grid,constants,tStart = setupCylindricalGrid(npts   = npts,
                                layout = 'poloidal')
    p = SlicePlotterNd(grid,0,1,True, sliderDimensions=[2,3], sliderNames=['z','v'])
    p.setLabels('x','y')
    p.show()
