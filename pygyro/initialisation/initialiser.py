import numpy as np

from . import initialiser_func as IF_MOD

if ('initialiser_func' in dir(IF_MOD)):
    IF_MOD = IF_MOD.initialiser_func
    modFunc = np.transpose
else:
    modFunc = lambda c: c

def initialise_flux_surface(grid,constants):
    for i,r in grid.getCoords(0):
        for j,v in grid.getCoords(1):
            # Get surface
            FluxSurface = grid.get2DSlice([i,j])
            # Get coordinate values
            theta = grid.getCoordVals(2)
            z = grid.getCoordVals(3)

            IF_MOD.init_f_flux(modFunc(FluxSurface),r,theta,z,v,
                    constants.m,constants.n,constants.eps,
                    constants.CN0,constants.kN0,constants.deltaRN0,
                    constants.rp,constants.CTi,constants.kTi,
                    constants.deltaRTi,constants.deltaR,constants.R0)

def initialise_poloidal(grid,constants):
    for i,v in grid.getCoords(0):
        for j,z in grid.getCoords(1):
            # Get surface
            PoloidalSurface = grid.get2DSlice([i,j])
            # Get coordinate values
            theta = grid.getCoordVals(2)
            r = grid.getCoordVals(3)

            IF_MOD.init_f_pol(modFunc(PoloidalSurface),r,theta,z,v,
                    constants.m,constants.n,constants.eps,
                    constants.CN0,constants.kN0,constants.deltaRN0,
                    constants.rp,constants.CTi,constants.kTi,
                    constants.deltaRTi,constants.deltaR,constants.R0)

def initialise_v_parallel(grid,constants):
    for i,r in grid.getCoords(0):
        for j,z in grid.getCoords(1):
            # Get surface
            Surface = grid.get2DSlice([i,j])
            # Get coordinate values
            theta = grid.getCoordVals(2)
            v = grid.getCoordVals(3)

            IF_MOD.init_f_vpar(modFunc(Surface),r,theta,z,v,
                    constants.m,constants.n,constants.eps,
                    constants.CN0,constants.kN0,constants.deltaRN0,
                    constants.rp,constants.CTi,constants.kTi,
                    constants.deltaRTi,constants.deltaR,constants.R0)
