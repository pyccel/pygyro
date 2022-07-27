from .initialiser_funcs import init_f_flux, init_f_pol, init_f_vpar


def initialise_flux_surface(grid, constants):
    """
    TODO
    """
    for i, r in grid.getCoords(0):
        for j, v in grid.getCoords(1):
            # Get surface
            FluxSurface = grid.get2DSlice([i, j])

            # Get coordinate values
            theta = grid.getCoordVals(2)
            z = grid.getCoordVals(3)

            init_f_flux(FluxSurface, r, theta, z, v,
                        constants.m, constants.n, constants.eps,
                        constants.CN0, constants.kN0, constants.deltaRN0,
                        constants.rp, constants.CTi, constants.kTi,
                        constants.deltaRTi, constants.deltaR, constants.R0)


def initialise_poloidal(grid, constants):
    """
    TODO
    """
    for i, v in grid.getCoords(0):
        for j, z in grid.getCoords(1):
            # Get surface
            PoloidalSurface = grid.get2DSlice([i, j])

            # Get coordinate values
            theta = grid.getCoordVals(2)
            r = grid.getCoordVals(3)

            init_f_pol(PoloidalSurface, r, theta, z, v,
                       constants.m, constants.n, constants.eps,
                       constants.CN0, constants.kN0, constants.deltaRN0,
                       constants.rp, constants.CTi, constants.kTi,
                       constants.deltaRTi, constants.deltaR, constants.R0)


def initialise_v_parallel(grid, constants):
    """
    TODO
    """
    for i, r in grid.getCoords(0):
        for j, z in grid.getCoords(1):
            # Get surface
            Surface = grid.get2DSlice([i, j])

            # Get coordinate values
            theta = grid.getCoordVals(2)
            v = grid.getCoordVals(3)

            init_f_vpar(Surface, r, theta, z, v,
                        constants.m, constants.n, constants.eps,
                        constants.CN0, constants.kN0, constants.deltaRN0,
                        constants.rp, constants.CTi, constants.kTi,
                        constants.deltaRTi, constants.deltaR, constants.R0)
