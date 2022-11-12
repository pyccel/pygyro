import numpy as np

from pygyro.model.grid import Grid
from pygyro.model.layout import Layout
# from ..arakawa.utilities import compute_int_f, compute_int_f_squared, get_potential_energy
from pygyro.initialisation.initialiser_funcs import f_eq, make_f_eq_grid


def make_trapz_grid(grid):
    """
    Generate a trapezoidal grid from a grid
    """
    d_grid = grid[1:] - grid[:-1]
    trapz_grid = np.append(d_grid[0] * 0.5,
                           np.append((d_grid + np.roll(d_grid, 1))[1:] * 0.5, d_grid[-1] * 0.5))
    return trapz_grid


class KineticEnergy:
    """
    TODO
    """

    def __init__(self, eta_grid: list, layout: Layout, constants):
        idx_r = layout.inv_dims_order[0]
        idx_v = layout.inv_dims_order[3]

        my_r = eta_grid[0][layout.starts[idx_r]:layout.ends[idx_r]]
        my_v = eta_grid[3][layout.starts[idx_v]:layout.ends[idx_v]]

        r = eta_grid[0]
        q = eta_grid[1]
        z = eta_grid[2]
        v = eta_grid[3]

        # dr = r[1:] - r[:-1]
        # dv = v[1:] - v[:-1]

        self._my_feq = np.empty(my_r.size * my_v.size)
        if (idx_r < idx_v):
            # my_feq = [f_eq(r, v, constants.CN0, constants.kN0, constants.deltaRN0, constants.rp,
            #                constants.CTi, constants.kTi, constants.deltaRTi)
            #           for r in my_r for v in my_v]
            make_f_eq_grid(constants.CN0, constants.kN0, constants.deltaRN0, constants.rp,
                           constants.CTi, constants.kTi, constants.deltaRTi, my_r, my_v, self._my_feq, 1)
        else:
            # my_feq = [f_eq(r, v, constants.CN0, constants.kN0, constants.deltaRN0, constants.rp,
            #                constants.CTi, constants.kTi, constants.deltaRTi)
            #           for v in my_v for r in my_r]
            make_f_eq_grid(constants.CN0, constants.kN0, constants.deltaRN0, constants.rp,
                           constants.CTi, constants.kTi, constants.deltaRTi, my_r, my_v, self._my_feq, 0)

        # assert np.allclose(self._my_feq, my_feq), 'These arrays are not the same'

        shape = [1, 1, 1, 1]
        shape[idx_r] = my_r.size
        shape[idx_v] = my_v.size
        self._my_feq = self._my_feq.reshape(shape)

        drMult = make_trapz_grid(r)
        dvMult = make_trapz_grid(v)

        mydrMult = drMult[layout.starts[idx_r]:layout.ends[idx_r]]
        mydvMult = dvMult[layout.starts[idx_v]:layout.ends[idx_v]]

        shape = [1, 1, 1, 1]
        shape[idx_r] = mydrMult.size
        shape[idx_v] = mydvMult.size

        self._factor1 = np.empty(shape)
        if (idx_r < idx_v):
            self._factor1.flat = ((mydrMult * my_r)[:, None]
                                  * (mydvMult * my_v**2)[None, :]).flat
        else:
            self._factor1.flat = ((mydrMult * my_r)[None, :]
                                  * (mydvMult * my_v**2)[:, None]).flat

        self._layout = layout.name

        dq = q[2] - q[1]
        assert dq * eta_grid[1].size - 2 * np.pi < 1e-7

        dz = z[2] - z[1]
        assert dq > 0
        assert dz > 0

        self._factor2 = 0.5 * dq * dz

    def getKE(self, grid: Grid):
        """
        TODO
        """
        assert self._layout == grid.currentLayout, \
            f'self._layout {self._layout} is not the same as grid.currentLayout {grid.currentLayout}'

        points = (grid._f - self._my_feq) * self._factor1

        return np.sum(points) * self._factor2


class PotentialEnergy:
    """
    TODO
    """

    def __init__(self, eta_grid: list, layout: Layout, constants):
        idx_r = layout.inv_dims_order[0]
        idx_q = layout.inv_dims_order[1]
        idx_z = layout.inv_dims_order[2]
        idx_v = layout.inv_dims_order[3]

        self.r_start_end = (layout.starts[idx_r], layout.ends[idx_r])
        self.q_start_end = (layout.starts[idx_q], layout.ends[idx_q])
        self.z_start_end = (layout.starts[idx_z], layout.ends[idx_z])
        self.v_start_end = (layout.starts[idx_v], layout.ends[idx_v])

        my_r = eta_grid[0][layout.starts[idx_r]:layout.ends[idx_r]]
        my_v = eta_grid[3][layout.starts[idx_v]:layout.ends[idx_v]]

        # Create temporary buffer for storing phi values
        self.phi_temp = np.zeros((self.r_start_end[1] - self.r_start_end[0],
                                  self.z_start_end[1] - self.z_start_end[0],
                                  1,
                                  self.v_start_end[1] - self.v_start_end[0]), dtype=float)

        r = eta_grid[0]
        q = eta_grid[1]
        z = eta_grid[2]
        v = eta_grid[3]

        # dr = r[1:] - r[:-1]
        # dv = v[1:] - v[:-1]

        self._my_feq = np.empty(my_r.size * my_v.size)
        if (idx_r < idx_v):
            # my_feq = [f_eq(r, v, constants.CN0, constants.kN0, constants.deltaRN0, constants.rp,
            #                constants.CTi, constants.kTi, constants.deltaRTi)
            #           for r in my_r for v in my_v]
            make_f_eq_grid(constants.CN0, constants.kN0, constants.deltaRN0, constants.rp,
                           constants.CTi, constants.kTi, constants.deltaRTi, my_r, my_v, self._my_feq, 1)
        else:
            # my_feq = [f_eq(r, v, constants.CN0, constants.kN0, constants.deltaRN0, constants.rp,
            #                constants.CTi, constants.kTi, constants.deltaRTi)
            #           for v in my_v for r in my_r]
            make_f_eq_grid(constants.CN0, constants.kN0, constants.deltaRN0, constants.rp,
                           constants.CTi, constants.kTi, constants.deltaRTi, my_r, my_v, self._my_feq, 0)

        # assert np.allclose(self._my_feq, my_feq), 'These arrays are not the same'

        shape = [1, 1, 1, 1]
        shape[idx_r] = my_r.size
        shape[idx_v] = my_v.size
        self._my_feq = self._my_feq.reshape(shape)

        drMult = make_trapz_grid(r)
        dvMult = make_trapz_grid(v)

        mydrMult = drMult[layout.starts[idx_r]:layout.ends[idx_r]]
        mydvMult = dvMult[layout.starts[idx_v]:layout.ends[idx_v]]

        shape = [1, 1, 1, 1]
        shape[idx_r] = mydrMult.size
        shape[idx_v] = mydvMult.size

        self._factor1 = np.empty(shape)
        if (idx_r < idx_v):
            self._factor1.flat = ((mydrMult * my_r)[:, None]
                                  * (mydvMult)[None, :]).flat
        else:
            self._factor1.flat = ((mydrMult * my_r)[None, :]
                                  * (mydvMult)[:, None]).flat

        self._layout = layout.name

        dq = q[2] - q[1]
        assert dq * eta_grid[1].size - 2 * np.pi < 1e-7

        dz = z[2] - z[1]
        assert dq > 0
        assert dz > 0

        self._factor2 = 0.5 * dq * dz

    def getPE(self, grid: Grid, phi: Grid):
        """
        TODO
        """
        assert self._layout == grid.currentLayout, \
            f'self._layout {self._layout} is not the same as grid.currentLayout {grid.currentLayout}'

        self.phi_temp.flat = np.real(
            phi._f[:, self.z_start_end[0]: self.z_start_end[1], :].flat)

        points = (grid._f - self._my_feq) * self.phi_temp * self._factor1

        return np.sum(points) * self._factor2
