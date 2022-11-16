import numpy as np

from pygyro.model.grid import Grid
from pygyro.model.layout import Layout
# from ..arakawa.utilities import compute_int_f, compute_int_f_squared, get_potential_energy
from pygyro.initialisation.initialiser_funcs import make_f_eq_grid, make_n0_grid


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

        if (idx_r < idx_v):
            self._my_feq = np.zeros((my_r.size, my_v.size), dtype=float)
            make_f_eq_grid(constants.CN0, constants.kN0, constants.deltaRN0, constants.rp,
                           constants.CTi, constants.kTi, constants.deltaRTi, my_r, my_v, self._my_feq, 0)
        elif (idx_v < idx_r):
            self._my_feq = np.zeros((my_v.size, my_r.size), dtype=float)
            make_f_eq_grid(constants.CN0, constants.kN0, constants.deltaRN0, constants.rp,
                           constants.CTi, constants.kTi, constants.deltaRTi, my_r, my_v, self._my_feq, 1)

        shape = [1, 1, 1, 1]
        shape[idx_r] = my_r.size
        shape[idx_v] = my_v.size
        # Use resize for in-place reshaping
        self._my_feq.resize(shape)

        drMult = make_trapz_grid(r)
        self.mydrMult = drMult[layout.starts[idx_r]:layout.ends[idx_r]] * my_r

        shape = [1, 1, 1, 1]
        shape[idx_r] = self.mydrMult.size
        self.mydrMult.resize(shape)

        dvMult = make_trapz_grid(v)
        self.mydvMult = dvMult[layout.starts[idx_v]:layout.ends[idx_v]] \
            * (my_v ** 2)

        shape = [1, 1, 1, 1]
        shape[idx_v] = self.mydvMult.size
        self.mydvMult.resize(shape)

        self._layout = layout.name

        dq = q[2] - q[1]
        assert dq * eta_grid[1].size - 2 * np.pi < 1e-7

        dz = z[2] - z[1]
        assert dq > 0
        assert dz > 0

        self._factor_int_theta_z = 0.5 * dq * dz

    def getKE(self, grid: Grid):
        """
        TODO
        """
        assert self._layout == grid.currentLayout, \
            f'self._layout {self._layout} is not the same as grid.currentLayout {grid.currentLayout}'

        points = ((grid._f - self._my_feq) * self.mydrMult) * self.mydvMult

        return np.sum(points) * self._factor_int_theta_z


class PotentialEnergy:
    """
    TODO
    """

    def __init__(self, eta_grid: list, layout, constants):
        idx_r = layout.inv_dims_order[0]
        idx_q = layout.inv_dims_order[1]
        idx_z = layout.inv_dims_order[2]
        self.idx_v = layout.inv_dims_order[3]

        my_r = eta_grid[0][layout.starts[idx_r]:layout.ends[idx_r]]
        my_q = eta_grid[1][layout.starts[idx_q]:layout.ends[idx_q]]
        my_z = eta_grid[2][layout.starts[idx_z]:layout.ends[idx_z]]
        my_v = eta_grid[3][layout.starts[self.idx_v]:layout.ends[self.idx_v]]

        shape = [1, 1, 1]
        shape[idx_r] = my_r.size
        shape[idx_q] = my_q.size
        shape[idx_z] = my_z.size
        self._shape_phi = shape

        r = eta_grid[0]
        q = eta_grid[1]
        z = eta_grid[2]
        v = eta_grid[3]

        shape = [1, 1, 1]
        shape[idx_r] = my_r.size
        self.n_0 = np.zeros(my_r.size, dtype=float)

        make_n0_grid(constants.CN0, constants.kN0,
                     constants.deltaRN0, constants.rp, my_r, self.n_0)
        self.n_0.resize(shape)

        drMult = make_trapz_grid(r)
        self.mydrMult = drMult[layout.starts[idx_r]:layout.ends[idx_r]] * my_r

        shape = [1, 1, 1, 1]
        shape[idx_r] = self.mydrMult.size
        self.mydrMult.resize(shape)

        dvMult = make_trapz_grid(v)
        self.mydvMult = dvMult[layout.starts[self.idx_v]:layout.ends[self.idx_v]] \
            * (my_v ** 2)

        shape = [1, 1, 1, 1]
        shape[self.idx_v] = self.mydvMult.size
        self.mydvMult.resize(shape)

        self._layout = layout.name

        dq = q[2] - q[1]
        assert dq * eta_grid[1].size - 2 * np.pi < 1e-7

        dz = z[2] - z[1]
        assert dq > 0
        assert dz > 0

        self._factor_int_theta_z = 0.5 * dq * dz

    def getPE(self, f: Grid, phi: Grid):
        """
        TODO
        """
        assert self._layout == f.currentLayout

        phi_grid = np.empty(self._shape_phi)
        phi_grid.flat = np.real(phi._f).flat

        n_i = np.sum(f._f * self.mydvMult, axis=self.idx_v)

        # print(f'shape of f : {np.shape(f._f)}')
        # print(f'shape of n0 : {np.shape(self.n_0)}')
        # print(f'shape of phi : {np.shape(phi_grid)}')
        # print(f'shape of dvMult : {np.shape(self.mydvMult)}')
        # print(f'shape of f * dvMult : {np.shape(f._f * self.mydvMult)}')
        # print(f'shape of n_i : {np.shape(n_i)}')

        # """
        # shape of f : (10, 8, 20, 40)
        # shape of n0 : (10, 1, 1)
        # shape of phi : (10, 8, 20)
        # shape of dvMult : (1, 1, 1, 40)
        # shape of f * dvMult : (10, 8, 20, 40)
        # shape of n_i : (10, 8, 20)
        # """

        points = (((n_i - self.n_0) * phi_grid))

        return np.sum(points * self.mydrMult) * self._factor_int_theta_z


class L2phi:
    """
    TODO
    """

    def __init__(self, eta_grid: list, layout):
        idx_r = layout.inv_dims_order[0]
        idx_q = layout.inv_dims_order[1]
        idx_z = layout.inv_dims_order[2]

        my_r = eta_grid[0][layout.starts[idx_r]:layout.ends[idx_r]]
        my_q = eta_grid[1][layout.starts[idx_q]:layout.ends[idx_q]]
        my_z = eta_grid[2][layout.starts[idx_z]:layout.ends[idx_z]]

        shape = [1, 1, 1]
        shape[idx_r] = my_r.size
        shape[idx_q] = my_q.size
        shape[idx_z] = my_z.size
        self._shape_phi = shape

        r = eta_grid[0]
        q = eta_grid[1]
        z = eta_grid[2]

        drMult = make_trapz_grid(r)
        self.mydrMult = drMult[layout.starts[idx_r]:layout.ends[idx_r]] * my_r

        shape = [1, 1, 1]
        shape[idx_r] = self.mydrMult.size
        self.mydrMult.resize(shape)

        dq = q[2] - q[1]
        assert dq * eta_grid[1].size - 2 * np.pi < 1e-7

        dz = z[2] - z[1]
        assert dq > 0
        assert dz > 0

        self._factor_int_theta_z = 0.5 * dq * dz

    def getl2(self, phi: Grid):
        """
        TODO
        """
        phi_grid = np.empty(self._shape_phi)
        phi_grid.flat = np.real(phi._f).flat

        return np.sum(phi_grid**2 * self.mydrMult) * self._factor_int_theta_z
