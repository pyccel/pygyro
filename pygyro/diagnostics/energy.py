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
    Layout has to be in v_parallel, i.e.:
        (r, z, theta, v)

    Integrate first over v, then theta, then z, then r
    """

    def __init__(self, eta_grid: list, layout: Layout, constants):
        assert layout.name == 'v_parallel', f'Layout is {layout.name} not v_parallel'
        self._layout = layout.name

        self.idx_r = layout.inv_dims_order[0]
        self.idx_q = layout.inv_dims_order[1]
        self.idx_z = layout.inv_dims_order[2]
        self.idx_v = layout.inv_dims_order[3]

        # local grids
        my_r = eta_grid[0][layout.starts[self.idx_r]:layout.ends[self.idx_r]]
        my_q = eta_grid[1][layout.starts[self.idx_q]:layout.ends[self.idx_q]]
        my_z = eta_grid[2][layout.starts[self.idx_z]:layout.ends[self.idx_z]]
        my_v = eta_grid[3][layout.starts[self.idx_v]:layout.ends[self.idx_v]]

        # global grids
        r = eta_grid[0]
        q = eta_grid[1]
        z = eta_grid[2]
        v = eta_grid[3]

        # Make trapezoidal grid for integration over v
        dvMult = make_trapz_grid(v)
        self.mydvMult = dvMult[layout.starts[self.idx_v]:layout.ends[self.idx_v]]
        shape_v = [1, 1, 1, 1]
        shape_v[self.idx_v] = my_v.size
        self.mydvMult.resize(shape_v)

        # Make f_eq array (only depends on r and v but has to have full size)
        self.f_eq = np.zeros((my_r.size, my_v.size), dtype=float)
        make_f_eq_grid(constants.CN0, constants.kN0, constants.deltaRN0, constants.rp,
                       constants.CTi, constants.kTi, constants.deltaRTi, my_r, my_v, self.f_eq)
        shape_f_eq = [1, 1, 1, 1]
        shape_f_eq[self.idx_r] = my_r.size
        shape_f_eq[self.idx_v] = my_v.size
        self.f_eq.resize(shape_f_eq)

        # Make trapezoidal grid for integration over theta (aka q)
        dqMult = make_trapz_grid(q)
        self.mydqMult = dqMult[layout.starts[self.idx_q]:layout.ends[self.idx_q]]
        shape_q = [1, 1, 1]
        shape_q[self.idx_q] = my_q.size
        self.mydqMult.resize(shape_q)

        # Make trapezoidal grid for integration over z
        dzMult = make_trapz_grid(z)
        self.mydzMult = dzMult[layout.starts[self.idx_z]:layout.ends[self.idx_z]]
        shape_z = [1, 1]
        shape_z[self.idx_z] = my_z.size
        self.mydzMult.resize(shape_z)

        # Make trapezoidal grid for integration over r
        drMult = make_trapz_grid(r)
        self.mydrMult = drMult[layout.starts[self.idx_r]:layout.ends[self.idx_r]] * my_r
        shape_r = [1]
        shape_r[self.idx_r] = my_r.size
        self.mydrMult.resize(shape_r)

    def getKE(self, f: Grid):
        """
        Integrate first over v, then theta, then z, then r
        """
        # must be in v_parallel layout
        assert f.currentLayout == 'v_parallel'

        # Integration over v of (f - f_eq)
        int_v = np.sum((f._f - self.f_eq) * self.mydvMult, axis=self.idx_v)

        int_q = np.sum(int_v * self.mydqMult, axis=self.idx_q)
        int_z = np.sum(int_q * self.mydzMult, axis=self.idx_z)
        int_r = np.sum(int_z * self.mydrMult, axis=self.idx_r)

        return int_r * 0.5


class PotentialEnergy:
    """
    Layout has to be in v_parallel, i.e.:
        (r, z, theta, v)

    Integrate first over v, then theta, then z, then r
    """

    def __init__(self, eta_grid: list, layout: Layout, constants):
        assert layout.name == 'v_parallel', f'Layout is {layout.name} not v_parallel'
        self._layout = layout.name

        self.idx_r = layout.inv_dims_order[0]
        self.idx_q = layout.inv_dims_order[1]
        self.idx_z = layout.inv_dims_order[2]
        self.idx_v = layout.inv_dims_order[3]

        # local grids
        my_r = eta_grid[0][layout.starts[self.idx_r]:layout.ends[self.idx_r]]
        my_q = eta_grid[1][layout.starts[self.idx_q]:layout.ends[self.idx_q]]
        my_z = eta_grid[2][layout.starts[self.idx_z]:layout.ends[self.idx_z]]
        self.z_start = layout.starts[self.idx_z]
        self.z_end = layout.ends[self.idx_z]
        my_v = eta_grid[3][layout.starts[self.idx_v]:layout.ends[self.idx_v]]

        # global grids
        r = eta_grid[0]
        q = eta_grid[1]
        z = eta_grid[2]
        v = eta_grid[3]

        shape_phi = [1, 1, 1]
        shape_phi[self.idx_r] = my_r.size
        shape_phi[self.idx_z] = my_z.size
        shape_phi[self.idx_q] = my_q.size
        self._shape_phi = shape_phi

        # Make trapezoidal grid for integration over v
        dvMult = make_trapz_grid(v)
        self.mydvMult = dvMult[layout.starts[self.idx_v]:layout.ends[self.idx_v]]
        shape_v = [1, 1, 1, 1]
        shape_v[self.idx_v] = my_v.size
        self.mydvMult.resize(shape_v)

        # Make trapezoidal grid for integration over theta (aka q)
        dqMult = make_trapz_grid(q)
        self.mydqMult = dqMult[layout.starts[self.idx_q]:layout.ends[self.idx_q]]
        shape_q = [1, 1, 1]
        shape_q[self.idx_q] = my_q.size
        self.mydqMult.resize(shape_q)

        # Make trapezoidal grid for integration over z
        dzMult = make_trapz_grid(z)
        self.mydzMult = dzMult[layout.starts[self.idx_z]:layout.ends[self.idx_z]]
        shape_z = [1, 1]
        shape_z[self.idx_z] = my_z.size
        self.mydzMult.resize(shape_z)

        # Make trapezoidal grid for integration over r
        drMult = make_trapz_grid(r)
        self.mydrMult = drMult[layout.starts[self.idx_r]:layout.ends[self.idx_r]] * my_r
        shape_r = [1]
        shape_r[self.idx_r] = my_r.size
        self.mydrMult.resize(shape_r)

        # Make n_0 array (only depends on r but must be 3d (r, z, theta))
        self.n_0 = np.zeros(my_r.size, dtype=float)
        make_n0_grid(constants.CN0, constants.kN0,
                     constants.deltaRN0, constants.rp, my_r, self.n_0)
        shape_n0 = [1, 1, 1]
        shape_n0[self.idx_r] = my_r.size
        self.n_0.resize(shape_n0)

    def getPE(self, f: Grid, phi: Grid):
        """
        Layout has to be in v_parallel, i.e.:
            (r, z, theta, v)
        
        Phi is in 'v_parallel_1d':
            (r, z, theta)

        Integrate first over v, then theta, then z, then r
        """
        # must be in v_parallel layout
        assert f.currentLayout == 'v_parallel'

        # phi must be in v_parallel_1d layout
        assert phi.currentLayout == 'v_parallel_1d'

        # Integration over v of f -> yields n_i
        n_i = np.sum(f._f * self.mydvMult, axis=self.idx_v)

        # print(f'shape of f : {np.shape(f._f)}')
        # print(f'shape of dvMult : {np.shape(self.mydvMult)}')
        # print(f'shape of f * dvMult : {np.shape(f._f * self.mydvMult)}')
        # print(f'shape of n_i : {np.shape(n_i)}')
        # print(f'shape of n0 : {np.shape(self.n_0)}')
        # print(f'shape of phi._f : {np.shape(phi._f)}')
        # print(f'layout of phi._f : {phi.currentLayout}')
        # print(f'shape of drMult : {np.shape(self.mydrMult)}')

        int_q = np.sum((n_i - self.n_0) * np.real(phi._f[:, self.z_start:self.z_end, :]) * self.mydqMult,
                       axis=self.idx_q)
        int_z = np.sum(int_q * self.mydzMult, axis=self.idx_z)
        int_r = np.sum(int_z * self.mydrMult, axis=self.idx_r)

        return int_r * 0.5


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
