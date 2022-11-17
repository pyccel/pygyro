import numpy as np

from pygyro.model.grid import Grid
from pygyro.model.layout import Layout
from pygyro.initialisation.initialiser_funcs import make_f_eq_grid, make_n0_grid


def make_trapz_grid(grid):
    """
    Generate a trapezoidal grid from a grid
    """
    d_grid = grid[1:] - grid[:-1]
    trapz_grid = np.append(d_grid[0] * 0.5,
                           np.append((d_grid + np.roll(d_grid, 1))[1:] * 0.5,
                           d_grid[-1] * 0.5))
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

        # global grids
        r = eta_grid[0]
        q = eta_grid[1]
        z = eta_grid[2]
        v = eta_grid[3]

        while z[0] != np.min(z):
            z = np.roll(z, shift=-1)

        # local grids
        my_r = r[layout.starts[self.idx_r]:layout.ends[self.idx_r]]
        my_q = q[layout.starts[self.idx_q]:layout.ends[self.idx_q]]
        my_z = z[layout.starts[self.idx_z]:layout.ends[self.idx_z]]
        self.z_start = layout.starts[self.idx_z]
        self.z_end = layout.ends[self.idx_z]
        my_v = v[layout.starts[self.idx_v]:layout.ends[self.idx_v]]

        # Make trapezoidal grid for integration over v
        dvMult = make_trapz_grid(v)
        self.mydvMult = dvMult[layout.starts[self.idx_v]:layout.ends[self.idx_v]] \
            * (my_v ** 2)
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
        self.mydqMult = dqMult[layout.starts[self.idx_q]
            :layout.ends[self.idx_q]]
        shape_q = [1, 1, 1]
        shape_q[self.idx_q] = my_q.size
        self.mydqMult.resize(shape_q)

        # Make trapezoidal grid for integration over z
        dzMult = make_trapz_grid(z)
        self.mydzMult = dzMult[layout.starts[self.idx_z]
            :layout.ends[self.idx_z]]
        shape_z = [1, 1]
        shape_z[self.idx_z] = my_z.size
        self.mydzMult.resize(shape_z)

        # Make trapezoidal grid for integration over r
        drMult = make_trapz_grid(r)
        self.mydrMult = drMult[layout.starts[self.idx_r]
            :layout.ends[self.idx_r]] * my_r
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
        int_v = - np.sum(np.abs((f._f - self.f_eq)) * self.mydvMult, axis=self.idx_v)

        int_q = np.sum(int_v * self.mydqMult, axis=self.idx_q)
        int_z = np.sum(int_q * self.mydzMult, axis=self.idx_z)
        int_r = np.sum(int_z * self.mydrMult, axis=self.idx_r)

        return int_r * 0.5


class Mass_f:
    """
    Layout has to be in v_parallel, i.e.:
        (r, z, theta, v)

    Integrate first over v, then theta, then z, then r
    """

    def __init__(self, eta_grid: list, layout: Layout):
        assert layout.name == 'v_parallel', f'Layout is {layout.name} not v_parallel'
        self._layout = layout.name

        self.idx_r = layout.inv_dims_order[0]
        self.idx_q = layout.inv_dims_order[1]
        self.idx_z = layout.inv_dims_order[2]
        self.idx_v = layout.inv_dims_order[3]

        # global grids
        r = eta_grid[0]
        q = eta_grid[1]
        z = eta_grid[2]
        v = eta_grid[3]

        while z[0] != np.min(z):
            z = np.roll(z, shift=-1)

        # local grids
        my_r = r[layout.starts[self.idx_r]:layout.ends[self.idx_r]]
        my_q = q[layout.starts[self.idx_q]:layout.ends[self.idx_q]]
        my_z = z[layout.starts[self.idx_z]:layout.ends[self.idx_z]]
        self.z_start = layout.starts[self.idx_z]
        self.z_end = layout.ends[self.idx_z]
        my_v = v[layout.starts[self.idx_v]:layout.ends[self.idx_v]]

        shape_phi = [1, 1, 1]
        shape_phi[self.idx_r] = my_r.size
        shape_phi[self.idx_z] = my_z.size
        shape_phi[self.idx_q] = my_q.size
        self._shape_phi = shape_phi

        # Make trapezoidal grid for integration over v
        dvMult = make_trapz_grid(v)
        self.mydvMult = dvMult[layout.starts[self.idx_v]
            :layout.ends[self.idx_v]]
        shape_v = [1, 1, 1, 1]
        shape_v[self.idx_v] = my_v.size
        self.mydvMult.resize(shape_v)

        # Make trapezoidal grid for integration over theta (aka q)
        dqMult = make_trapz_grid(q)
        self.mydqMult = dqMult[layout.starts[self.idx_q]
            :layout.ends[self.idx_q]]
        shape_q = [1, 1, 1]
        shape_q[self.idx_q] = my_q.size
        self.mydqMult.resize(shape_q)

        # Make trapezoidal grid for integration over z
        dzMult = make_trapz_grid(z)
        self.mydzMult = dzMult[layout.starts[self.idx_z]
            :layout.ends[self.idx_z]]
        shape_z = [1, 1]
        shape_z[self.idx_z] = my_z.size
        self.mydzMult.resize(shape_z)

        # Make trapezoidal grid for integration over r
        drMult = make_trapz_grid(r)
        self.mydrMult = drMult[layout.starts[self.idx_r]
            :layout.ends[self.idx_r]] * my_r
        shape_r = [1]
        shape_r[self.idx_r] = my_r.size
        self.mydrMult.resize(shape_r)

    def getMASSF(self, f: Grid):
        """
        Layout has to be in v_parallel, i.e.:
            (r, z, theta, v)

        Integrate first over v, then theta, then z, then r
        """
        # must be in v_parallel layout
        assert f.currentLayout == 'v_parallel'

        # Integration over v of f -> yields n_i
        n_i = np.sum(f._f * self.mydvMult, axis=self.idx_v)

        int_q = np.sum(n_i * self.mydqMult, axis=self.idx_q)
        int_z = np.sum(int_q * self.mydzMult, axis=self.idx_z)
        int_r = np.sum(int_z * self.mydrMult, axis=self.idx_r)

        return int_r * 0.5


class L2_f:
    """
    Layout has to be in v_parallel, i.e.:
        (r, z, theta, v)

    Integrate first over v, then theta, then z, then r
    """

    def __init__(self, eta_grid: list, layout: Layout):
        assert layout.name == 'v_parallel', f'Layout is {layout.name} not v_parallel'
        self._layout = layout.name

        self.idx_r = layout.inv_dims_order[0]
        self.idx_q = layout.inv_dims_order[1]
        self.idx_z = layout.inv_dims_order[2]
        self.idx_v = layout.inv_dims_order[3]

        # global grids
        r = eta_grid[0]
        q = eta_grid[1]
        z = eta_grid[2]
        v = eta_grid[3]

        while z[0] != np.min(z):
            z = np.roll(z, shift=-1)

        # local grids
        my_r = r[layout.starts[self.idx_r]:layout.ends[self.idx_r]]
        my_q = q[layout.starts[self.idx_q]:layout.ends[self.idx_q]]
        my_z = z[layout.starts[self.idx_z]:layout.ends[self.idx_z]]
        self.z_start = layout.starts[self.idx_z]
        self.z_end = layout.ends[self.idx_z]
        my_v = v[layout.starts[self.idx_v]:layout.ends[self.idx_v]]

        shape_phi = [1, 1, 1]
        shape_phi[self.idx_r] = my_r.size
        shape_phi[self.idx_z] = my_z.size
        shape_phi[self.idx_q] = my_q.size
        self._shape_phi = shape_phi

        # Make trapezoidal grid for integration over v
        dvMult = make_trapz_grid(v)
        self.mydvMult = dvMult[layout.starts[self.idx_v]
            :layout.ends[self.idx_v]]
        shape_v = [1, 1, 1, 1]
        shape_v[self.idx_v] = my_v.size
        self.mydvMult.resize(shape_v)

        # Make trapezoidal grid for integration over theta (aka q)
        dqMult = make_trapz_grid(q)
        self.mydqMult = dqMult[layout.starts[self.idx_q]
            :layout.ends[self.idx_q]]
        shape_q = [1, 1, 1]
        shape_q[self.idx_q] = my_q.size
        self.mydqMult.resize(shape_q)

        # Make trapezoidal grid for integration over z
        dzMult = make_trapz_grid(z)
        self.mydzMult = dzMult[layout.starts[self.idx_z]
            :layout.ends[self.idx_z]]
        shape_z = [1, 1]
        shape_z[self.idx_z] = my_z.size
        self.mydzMult.resize(shape_z)

        # Make trapezoidal grid for integration over r
        drMult = make_trapz_grid(r)
        self.mydrMult = drMult[layout.starts[self.idx_r]
            :layout.ends[self.idx_r]] * my_r
        shape_r = [1]
        shape_r[self.idx_r] = my_r.size
        self.mydrMult.resize(shape_r)

    def getL2F(self, f: Grid):
        """
        Layout has to be in v_parallel, i.e.:
            (r, z, theta, v)

        Integrate first over v, then theta, then z, then r
        """
        # must be in v_parallel layout
        assert f.currentLayout == 'v_parallel'

        # Integration over v of f^2
        int_v = np.sum(f._f**2 * self.mydvMult, axis=self.idx_v)

        int_q = np.sum(int_v * self.mydqMult,
                       axis=self.idx_q)
        int_z = np.sum(int_q * self.mydzMult, axis=self.idx_z)
        int_r = np.sum(int_z * self.mydrMult, axis=self.idx_r)

        return int_r


class L2_phi:
    """
    Layout has to be in v_parallel_1d, i.e.:
        (r, z, theta)

    Integrate first over v, then theta, then z, then r
    """

    def __init__(self, eta_grid: list, layout: Layout):
        assert layout.name == 'v_parallel', f'Layout is {layout.name} not v_parallel'
        self._layout = layout.name

        self.idx_r = layout.inv_dims_order[0]
        self.idx_q = layout.inv_dims_order[1]
        self.idx_z = layout.inv_dims_order[2]

        # global grids
        r = eta_grid[0]
        q = eta_grid[1]
        z = eta_grid[2]

        while z[0] != np.min(z):
            z = np.roll(z, shift=-1)

        # local grids
        my_r = r[layout.starts[self.idx_r]:layout.ends[self.idx_r]]
        my_q = q[layout.starts[self.idx_q]:layout.ends[self.idx_q]]
        my_z = z[layout.starts[self.idx_z]:layout.ends[self.idx_z]]
        self.z_start = layout.starts[self.idx_z]
        self.z_end = layout.ends[self.idx_z]

        shape_phi = [1, 1, 1]
        shape_phi[self.idx_r] = my_r.size
        shape_phi[self.idx_z] = my_z.size
        shape_phi[self.idx_q] = my_q.size
        self._shape_phi = shape_phi

        # Make trapezoidal grid for integration over theta (aka q)
        dqMult = make_trapz_grid(q)
        self.mydqMult = dqMult[layout.starts[self.idx_q]
            :layout.ends[self.idx_q]]
        shape_q = [1, 1, 1]
        shape_q[self.idx_q] = my_q.size
        self.mydqMult.resize(shape_q)

        # Make trapezoidal grid for integration over z
        dzMult = make_trapz_grid(z)
        self.mydzMult = dzMult[layout.starts[self.idx_z]
            :layout.ends[self.idx_z]]
        shape_z = [1, 1]
        shape_z[self.idx_z] = my_z.size
        self.mydzMult.resize(shape_z)

        # Make trapezoidal grid for integration over r
        drMult = make_trapz_grid(r)
        self.mydrMult = drMult[layout.starts[self.idx_r]
            :layout.ends[self.idx_r]] * my_r
        shape_r = [1]
        shape_r[self.idx_r] = my_r.size
        self.mydrMult.resize(shape_r)

    def getL2Phi(self, phi: Grid):
        """
        Phi is in 'v_parallel_1d':
            (r, z, theta)

        Integrate first over v, then theta, then z, then r
        """
        # phi must be in v_parallel_1d layout
        assert phi.currentLayout == 'v_parallel_1d'

        int_q = np.sum(np.abs(phi._f[:, self.z_start: self.z_end, :])**2 * self.mydqMult,
                       axis=self.idx_q)
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

        # global grids
        r = eta_grid[0]
        q = eta_grid[1]
        z = eta_grid[2]
        v = eta_grid[3]

        while z[0] != np.min(z):
            z = np.roll(z, shift=-1)

        # local grids
        my_r = r[layout.starts[self.idx_r]:layout.ends[self.idx_r]]
        my_q = q[layout.starts[self.idx_q]:layout.ends[self.idx_q]]
        my_z = z[layout.starts[self.idx_z]:layout.ends[self.idx_z]]
        self.z_start = layout.starts[self.idx_z]
        self.z_end = layout.ends[self.idx_z]
        my_v = v[layout.starts[self.idx_v]:layout.ends[self.idx_v]]

        shape_phi = [1, 1, 1]
        shape_phi[self.idx_r] = my_r.size
        shape_phi[self.idx_z] = my_z.size
        shape_phi[self.idx_q] = my_q.size
        self._shape_phi = shape_phi

        # Make trapezoidal grid for integration over v
        dvMult = make_trapz_grid(v)
        self.mydvMult = dvMult[layout.starts[self.idx_v]
            :layout.ends[self.idx_v]]
        shape_v = [1, 1, 1, 1]
        shape_v[self.idx_v] = my_v.size
        self.mydvMult.resize(shape_v)

        # Make trapezoidal grid for integration over theta (aka q)
        dqMult = make_trapz_grid(q)
        self.mydqMult = dqMult[layout.starts[self.idx_q]
            :layout.ends[self.idx_q]]
        shape_q = [1, 1, 1]
        shape_q[self.idx_q] = my_q.size
        self.mydqMult.resize(shape_q)

        # Make trapezoidal grid for integration over z
        dzMult = make_trapz_grid(z)
        self.mydzMult = dzMult[layout.starts[self.idx_z]
            :layout.ends[self.idx_z]]
        shape_z = [1, 1]
        shape_z[self.idx_z] = my_z.size
        self.mydzMult.resize(shape_z)

        # Make trapezoidal grid for integration over r
        drMult = make_trapz_grid(r)
        self.mydrMult = drMult[layout.starts[self.idx_r]
            :layout.ends[self.idx_r]] * my_r
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

        int_q = np.sum((n_i - self.n_0) * np.real(phi._f[:, self.z_start: self.z_end, :]) * self.mydqMult,
                       axis=self.idx_q)
        int_z = np.sum(int_q * self.mydzMult, axis=self.idx_z)
        int_r = np.sum(int_z * self.mydrMult, axis=self.idx_r)

        return int_r * 0.5
