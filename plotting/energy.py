import numpy as np
from mpi4py import MPI

from pygyro.model.grid import Grid
from pygyro.model.layout import Layout
from pygyro.initialisation.constants import Constants
from pygyro.initialisation.initialiser_funcs import make_f_eq_grid, make_n0_grid, n0, f_eq


def make_trapz_grid(grid):
    """
    Generate a trapezoidal grid from a grid
    """
    d_grid = grid[1:] - grid[:-1]
    trapz_grid = np.append(d_grid[0] * 0.5,
                           np.append((d_grid + np.roll(d_grid, 1))[1:] * 0.5,
                           d_grid[-1] * 0.5))
    return trapz_grid


class KineticEnergy_v2:
    """
    Layout has to be in poloidal, i.e.:
        (v, z, theta, r)

    Integrate first over r, then theta, then z, then v
    """

    def __init__(self, eta_grid: list, layout: Layout, constants: Constants):
        assert layout.name == 'poloidal', f'Layout is {layout.name} not poloidal'
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

        # Introduce shift because z variable is another bish
        self.shift = 0
        while z[0] != np.min(z):
            z = np.roll(z, shift=-1)
            self.shift -= 1

        # local grids
        my_r = r[layout.starts[self.idx_r]:layout.ends[self.idx_r]]
        my_q = q[layout.starts[self.idx_q]:layout.ends[self.idx_q]]
        my_z = z[layout.starts[self.idx_z]:layout.ends[self.idx_z]]
        self.z_start = layout.starts[self.idx_z]
        self.z_end = layout.ends[self.idx_z]
        my_v = v[layout.starts[self.idx_v]:layout.ends[self.idx_v]]

        # Make trapezoidal grid for integration over r
        drMult = make_trapz_grid(r)
        self.mydrMult = drMult[layout.starts[self.idx_r]:layout.ends[self.idx_r]] \
            * my_r
        shape_r = [1, 1, 1, 1]
        shape_r[self.idx_r] = my_r.size
        self.mydrMult.resize(shape_r)

        # Make f_eq array (only depends on v and r but has to have full size)
        self.f_eq = np.zeros((my_v.size, my_r.size), dtype=float)
        make_f_eq_grid(constants.CN0, constants.kN0, constants.deltaRN0, constants.rp,
                       constants.CTi, constants.kTi, constants.deltaRTi, my_v, my_r, self.f_eq)
        shape_f_eq = [1, 1, 1, 1]
        shape_f_eq[self.idx_v] = my_v.size
        shape_f_eq[self.idx_r] = my_r.size
        self.f_eq.resize(shape_f_eq)

        # Use simple approximation of the integral by summing because theta variable is a bish
        self.dq = q[1] - q[0]
        assert self.dq * q.size - 2 * np.pi < 1e-7, "Grid spacing in theta direction is wrong"

        # Make trapezoidal grid for integration over z, shift because z is also a bish
        dzMult = make_trapz_grid(z)
        self.mydzMult = np.roll(dzMult, shift=-self.shift)[layout.starts[self.idx_z]:layout.ends[self.idx_z]]
        shape_z = [1, 1]
        shape_z[self.idx_z] = my_z.size
        self.mydzMult.resize(shape_z)

        # Make trapezoidal grid for integration over v
        dvMult = make_trapz_grid(v)
        self.mydvMult = dvMult[layout.starts[self.idx_v]:layout.ends[self.idx_v]] \
            * (my_v ** 2)
        shape_v = [1]
        shape_v[self.idx_v] = my_v.size
        self.mydvMult.resize(shape_v)

    def getKE(self, f: Grid):
        """
        Layout has to be in poloidal, i.e.:
            (v, z, theta, r)

        Integrate first over r, then theta, then z, then v
        """
        # must be in poloidal layout
        assert f.currentLayout == 'poloidal'

        # Integration over r of (f - f_eq)
        int_r = np.sum((f._f - self.f_eq) * self.mydrMult, axis=self.idx_r)

        # Integration over theta by summation
        int_q = np.sum(int_r, axis=self.idx_q) * self.dq

        # Integrate over z
        int_z = np.sum(int_q * self.mydzMult, axis=self.idx_z)
        int_v = np.sum(int_z * self.mydvMult, axis=self.idx_v)

        return int_v * 0.5


class Mass_f:
    """
    Layout has to be in poloidal, i.e.:
        (v, z, theta, r)

    Integrate first over r, then theta, then z, then v
    """

    def __init__(self, eta_grid: list, layout: Layout):
        assert layout.name == 'poloidal', f'Layout is {layout.name} not poloidal'
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

        # Introduce shift because z variable is another bish
        self.shift = 0
        while z[0] != np.min(z):
            z = np.roll(z, shift=-1)
            self.shift -= 1

        # local grids
        my_r = r[layout.starts[self.idx_r]:layout.ends[self.idx_r]]
        my_q = q[layout.starts[self.idx_q]:layout.ends[self.idx_q]]
        my_z = z[layout.starts[self.idx_z]:layout.ends[self.idx_z]]
        self.z_start = layout.starts[self.idx_z]
        self.z_end = layout.ends[self.idx_z]
        my_v = v[layout.starts[self.idx_v]:layout.ends[self.idx_v]]

        # Make trapezoidal grid for integration over r
        drMult = make_trapz_grid(r)
        self.mydrMult = drMult[layout.starts[self.idx_r]:layout.ends[self.idx_r]] \
            * my_r
        shape_r = [1, 1, 1, 1]
        shape_r[self.idx_r] = my_r.size
        self.mydrMult.resize(shape_r)

        # Use simple approximation of the integral by summing because theta variable is a bish
        self.dq = q[1] - q[0]
        assert self.dq * q.size - 2 * np.pi < 1e-7, "Grid spacing in theta direction is wrong"

        # Make trapezoidal grid for integration over z, shift because z is also a bish
        dzMult = make_trapz_grid(z)
        self.mydzMult = np.roll(dzMult, shift=-self.shift)[layout.starts[self.idx_z]:layout.ends[self.idx_z]]
        shape_z = [1, 1]
        shape_z[self.idx_z] = my_z.size
        self.mydzMult.resize(shape_z)

        # Make trapezoidal grid for integration over v
        dvMult = make_trapz_grid(v)
        self.mydvMult = dvMult[layout.starts[self.idx_v]:layout.ends[self.idx_v]]
        shape_v = [1]
        shape_v[self.idx_v] = my_v.size
        self.mydvMult.resize(shape_v)

        # Trapezoidal grid for integrating f_eq over v and comparing to n_0
        shape_dv_int_f_eq = [1, 1, 1, 1]
        shape_dv_int_f_eq[self.idx_v] = my_v.size
        self.mydvMult_int_f_eq = self.mydvMult.reshape(shape_dv_int_f_eq)

    def getMASSF(self, f: Grid):
        """
        Layout has to be in poloidal, i.e.:
            (v, z, theta, r)

        Integrate first over r, then theta, then z, then v
        """
        # must be in poloidal layout
        assert f.currentLayout == 'poloidal'

        # Integration over r of f
        int_r = np.sum(f._f * self.mydrMult, axis=self.idx_r)

        int_q = np.sum(int_r, axis=self.idx_q) * self.dq
        int_z = np.sum(int_q * self.mydzMult, axis=self.idx_z)
        int_v = np.sum(int_z * self.mydvMult, axis=self.idx_v)

        return int_v * 0.5


class L2_f:
    """
    Layout has to be in poloidal, i.e.:
        (v, z, theta, r)

    Integrate first over r, then theta, then z, then v
    """

    def __init__(self, eta_grid: list, layout: Layout):
        assert layout.name == 'poloidal', f'Layout is {layout.name} not poloidal'
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

        # Introduce shift because z variable is another bish
        self.shift = 0
        while z[0] != np.min(z):
            z = np.roll(z, shift=-1)
            self.shift -= 1

        # local grids
        my_r = r[layout.starts[self.idx_r]:layout.ends[self.idx_r]]
        my_q = q[layout.starts[self.idx_q]:layout.ends[self.idx_q]]
        my_z = z[layout.starts[self.idx_z]:layout.ends[self.idx_z]]
        self.z_start = layout.starts[self.idx_z]
        self.z_end = layout.ends[self.idx_z]
        my_v = v[layout.starts[self.idx_v]:layout.ends[self.idx_v]]

        # Make trapezoidal grid for integration over r
        drMult = make_trapz_grid(r)
        self.mydrMult = drMult[layout.starts[self.idx_r]:layout.ends[self.idx_r]] \
            * my_r
        shape_r = [1, 1, 1, 1]
        shape_r[self.idx_r] = my_r.size
        self.mydrMult.resize(shape_r)

        # Use simple approximation of the integral by summing because theta variable is a bish
        self.dq = q[1] - q[0]
        assert self.dq * q.size - 2 * np.pi < 1e-7, "Grid spacing in theta direction is wrong"

        # Make trapezoidal grid for integration over z, shift because z is also a bish
        dzMult = make_trapz_grid(z)
        self.mydzMult = np.roll(dzMult, shift=-self.shift)[layout.starts[self.idx_z]:layout.ends[self.idx_z]]
        shape_z = [1, 1]
        shape_z[self.idx_z] = my_z.size
        self.mydzMult.resize(shape_z)

        # Make trapezoidal grid for integration over v
        dvMult = make_trapz_grid(v)
        self.mydvMult = dvMult[layout.starts[self.idx_v]:layout.ends[self.idx_v]]
        shape_v = [1]
        shape_v[self.idx_v] = my_v.size
        self.mydvMult.resize(shape_v)

    def getL2F(self, f: Grid):
        """
        Layout has to be in poloidal, i.e.:
            (v, z, theta, r)

        Integrate first over r, then theta, then z, then v
        """
        # must be in poloidal layout
        assert f.currentLayout == 'poloidal'

        # Integration over v of f^2
        int_r = np.sum(np.abs(f._f)**2 * self.mydrMult, axis=self.idx_r)

        int_q = np.sum(int_r, axis=self.idx_q) * self.dq
        int_z = np.sum(int_q * self.mydzMult, axis=self.idx_z)
        int_v = np.sum(int_z * self.mydvMult, axis=self.idx_v)

        return int_v


class L2_phi:
    """
    Layout has to be in poloidal, i.e.:
        (v, z, theta, r)

    Integrate first over r, then theta, then z
    """

    def __init__(self, eta_grid: list, layout: Layout):
        assert layout.name == 'poloidal', f'Layout is {layout.name} not poloidal'
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

        # Introduce shift because z variable is another bish
        self.shift = 0
        while z[0] != np.min(z):
            z = np.roll(z, shift=-1)
            self.shift -= 1

        # local grids
        my_r = r[layout.starts[self.idx_r]:layout.ends[self.idx_r]]
        my_q = q[layout.starts[self.idx_q]:layout.ends[self.idx_q]]
        my_z = z[layout.starts[self.idx_z]:layout.ends[self.idx_z]]
        self.z_start = layout.starts[self.idx_z]
        self.z_end = layout.ends[self.idx_z]

        # Make trapezoidal grid for integration over r
        drMult = make_trapz_grid(r)
        self.mydrMult = drMult[layout.starts[self.idx_r]:layout.ends[self.idx_r]] \
            * my_r
        shape_r = [1, 1, 1, 1]
        shape_r[self.idx_r] = my_r.size
        self.mydrMult.resize(shape_r)

        # Use simple approximation of the integral by summing because theta variable is a bish
        self.dq = q[1] - q[0]
        assert self.dq * q.size - 2 * np.pi < 1e-7, "Grid spacing in theta direction is wrong"

        # Make trapezoidal grid for integration over z, shift because z is also a bish
        dzMult = make_trapz_grid(z)
        self.mydzMult = np.roll(dzMult, shift=-self.shift)[layout.starts[self.idx_z]:layout.ends[self.idx_z]]
        shape_z = [1, 1]
        shape_z[self.idx_z] = my_z.size
        self.mydzMult.resize(shape_z)

    def getL2Phi(self, phi: Grid):
        """
        Phi has to be in poloidal, i.e.:
            (z, theta, r)

        Integrate first over r, then theta, then z
        """
        # phi must be in poloidal layout
        assert phi.currentLayout == 'poloidal'

        int_r = np.sum(np.abs(phi._f)**2 * self.mydrMult,
                       axis=self.idx_r)
        int_q = np.sum(int_r, axis=self.idx_q) * self.dq
        int_z = np.sum(int_q * self.mydzMult, axis=self.idx_z)

        return int_z * 0.5


class PotentialEnergy_v2:
    """
    Layout has to be in poloidal, i.e.:
        (v, z, theta, r)

    Integrate first over r, then theta, then z, then v
    """

    def __init__(self, eta_grid: list, layout: Layout, constants: Constants):
        assert layout.name == 'poloidal', f'Layout is {layout.name} not poloidal'
        self._layout = layout.name

        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.idx_r = layout.inv_dims_order[0]
        self.idx_q = layout.inv_dims_order[1]
        self.idx_z = layout.inv_dims_order[2]
        self.idx_v = layout.inv_dims_order[3]

        # global grids
        r = eta_grid[0]
        q = eta_grid[1]
        z = eta_grid[2]
        v = eta_grid[3]

        # Introduce shift because z variable is another bish
        self.shift = 0
        while z[0] != np.min(z):
            z = np.roll(z, shift=-1)
            self.shift -= 1

        # local grids
        my_r = r[layout.starts[self.idx_r]:layout.ends[self.idx_r]]
        my_q = q[layout.starts[self.idx_q]:layout.ends[self.idx_q]]
        my_z = z[layout.starts[self.idx_z]:layout.ends[self.idx_z]]
        self.z_start = layout.starts[self.idx_z]
        self.z_end = layout.ends[self.idx_z]
        my_v = v[layout.starts[self.idx_v]:layout.ends[self.idx_v]]

        # Make trapezoidal grid for integration over r
        drMult = make_trapz_grid(r)
        self.mydrMult = drMult[layout.starts[self.idx_r]:layout.ends[self.idx_r]] \
            * my_r
        shape_r = [1, 1, 1, 1]
        shape_r[self.idx_r] = my_r.size
        self.mydrMult.resize(shape_r)

        # Use simple approximation of the integral by summing because theta variable is a bish
        self.dq = q[1] - q[0]
        assert self.dq * q.size - 2 * np.pi < 1e-7, "Grid spacing in theta direction is wrong"

        # Make trapezoidal grid for integration over z, shift because z is also a bish
        dzMult = make_trapz_grid(z)
        self.mydzMult = np.roll(dzMult, shift=-self.shift)[layout.starts[self.idx_z]:layout.ends[self.idx_z]]
        shape_z = [1, 1]
        shape_z[self.idx_z] = my_z.size
        self.mydzMult.resize(shape_z)

        # Make trapezoidal grid for integration over v
        dvMult = make_trapz_grid(v)
        self.mydvMult = dvMult[layout.starts[self.idx_v]:layout.ends[self.idx_v]]
        shape_v = [1]
        shape_v[self.idx_v] = my_v.size
        self.mydvMult.resize(shape_v)

        # Make f_eq array (only depends on v and r but has to have full size)
        self.f_eq = np.zeros((my_v.size, my_r.size), dtype=float)
        make_f_eq_grid(constants.CN0, constants.kN0, constants.deltaRN0, constants.rp,
                       constants.CTi, constants.kTi, constants.deltaRTi, my_v, my_r, self.f_eq)
        shape_f_eq = [1, 1, 1, 1]
        shape_f_eq[self.idx_v] = my_v.size
        shape_f_eq[self.idx_r] = my_r.size
        self.f_eq.resize(shape_f_eq)

    def getPE(self, f: Grid, phi: Grid):
        """
        Layout has to be in poloidal, i.e.:
            (v, z, theta, r)

        Phi has to be in poloidal, i.e.:
            (z, theta, r)

        Integrate first over r, then theta, then z, then v
        """
        # must be in poloidal layout
        assert f.currentLayout == 'poloidal'

        # phi must be in poloidal layout
        assert phi.currentLayout == 'poloidal'

        # Integration over r of (f - n_0) * phi
        int_r = np.sum((f._f - self.f_eq) * np.real(phi._f) \
                       * self.mydrMult, axis=self.idx_r)

        # Integrate over theta, then z, then v
        int_q = np.sum(int_r, axis=self.idx_q) * self.dq
        int_z = np.sum(int_q * self.mydzMult, axis=self.idx_z)
        int_v = np.sum(int_z * self.mydvMult, axis=self.idx_v)

        return int_v * 0.5
