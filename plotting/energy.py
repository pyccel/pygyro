import numpy as np
from mpi4py import MPI

from pygyro.model.grid import Grid
from pygyro.model.layout import Layout
from pygyro.initialisation.constants import Constants
from pygyro.initialisation.initialiser_funcs import make_f_eq_grid, f_eq, n0


def make_trapz_grid(grid):
    """
    Generate a trapezoidal grid from a grid
    """
    d_grid = grid[1:] - grid[:-1]
    trapz_grid = np.zeros(grid.shape, dtype=float)
    trapz_grid[0] = d_grid[0] * 0.5
    trapz_grid[1:-1] = (d_grid + np.roll(d_grid, 1))[1:] * 0.5
    trapz_grid[-1] = d_grid[-1] * 0.5
    return trapz_grid


class KineticEnergy_v2:
    """
    Layout has to be in poloidal, i.e.:
        (v, z, theta, r)

    Integrate first over r, then theta, then z, then v
    """

    def __init__(self, eta_grid: list, layout: Layout, constants: Constants, with_feq: bool = True):
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

        # local grids
        my_r = r[layout.starts[self.idx_r]:layout.ends[self.idx_r]]
        self.z_start = layout.starts[self.idx_z]
        self.z_end = layout.ends[self.idx_z]
        my_v = v[layout.starts[self.idx_v]:layout.ends[self.idx_v]]

        # Make trapezoidal grid for integration over r
        drMult = make_trapz_grid(r)
        self.mydrMult = drMult[layout.starts[self.idx_r]:layout.ends[self.idx_r]] \
            # * my_r
        shape_r = [1, 1, 1, 1]
        shape_r[self.idx_r] = my_r.size
        self.mydrMult.resize(shape_r)

        # Make f_eq array (only depends on v and r but has to have full size)
        self.f_eq = np.zeros((my_v.size, my_r.size), dtype=float)
        if with_feq:
            make_f_eq_grid(constants.CN0, constants.kN0, constants.deltaRN0, constants.rp,
                        constants.CTi, constants.kTi, constants.deltaRTi, my_v, my_r, self.f_eq)
        else:
            assert np.all(self.f_eq[:, :] == 0.)
        shape_f_eq = [1, 1, 1, 1]
        shape_f_eq[self.idx_v] = my_v.size
        shape_f_eq[self.idx_r] = my_r.size
        self.f_eq.resize(shape_f_eq)

        # trapezoidal rule for periodic, uniform grid is just summing
        self.dq = q[1] - q[0]
        assert np.abs(self.dq * q.size - 2 * np.pi) < 1e-7, \
            "Grid spacing in theta direction is wrong"

        # trapezoidal rule for periodic, uniform grid is just summing
        self.dz = z[2] - z[1]
        assert np.all(z[2:] - z[1:-1] - self.dz < 1e-10), f'dz = {self.dz} but grid is \n {z[2:] - z[1:-1]}'
        # because z is periodic and so last entry is zMax - dz
        # first entry can be either 0 or zMax (because of spline reasons)
        assert np.abs(self.dz * (z.size - 1) - z[-1]) < 1e-7, \
            f"{self.dz} * {z.size} != {np.max(z)}"

        # Make trapezoidal grid for integration over v
        dvMult = make_trapz_grid(v)
        self.mydvMult = dvMult[layout.starts[self.idx_v]:layout.ends[self.idx_v]] \
            # * (my_v ** 2)
        shape_v = [1]
        shape_v[self.idx_v] = my_v.size
        self.mydvMult.resize(shape_v)

        # temp = [slice(None)] * 2
        # temp[self.idx_z] = 0
        # self.inds = tuple(temp)

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

        # Integration over theta, then z, then v
        int_q = np.sum(int_r, axis=self.idx_q) * self.dq
        int_z = np.sum(int_q, axis=self.idx_z) * self.dz
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

        # local grids
        my_r = r[layout.starts[self.idx_r]:layout.ends[self.idx_r]]
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

        # Use simple approximation of the integral by summing because z variable is also a bish
        self.dz = z[2] - z[1]
        # because z is periodic and so last entry is zMax - dz
        # first entry can be either 0 or zMax
        assert np.abs(self.dz * (z.size - 1) - z[-1]) < 1e-7, \
            f"{self.dz} * {z.size} != {np.max(z)}"

        # Make trapezoidal grid for integration over v
        dvMult = make_trapz_grid(v)
        self.mydvMult = dvMult[layout.starts[self.idx_v]:layout.ends[self.idx_v]]
        shape_v = [1]
        shape_v[self.idx_v] = my_v.size
        self.mydvMult.resize(shape_v)

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
        int_z = np.sum(int_q, axis=self.idx_z) * self.dz
        int_v = np.sum(int_z * self.mydvMult, axis=self.idx_v)

        return int_v


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

        # local grids
        my_r = r[layout.starts[self.idx_r]:layout.ends[self.idx_r]]
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

        # Use simple approximation of the integral by summing because z variable is also a bish
        self.dz = z[2] - z[1]
        # because z is periodic and so last entry is zMax - dz
        # first entry can be either 0 or zMax
        assert np.abs(self.dz * (z.size - 1) - z[-1]) < 1e-7, \
            f"{self.dz} * {z.size} != {np.max(z)}"

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
        int_z = np.sum(int_q, axis=self.idx_z) * self.dz
        int_v = np.sum(int_z * self.mydvMult, axis=self.idx_v)

        return int_v


class L2_phi:
    """
    Layout has to be in poloidal, i.e.:
        (z, theta, r)

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

        # local grids
        my_r = r[layout.starts[self.idx_r]:layout.ends[self.idx_r]]
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

        # Use simple approximation of the integral by summing because z variable is also a bish
        self.dz = z[2] - z[1]
        # because z is periodic and so last entry is zMax - dz
        # first entry can be either 0 or zMax
        assert np.abs(self.dz * (z.size - 1) - z[-1]) < 1e-7, \
            f"{self.dz} * {z.size} != {np.max(z)}"

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
        int_z = np.sum(int_q, axis=self.idx_z) * self.dz

        return int_z


class PotentialEnergy_v2:
    """
    Layout has to be in poloidal, i.e.:
        (v, z, theta, r)

    Integrate first over r, then theta, then z, then v
    """

    def __init__(self, eta_grid: list, layout: Layout, constants: Constants, with_feq: bool = True):
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

        # local grids
        my_r = r[layout.starts[self.idx_r]:layout.ends[self.idx_r]]
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

        # Use simple approximation of the integral by summing because z variable is also a bish
        self.dz = z[2] - z[1]
        # because z is periodic and so last entry is zMax - dz
        # first entry can be either 0 or zMax
        assert np.abs(self.dz * (z.size - 1) - z[-1]) < 1e-7, \
            f"{self.dz} * {z.size} != {np.max(z)}"

        # Make trapezoidal grid for integration over v
        dvMult = make_trapz_grid(v)
        self.mydvMult = dvMult[layout.starts[self.idx_v]:layout.ends[self.idx_v]]
        shape_v = [1]
        shape_v[self.idx_v] = my_v.size
        self.mydvMult.resize(shape_v)

        # Make f_eq array (only depends on v and r but has to have full size)
        self.f_eq = np.zeros((my_v.size, my_r.size), dtype=float)
        if with_feq:
            make_f_eq_grid(constants.CN0, constants.kN0, constants.deltaRN0, constants.rp,
                        constants.CTi, constants.kTi, constants.deltaRTi, my_v, my_r, self.f_eq)
        else:
            assert np.all(self.f_eq[:, :] == 0.)
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
        int_r = np.sum((f._f - self.f_eq) * np.real(phi._f)
                       * self.mydrMult, axis=self.idx_r)

        # Integrate over theta, then z, then v
        int_q = np.sum(int_r, axis=self.idx_q) * self.dq
        int_z = np.sum(int_q, axis=self.idx_z) * self.dz
        int_v = np.sum(int_z * self.mydvMult, axis=self.idx_v)

        return int_v * 0.5


class KineticEnergy_fresch:
    """
    TODO
    """

    def __init__(self, eta_grid: list, layout, constants):
        idx_r = layout.inv_dims_order[0]
        idx_v = layout.inv_dims_order[3]

        my_r = eta_grid[0][layout.starts[idx_r]:layout.ends[idx_r]]
        my_v = eta_grid[3][layout.starts[idx_v]:layout.ends[idx_v]]

        r = eta_grid[0]
        q = eta_grid[1]
        z = eta_grid[2]
        v = eta_grid[3]

        dr = r[1:] - r[:-1]
        dv = v[1:] - v[:-1]

        shape = [1, 1, 1, 1]
        shape[idx_r] = my_r.size
        shape[idx_v] = my_v.size
        self._my_feq = np.empty(shape)
        if (idx_r < idx_v):
            my_feq = [f_eq(r, v, constants.CN0, constants.kN0, constants.deltaRN0, constants.rp,
                           constants.CTi, constants.kTi, constants.deltaRTi)
                      for r in my_r for v in my_v]
        else:
            my_feq = [f_eq(r, v, constants.CN0, constants.kN0, constants.deltaRN0, constants.rp,
                           constants.CTi, constants.kTi, constants.deltaRTi)
                      for v in my_v for r in my_r]
        self._my_feq.flat = my_feq

        drMult = np.array(
            [dr[0] * 0.5, *((dr[1:] + dr[:-1]) * 0.5), dr[-1] * 0.5])
        dvMult = np.array(
            [dv[0] * 0.5, *((dv[1:] + dv[:-1]) * 0.5), dv[-1] * 0.5])

        mydrMult = drMult[layout.starts[idx_r]:layout.ends[idx_r]]
        mydvMult = dvMult[layout.starts[idx_v]:layout.ends[idx_v]]

        shape = [1, 1, 1, 1]
        shape[idx_r] = mydrMult.size
        shape[idx_v] = mydvMult.size

        self._factor1 = np.empty(shape)
        if (idx_r < idx_v):
            self._factor1.flat = (
                (mydrMult * my_r)[:, None] * (mydvMult * my_v**2)[None, :]).flat
        else:
            self._factor1.flat = (
                (mydrMult * my_r)[None, :] * (mydvMult * my_v**2)[:, None]).flat

        self._layout = layout.name

        dq = q[2] - q[1]
        assert dq * eta_grid[1].size - 2 * np.pi < 1e-7

        dz = z[2] - z[1]
        assert dq > 0
        assert dz > 0

        self._factor2 = 0.5 * dq * dz

    def getKE(self, grid):
        """
        TODO
        """
        assert self._layout == grid.currentLayout

        points = (grid._f - self._my_feq) * self._factor1

        return np.sum(points) * self._factor2


class PotentialEnergy_fresch:
    """
    TODO
    """

    def __init__(self, eta_grid: list, layout, constants):
        idx_r = layout.inv_dims_order[0]
        idx_q = layout.inv_dims_order[1]
        idx_z = layout.inv_dims_order[2]
        idx_v = layout.inv_dims_order[3]

        my_r = eta_grid[0][layout.starts[idx_r]:layout.ends[idx_r]]
        my_q = eta_grid[1][layout.starts[idx_q]:layout.ends[idx_q]]
        my_z = eta_grid[2][layout.starts[idx_z]:layout.ends[idx_z]]
        my_v = eta_grid[3][layout.starts[idx_v]:layout.ends[idx_v]]

        shape = [1, 1, 1, 1]
        shape[idx_r] = my_r.size
        shape[idx_q] = my_q.size
        shape[idx_z] = my_z.size
        self._shape_phi = shape

        r = eta_grid[0]
        q = eta_grid[1]
        z = eta_grid[2]
        v = eta_grid[3]

        dr = r[1:] - r[:-1]
        dv = v[1:] - v[:-1]

        shape = [1, 1, 1]
        shape[idx_r] = my_r.size
        self._myn0 = np.empty(shape)
        my_n0 = [n0(r, constants.CN0, constants.kN0, constants.deltaRN0, constants.rp)
                 for r in my_r]
        self._myn0.flat = my_n0[:]

        drMult = np.array(
            [dr[0] * 0.5, *((dr[1:] + dr[:-1]) * 0.5), dr[-1] * 0.5])
        dvMult = np.array(
            [dv[0] * 0.5, *((dv[1:] + dv[:-1]) * 0.5), dv[-1] * 0.5])

        mydrMult = drMult[layout.starts[idx_r]:layout.ends[idx_r]]
        mydvMult = dvMult[layout.starts[idx_v]:layout.ends[idx_v]]

        shape = [1, 1, 1, 1]
        shape[idx_r] = mydrMult.size
        shape[idx_v] = mydvMult.size

        self._factor1 = np.empty(shape)
        if (idx_r < idx_v):
            self._factor1.flat = (
                (mydrMult * my_r)[:, None] * (mydvMult)[None, :]).flat
        else:
            self._factor1.flat = (
                (mydrMult * my_r)[None, :] * (mydvMult)[:, None]).flat

        shape = [1, 1, 1]
        shape[idx_r] = mydrMult.size
        self._factor12 = np.empty(shape)
        self._factor12.flat = mydrMult * my_r

        self._layout = layout.name

        dq = q[2] - q[1]
        assert dq * eta_grid[1].size - 2 * np.pi < 1e-7

        dz = z[2] - z[1]
        assert dq > 0
        assert dz > 0

        self._factor2 = 0.5 * dq * dz

    def getPE(self, f, phi):
        """
        TODO
        """
        assert self._layout == f.currentLayout
        phi_grid = np.empty(self._shape_phi)
        phi_grid.flat = np.real(phi._f).flat

        points = np.real(f._f) * phi_grid * self._factor1
        points2 = -self._myn0 * np.real(phi._f) * self._factor12

        return (np.sum(points2)+np.sum(points)) * self._factor2
