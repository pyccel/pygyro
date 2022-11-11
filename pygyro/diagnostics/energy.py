import numpy as np

from pygyro.model.grid import Grid
from pygyro.model.layout import Layout
from ..arakawa.utilities import compute_int_f, compute_int_f_squared, get_potential_energy
from pygyro.initialisation.initialiser_funcs import f_eq


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

        print(f'index r : {idx_r}')
        print(f'index q : {idx_q}')
        print(f'index z : {idx_z}')
        print(f'index v : {idx_v}')

        my_r = eta_grid[0][layout.starts[idx_r]:layout.ends[idx_r]]
        my_q = eta_grid[1][layout.starts[idx_q]:layout.ends[idx_q]]
        my_z = eta_grid[2][layout.starts[idx_z]:layout.ends[idx_z]]
        my_v = eta_grid[3][layout.starts[idx_v]:layout.ends[idx_v]]

        print(f'local size of r : {my_r.size}')
        print(f'local size of q : {my_q.size}')
        print(f'local size of z : {my_z.size}')
        print(f'local size of v : {my_v.size}')

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
                (mydrMult * my_r)[:, None] * (mydvMult)[None, :]).flat
        else:
            self._factor1.flat = (
                (mydrMult * my_r)[None, :] * (mydvMult)[:, None]).flat

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

        # Create temporary buffer for storing phi values
        phi_temp = np.zeros((self.r_start_end[1] - self.r_start_end[0],
                            self.z_start_end[1] - self.z_start_end[0],
                            1,
                            self.v_start_end[1] - self.v_start_end[0]), dtype=float)
        
        phi_temp.flat = np.real(phi._f[:, self.z_start_end[0]: self.z_start_end[1], :].flat)

        print(f'f layout : {grid.currentLayout}')
        print(f'phi layout : {phi.currentLayout}')
        print(f'f shape : {np.shape(grid._f)}')
        print(f'f_eq shape : {np.shape(self._my_feq)}')
        print(f'phi shape : {np.shape(phi._f)}')

        points = (grid._f - self._my_feq) * phi_temp * self._factor1

        return np.sum(points) * self._factor2


# class PotentialEnergy:
#     """
#     TODO
#     """

#     def __init__(self, eta_grid: list):
#         self._r_grid = eta_grid[0]
#         self._theta_grid = eta_grid[1]
#         self._z_grid = eta_grid[2]
#         self._v_grid = eta_grid[3]

#         self._dr = self._r_grid[1] - self._r_grid[0]
#         self._dtheta = self._theta_grid[1] - self._theta_grid[0]

#         dv = self._v_grid[1:] - self._v_grid[:-1]
#         dz = self._z_grid[1:] - self._z_grid[:-1]
#         self._dvMult = np.append(dv[0] * 0.5,
#                                  np.append((dv + np.roll(dv, 1))[1:] * 0.5, dv[-1] * 0.5))
#         self._dzMult = np.append(dz[0] * 0.5,
#                                  np.append((dz + np.roll(dz, 1))[1:] * 0.5, dz[-1] * 0.5))

#     def getPE(self, f: Grid, phi: Grid):
#         """
#         TODO
#         """
#         # get list of global indices when wanting to save diagnostics
#         # global_inds_v = f.getGlobalIdxVals(0)
#         # global_inds_z = f.getGlobalIdxVals(1)

#         int_f = 0.
#         int_f_squared = 0.
#         energy = 0.

#         for index_v, _ in f.getCoords(0):  # v
#             for index_z, _ in f.getCoords(1):  # z

#                 # global_v = global_inds_v[index_v]
#                 # global_z = global_inds_z[index_z]

#                 # # Only compute for the first slice in z-direction save it
#                 # if global_z == 0:
#                 #     # Compute mass and l2-norm if v is in the middle of the velocity distribution
#                 #     if global_v == (f.eta_grid[3].size // 2):
#                 #         int_f = compute_int_f(f.get2DSlice(index_v, index_z), self._dtheta, self._dr,
#                 #                               self._r_grid, method='trapz')
#                 #         int_f_squared = compute_int_f_squared(f.get2DSlice(index_v, index_z), self._dtheta, self._dr,
#                 #                                               self._r_grid, method='trapz')

#                 energy += get_potential_energy(f.get2DSlice(index_v, index_z), phi.get2DSlice(index_z), self._dtheta, self._dr,
#                                             self._r_grid, method='trapz') * self._dzMult[index_z]

#             energy *= self._dvMult[index_v]

#         return energy * 0.5
