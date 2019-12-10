import numpy as np

from pygyro.model.grid                      import Grid
from pygyro.model.layout                    import Layout

class KineticEnergy:
    def __init__(self, eta_grid: list, layout: Layout):
        idx_r = layout.inv_dims_order[0]
        idx_v = layout.inv_dims_order[3]
        my_r = eta_grid[0][layout.starts[idx_r]:layout.ends[idx_r]]
        my_v = eta_grid[3][layout.starts[idx_v]:layout.ends[idx_v]]
        r = eta_grid[0]
        q = eta_grid[1]
        z = eta_grid[2]
        v = eta_grid[3]
        dr = r[1:]-r[:-1]
        dv = v[1:]-v[:-1]
        drMult = np.array([dr[0]*0.5, *((dr[1:]+dr[:-1])*0.5), dr[-1]*0.5])
        dvMult = np.array([dv[0]*0.5, *((dv[1:]+dv[:-1])*0.5), dv[-1]*0.5])
        mydrMult = drMult[layout.starts[idx_r]:layout.ends[idx_r]]
        mydvMult = dvMult[layout.starts[idx_v]:layout.ends[idx_v]]

        shape = [1,1,1,1]
        shape[idx_r] = mydrMult.size
        shape[idx_v] = mydvMult.size

        self._factor1 = np.empty(shape)
        if (idx_r<idx_v):
            self._factor1.flat = ((mydrMult*my_r)[:,None] * (mydvMult*my_v**2)[None,:]).flat
        else:
            self._factor1.flat = ((mydrMult*my_r)[None,:] * (mydvMult*my_v**2)[:,None]).flat

        self._layout = layout.name

        dq = q[2]-q[1]
        assert(dq*eta_grid[1].size-2*np.pi<1e-7)
        dz = z[2]-z[1]
        assert(dq>0)
        assert(dz>0)

        self._factor2 = 0.5*dq*dz

    def getKE(self, grid: Grid):
        assert(self._layout == grid.currentLayout)
        points = np.real(grid._f)*self._factor1

        return np.sum(points)*self._factor2
