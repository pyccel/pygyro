import numpy as np

from pygyro.model.grid                      import Grid
from pygyro.model.layout                    import Layout

class l2:
    def __init__(self, eta_grid: list, layout: Layout):
        idx0 = layout.inv_dims_order[0]
        my_r = eta_grid[0][layout.starts[idx0]:layout.ends[idx0]]
        q = eta_grid[1]
        z = eta_grid[2]
        dr = eta_grid[0][1:]-eta_grid[0][:-1]
        drMult = np.array([dr[0]*0.5, *((dr[1:]+dr[:-1])*0.5), dr[-1]*0.5])
        mydrMult = drMult[layout.starts[idx0]:layout.ends[idx0]]
        
        shape = [1,1,1]
        shape[idx0] = mydrMult.size
        
        self._factor1 = np.empty(shape)
        self._factor1.flat = mydrMult*my_r
        
        self._layout = layout.name
        
        dq = q[2]-q[1]
        assert(dq*eta_grid[1].size-2*np.pi<1e-7)
        dz = z[2]-z[1]
        assert(dq>0)
        assert(dz>0)
        
        self._factor2 = dq*dz
    
    def l2NormSquared(self, phi: Grid):
        assert(self._layout == phi.currentLayout)
        points = np.real(phi._f*phi._f.conj())*self._factor1
        
        return np.sum(points)*self._factor2

class l1:
    def __init__(self, eta_grid: list, layout: Layout):
        idx0 = layout.inv_dims_order[0]
        my_r = eta_grid[0][layout.starts[idx0]:layout.ends[idx0]]
        q = eta_grid[1]
        z = eta_grid[2]
        dr = eta_grid[0][1:]-eta_grid[0][:-1]
        drMult = np.array([dr[0]*0.5, *((dr[1:]+dr[:-1])*0.5), dr[-1]*0.5])
        mydrMult = drMult[layout.starts[idx0]:layout.ends[idx0]]
        
        shape = [1,1,1]
        shape[idx0] = mydrMult.size
        
        self._factor1 = np.empty(shape)
        self._factor1.flat = mydrMult*my_r
        
        self._layout = layout.name
        
        dq = q[2]-q[1]
        assert(dq*eta_grid[1].size-2*np.pi<1e-7)
        dz = z[2]-z[1]
        assert(dq>0)
        assert(dz>0)
        
        self._factor2 = dq*dz
    
    def l1Norm(self, phi: Grid):
        assert(self._layout == phi.currentLayout)
        points = np.real(phi._f)*self._factor1
        
        return np.sum(points)*self._factor2
