import numpy as np

from pygyro.model.grid                      import Grid
from pygyro.model.layout                    import Layout

class l2:
    def __init__(self, eta_grid: list, layout: Layout):
        assert(layout.dims_order==(0,2,1))
        my_r = eta_grid[0][layout.starts[0]:layout.ends[0]]
        q = eta_grid[1]
        z = eta_grid[2]
        dr = eta_grid[0][1:]-eta_grid[0][:-1]
        drMult = np.array([dr[0]*0.5, *((dr[1:]+dr[:-1])*0.5), dr[-1]*0.5])
        mydrMult = drMult[layout.starts[0]:layout.ends[0]]
        
        self._factor1 = (mydrMult*my_r)[:,None,None]
        
        dq = q[2]-q[1]
        dz = z[2]-z[1]
        
        self._factor2 = dq*dz
    
    def l2NormSquared(self, phi: Grid):
        points = np.real(phi._f*phi._f.conj())*self._factor1
        # Edge points for theta
        points[:,:,0]*=0.5
        points[:,:,-1]*=0.5
        # Edge points for z
        points[:,0,:]*=0.5
        points[:,-1,:]*=0.5
        
        return np.sum(points)*self._factor2
