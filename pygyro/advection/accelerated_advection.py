
def poloidalStep( self, f: np.ndarray, dt: float, phi: Spline2D, v: float ):
    """
    Carry out an advection step for the poloidal advection

    Parameters
    ----------
    f: array_like
        The current value of the function at the nodes.
        The result will be stored here
    
    dt: float
        Time-step
    
    phi: Spline2D
        Advection parameter d_tf + {phi,f}=0
    
    r: float
        The parallel velocity coordinate
    
    """
    assert(f.shape==self._nPoints)
    self._interpolator.compute_interpolant(f,self._spline)
    
    multFactor = dt/constants.B0
    
    drPhi_0 = phi.eval(*self._points,0,1)/self._points[1]
    dthetaPhi_0 = phi.eval(*self._points,1,0)/self._points[1]
    
    # Step one of Heun method
    # x' = x^n + f(x^n)
    endPts_k1 = ( self._shapedQ   -     drPhi_0*multFactor,
                 self._points[1] + dthetaPhi_0*multFactor )
    
    drPhi_k = np.empty_like(drPhi_0)
    dthetaPhi_k = np.empty_like(dthetaPhi_0)
    
    multFactor*=0.5
    
    while (True):
        
        for i in range(self._nPoints[0]):
            for j in range(self._nPoints[1]):
                # Handle theta boundary conditions
                while (endPts_k1[0][i,j]<0):
                    endPts_k1[0][i,j]+=2*pi
                while (endPts_k1[0][i,j]>2*pi):
                    endPts_k1[0][i,j]-=2*pi
                
                if (not (endPts_k1[1][i,j]<self._points[1][0] or 
                         endPts_k1[1][i,j]>self._points[1][-1])):
                    # Add the new value of phi to the derivatives
                    # x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
                    #                               ^^^^^^^^^^^^^^^
                    drPhi_k[i,j]     = phi.eval(endPts_k1[0][i,j],endPts_k1[1][i,j],0,1)/endPts_k1[1][i,j]
                    dthetaPhi_k[i,j] = phi.eval(endPts_k1[0][i,j],endPts_k1[1][i,j],1,0)/endPts_k1[1][i,j]
                else:
                    drPhi_k[i,j]     = 0.0
                    dthetaPhi_k[i,j] = 0.0
        
        if (self._explicit):
            # Step two of Heun method
            # x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
            endPts_k2 = ( np.mod(self._shapedQ   - (drPhi_0     + drPhi_k)*multFactor,2*pi),
                          self._points[1] + (dthetaPhi_0 + dthetaPhi_k)*multFactor )
            break
        else:
            # Step two of Heun method
            # x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )

            # Clipping is one method of avoiding infinite loops due to boundary conditions
            # Using the splines to extrapolate is not sufficient
            endPts_k2 = ( np.mod(self._shapedQ   - (drPhi_0     + drPhi_k)*multFactor,2*pi),
                          np.clip(self._points[1] + (dthetaPhi_0 + dthetaPhi_k)*multFactor,
                                  self._points[1][0], self._points[1][-1]) )
            
            norm = max(np.linalg.norm((endPts_k2[0]-endPts_k1[0]).flatten(),np.inf),
                       np.linalg.norm((endPts_k2[1]-endPts_k1[1]).flatten(),np.inf))
            if (norm<self._TOL):
                break
            endPts_k1=endPts_k2
    
    # Find value at the determined point
    for i,theta in enumerate(self._points[0]):
        for j,r in enumerate(self._points[1]):
            f[i,j]=self.evalFunc(endPts_k2[0][i,j],endPts_k2[1][i,j],v)
