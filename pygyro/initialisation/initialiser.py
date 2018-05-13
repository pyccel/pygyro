from math import pi
import numpy as np

from . import constants

def initF(r,theta,z,vPar,m = constants.m,n = constants.n):
    return fEq(r,vPar)*(1+constants.eps*perturbation(r,theta,z,m,n))

def perturbation(r,theta,z,m = constants.m,n = constants.n):
    return np.exp(-np.square(r-constants.rp)/constants.deltaR)*np.cos(constants.m*theta+constants.n*z/constants.R0)

def fEq(r,vPar):
    return n0(r)*np.exp(-0.5*vPar*vPar/Ti(r))/np.sqrt(2*pi*Ti(r))

def n0(r):
    return constants.CN0*np.exp(-constants.kN0*constants.deltaRN0*np.tanh((r-constants.rp)/constants.deltaRN0))

def Ti(r):
    return constants.CTi*np.exp(-constants.kTi*constants.deltaRTi*np.tanh((r-constants.rp)/constants.deltaRTi))

def Te(r):
    return constants.CTe*np.exp(-constants.kTe*constants.deltaRTe*np.tanh((r-constants.rp)/constants.deltaRTe))

def initialise_flux_surface(grid,m = constants.m,n = constants.n):
    for i,r in grid.getCoords(0):
        for j,v in grid.getCoords(1):
            # Get surface
            FluxSurface = grid.get2DSlice([i,j])
            # Get coordinate values
            theta = grid.getCoordVals(2)
            z = grid.getCoordVals(3)
            
            # transpose theta to use ufuncs
            theta = theta.reshape(theta.size,1)
            FluxSurface[:]=initF(r,theta,z,v,m,n)

def initialise_poloidal(grid,m = constants.m,n = constants.n):
    for i,v in grid.getCoords(0):
        for j,z in grid.getCoords(1):
            # Get surface
            PoloidalSurface = grid.get2DSlice([i,j])
            # Get coordinate values
            theta = grid.getCoordVals(2)
            r = grid.getCoordVals(3)
            
            # transpose theta to use ufuncs
            theta = theta.reshape(theta.size,1)
            PoloidalSurface[:]=initF(r,theta,z,v,m,n)

def initialise_v_parallel(grid,m = constants.m,n = constants.n):
    for i,r in grid.getCoords(0):
        for j,z in grid.getCoords(1):
            # Get surface
            Surface = grid.get2DSlice([i,j])
            # Get coordinate values
            theta = grid.getCoordVals(2)
            v = grid.getCoordVals(3)
            
            # transpose theta to use ufuncs
            theta = theta.reshape(theta.size,1)
            Surface[:]=initF(r,theta,z,v,m,n)
