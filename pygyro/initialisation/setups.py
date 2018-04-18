from math import pi
import numpy as np

from ..           import splines as spl
from ..model.grid import Grid, Layout
from .initialiser import initialise
from .            import constants

def setupCylindricalGrid(nr: int, ntheta: int, nz: int, nv: int, layout: Layout, **kwargs):
    """
    Setup using radial topology can be initialised using the following arguments:
    
    Compulsory arguments:
    nr     -- number of points in the radial direction
    ntheta -- number of points in the tangential direction
    nz     -- number of points in the axial direction
    nv     -- number of velocities for v perpendicular
    layout -- parallel distribution configuration
    
    Optional arguments:
    rMin   -- minimum radius, a float. (default constants.rMin)
    rMax   -- maximum radius, a float. (default constants.rMax)
    zMin   -- minimum value in the z direction, a float. (default constants.zMin)
    zMax   -- maximum value in the z direction, a float. (default constants.zMax)
    vMax   -- maximum velocity, a float. (default constants.vMax)
    vMin   -- minimum velocity, a float. (default -vMax)
    degree -- degree of splines. (default 3)
    
    >>> setupGrid(256,512,32,128,Layout.FIELD_ALIGNED)
    """
    rMin=kwargs.pop('rMin',constants.rMin)
    rMax=kwargs.pop('rMin',constants.rMax)
    zMin=kwargs.pop('rMin',constants.zMin)
    zMax=kwargs.pop('rMin',constants.zMax)
    vMax=kwargs.pop('rMin',constants.vMax)
    vMin=kwargs.pop('rMin',-vMax)
    m=kwargs.pop('m',constants.m)
    n=kwargs.pop('n',constants.n)
    degree=kwargs.pop('degree',3)
    nProcEta1=kwargs.pop('nProcEta1',1)
    nProcEta3=kwargs.pop('nProcEta3',1)
    nProcEta4=kwargs.pop('nProcEta4',1)
    
    # get spline knots
    rKnots = spl.make_knots(np.linspace(rMin,rMax,nr-degree+1),degree,False)
    qKnots = spl.make_knots(np.linspace(0,2*pi,ntheta+1),degree,True)
    zKnots = spl.make_knots(np.linspace(zMin,zMax,nz+1),degree,True)
    vKnots = spl.make_knots(np.linspace(vMin,vMax,nv-degree+1),degree,False)
    
    # make splines
    rSpline = spl.BSplines(rKnots,degree,False)
    qSpline = spl.BSplines(qKnots,degree,True)
    zSpline = spl.BSplines(zKnots,degree,True)
    vSpline = spl.BSplines(vKnots,degree,False)
    
    # get greville points
    rVals = rSpline.greville
    qVals = qSpline.greville
    zVals = zSpline.greville
    vVals = vSpline.greville
    
    grid=Grid(rVals,qVals,zVals,vVals,layout,nProcEta1=nProcEta1,nProcEta3=nProcEta3,nProcEta4=nProcEta4)
    initialise(grid,m,n)
    return grid
