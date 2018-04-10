from mpi4py import MPI
from math import pi
import numpy as np

from ..           import splines as spl
from ..model.grid import Grid
from .initialiser import Initialiser

def RadialSetup(nr: int, ntheta: int, nz: int, nv: int, rMin: float,
                rMax: float, zMin: float, zMax: float, vMax: float, vMin: float = None,m=None,n=None):
    """
    Setup using radial topology can be initialised using the following arguments:
    
    Compulsory arguments:
    nr     -- number of points in the radial direction
    ntheta -- number of points in the tangential direction
    nz     -- number of points in the axial direction
    nv     -- number of velocities for v perpendicular
    rMin   -- minimum radius, a float
    rMax   -- maximum radius, a float
    zMin   -- minimum value in the z direction, a float
    zMax   -- maximum value in the z direction, a float
    vMax   -- maximum velocity, a float
    
    Optional arguments:
    vMin -- minimum velocity, a float. (default -vMax)
    
    >>> RadialSetup(rMin = float,rMax = float,zMin = float,float
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if (vMin==None):
        vMin=-vMax
    elif (not isinstance(vMin,float)):
        raise TypeError("Wrong type for vMin in Radial Setup")
    
    degree=3
    
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
    
    nrOverflow=nr%size
    nzOverflow=nz%size
    ranks=np.arange(0,size)
    rStarts=nr//size*ranks + np.minimum(ranks,nrOverflow)
    zStarts=nz//size*ranks + np.minimum(ranks,nzOverflow)
    rStarts=np.append(rStarts,len(rVals))
    zStarts=np.append(zStarts,len(zVals))
    rLen=nr//size+np.less(ranks,nrOverflow).astype(int)
    zLen=nz//size+np.less(ranks,nzOverflow).astype(int)
    rEnd=rStarts[rank]+rLen[rank]
    
    # ordering chosen to increase step size to improve cache-coherency
    f = np.empty((nv,ntheta,rLen[rank],nz),float,order='F')
    
    Initialiser(f,rVals[rStarts[rank]:rEnd],qVals,zVals,vVals,m,n)
    return Grid(rVals,rSpline,qVals,qSpline,zVals,zSpline,vVals,vSpline,rStarts,zStarts,f,"radial")


def BlockSetup(nr: int, ntheta: int, nz: int, nv: int, rMin: float,
                rMax: float, zMin: float, zMax: float, vMax: float, vMin: float = None,m=None,n=None):
    """
    Setup using topology split in the z direction can be initialised using the following arguments:
    
    Compulsory arguments:
    nr     -- number of points in the radial direction
    ntheta -- number of points in the tangential direction
    nz     -- number of points in the axial direction
    nv     -- number of velocities for v perpendicular
    rMin   -- minimum radius, a float
    rMax   -- maximum radius, a float
    zMin   -- minimum value in the z direction, a float
    zMax   -- maximum value in the z direction, a float
    vMax   -- maximum velocity, a float
    
    Optional arguments:
    vMin -- minimum velocity, a float. (default -vMax)
    
    >>> BlockSetup(rMin = float,rMax = float,zMin = float,float
    """
        
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if (vMin==None):
        vMin=-vMax
    elif (not isinstance(vMin,float)):
        raise TypeError("Wrong type for vMin in Radial Setup")
    
    degree=3
    
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
    
    nrOverflow=nr%size
    nzOverflow=nz%size
    ranks=np.arange(0,size)
    rStarts=nr//size*ranks+np.minimum(ranks,nrOverflow)
    zStarts=nz//size*ranks+np.minimum(ranks,nzOverflow)
    rStarts=np.append(rStarts,len(rVals))
    zStarts=np.append(zStarts,len(zVals))
    rLen=nr//size+np.less(ranks,nrOverflow).astype(int)
    zLen=nz//size+np.less(ranks,nzOverflow).astype(int)
    zEnd=zStarts[rank]+zLen[rank]
    
    # ordering chosen to increase step size to improve cache-coherency
    f = np.empty((nv,ntheta,nr,zLen[rank]),float,order='F')
    
    Initialiser(f,rVals,qVals,zVals[zStarts[rank]:zEnd],vVals,m,n)
    return Grid(rVals,rSpline,qVals,qSpline,zVals,zSpline,vVals,vSpline,rStarts,zStarts,f,"block")
