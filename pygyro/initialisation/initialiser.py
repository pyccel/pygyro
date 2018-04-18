from math import sqrt, exp, pi, tanh, cos

from . import constants

def initF(r,theta,z,vPar,m = constants.m,n = constants.n):
    return fEq(r,vPar)*(1+constants.eps*perturbation(r,theta,z,m,n))

def perturbation(r,theta,z,m = constants.m,n = constants.n):
    return exp(-(r-constants.rp)**2/constants.deltaR)*cos(constants.m*theta+constants.n*z/constants.R0)

def fEq(r,vPar):
    return n0(r)*exp(-0.5*vPar*vPar/Ti(r))/sqrt(2*pi*Ti(r))

def n0(r):
    return constants.CN0*exp(-constants.kN0*constants.deltaRN0*tanh((r-constants.rp)/constants.deltaRN0))

def Ti(r):
    return constants.CTi*exp(-constants.kTi*constants.deltaRTi*tanh((r-constants.rp)/constants.deltaRTi))

def Te(r):
    return constants.CTe*exp(-constants.kTe*constants.deltaRTe*tanh((r-constants.rp)/constants.deltaRTe))

def initialise(grid,m = constants.m,n = constants.n):
    for i,r in grid.getEta1Coords():
        for j,z in grid.getEta3Coords():
            for k,v in grid.getEta4Coords():
                for l,theta in grid.getEta2Coords():
                    grid.f[i,j,k,l]=initF(r,theta,z,v,m,n)

def getPerturbation(grid,m = constants.m,n = constants.n):
    for i,r in grid.getEta1Coords():
        for j,z in grid.getEta3Coords():
            for k,v in grid.getEta4Coords():
                for l,theta in grid.getEta2Coords():
                    grid.f[i,j,k,l]=perturbation(r,theta,z,m,n)

def getEquilibrium(grid):
    for i,r in grid.getEta1Coords():
        for j,z in grid.getEta3Coords():
            for k,v in grid.getEta4Coords():
                for l,theta in grid.getEta2Coords():
                    grid.f[i,j,k,l]=fEq(r,v)
