from math import sqrt, exp, pi, tanh, cos

from . import Constants

def initF(r,theta,z,vPar,m = Constants.m,n = Constants.n):
    return fEq(r,vPar)*(1+Constants.eps*perturbation(r,theta,z,m,n))

def perturbation(r,theta,z,m = Constants.m,n = Constants.n):
    return exp(-(r-Constants.rp)**2/Constants.deltaR)*cos(Constants.m*theta+Constants.n*z/Constants.R0)

def fEq(r,vPar):
    return n0(r)*exp(-0.5*vPar*vPar/Ti(r))/sqrt(2*pi*Ti(r))

def n0(r):
    return Constants.CN0*exp(-Constants.kN0*Constants.deltaRN0*tanh((r-Constants.rp)/Constants.deltaRN0))

def Ti(r):
    return Constants.CTi*exp(-Constants.kTi*Constants.deltaRTi*tanh((r-Constants.rp)/Constants.deltaRTi))

def Te(r):
    return Constants.CTe*exp(-Constants.kTe*Constants.deltaRTe*tanh((r-Constants.rp)/Constants.deltaRTe))

def Initialiser(f,rVals,qVals,zVals,vVals,m = None,n = None):
    if (m==None):
        m=Constants.m
    if (n==None):
        n=Constants.n
    for i in range(0,len(vVals)):
        for j in range(0,len(qVals)):
            for k in range(0,len(rVals)):
                for l in range(0,len(zVals)):
                    f[i,j,k,l]=initF(rVals[k],qVals[j],zVals[l],vVals[i])

def getPerturbation(f,rVals,qVals,zVals,vVals,m = None,n = None):
    if (m==None):
        m=Constants.m
    if (n==None):
        n=Constants.n
    for i in range(0,len(vVals)):
        for j in range(0,len(qVals)):
            for k in range(0,len(rVals)):
                for l in range(0,len(zVals)):
                    f[i,j,k,l]=perturbation(rVals[k],qVals[j],zVals[l])

def getEquilibrium(f,rVals,qVals,zVals,vVals,m = None,n = None):
    if (m==None):
        m=Constants.m
    if (n==None):
        n=Constants.n
    for i in range(0,len(vVals)):
        for j in range(0,len(qVals)):
            for k in range(0,len(rVals)):
                for l in range(0,len(zVals)):
                    f[i,j,k,l]=fEq(rVals[k],vVals[i])
