from pyccel.decorators          import types
from .mod_initialiser_funcs     import fEq,perturbation

#~ def types(*args):
    #~ def id(f):
        #~ return f
    #~ return id

@types('double','double','double','double','int','int','double','double','double','double','double','double','double','double','double','double')
def init_f(r,theta,z,vPar,m,n, eps,CN0,kN0,deltaRN0,rp,CTi,kTi,deltaRTi,deltaR,R0):
    return fEq(r,vPar,CN0,kN0,deltaRN0,rp,CTi,kTi,deltaRTi)*(1+eps*perturbation(r,theta,z,m,n,rp,deltaR,R0))

@types('double[:,:]','double','double[:]','double[:]','double','int','int','double','double','double','double','double','double','double','double','double','double')
def init_f_flux(surface,r,theta,zVec,vPar,m,n, eps,CN0,kN0,deltaRN0,rp,CTi,kTi,deltaRTi,deltaR,R0):
    for i,q in enumerate(theta):
        for j,z in enumerate(zVec):
            surface[i,j]=fEq(r,vPar,CN0,kN0,deltaRN0,rp,CTi,kTi,deltaRTi)*(1+eps*perturbation(r,q,z,m,n,rp,deltaR,R0))

@types('double[:,:]','double[:]','double[:]','double','double','int','int','double','double','double','double','double','double','double','double','double','double')
def init_f_pol(surface,rVec,theta,z,vPar,m,n, eps,CN0,kN0,deltaRN0,rp,CTi,kTi,deltaRTi,deltaR,R0):
    for i,q in enumerate(theta):
        for j,r in enumerate(rVec):
            surface[i,j]=fEq(r,vPar,CN0,kN0,deltaRN0,rp,CTi,kTi,deltaRTi)*(1+eps*perturbation(r,q,z,m,n,rp,deltaR,R0))

@types('double[:,:]','double','double[:]','double','double[:]','int','int','double','double','double','double','double','double','double','double','double','double')
def init_f_vpar(surface,r,theta,z,vPar,m,n, eps,CN0,kN0,deltaRN0,rp,CTi,kTi,deltaRTi,deltaR,R0):
    for i,q in enumerate(theta):
        for j,v in enumerate(vPar):
            surface[i,j]=fEq(r,v,CN0,kN0,deltaRN0,rp,CTi,kTi,deltaRTi)*(1+eps*perturbation(r,q,z,m,n,rp,deltaR,R0))

@types('double[:,:]','double[:]','double[:]','double','double','double','double','double','double','double')
def fEq_vector(surface,r_vec,vPar,CN0,kN0,deltaRN0,rp,CTi,kTi,deltaRTi):
    for i,r in enumerate(r_vec):
        for j,v in enumerate(vPar):
            surface[i,j]=fEq(r,v,CN0,kN0,deltaRN0,rp,CTi,kTi,deltaRTi)

