from numpy import exp, sqrt, pi, tanh, cos

def n0(r,CN0,kN0,deltaRN0,rp):
    return CN0*exp(-kN0*deltaRN0*tanh((r-rp)/deltaRN0))

def Ti(r,CTi,kTi,deltaRTi,rp):
    return CTi*exp(-kTi*deltaRTi*tanh((r-rp)/deltaRTi))

def perturbation(r,theta,z,m,n,rp,deltaR,R0):
    return exp(-(r-rp)**2/deltaR)*cos(m*theta+n*z/R0)

def fEq(r,vPar,CN0,kN0,deltaRN0,rp,CTi,kTi,deltaRTi):
    return CN0*exp(-kN0*deltaRN0*tanh((r-rp)/deltaRN0))*exp(-0.5*vPar*vPar/Ti(r,CTi,kTi,deltaRTi,rp))/sqrt(2.0*pi*Ti(r,CTi,kTi,deltaRTi,rp))

def n0derivNormalised(r,kN0,rp,deltaRN0):
    return -kN0 * (1 - tanh( (r - rp ) / deltaRN0 )**2)

def Te(r,CTe,kTe,deltaRTe,rp):
    return CTe*exp(-kTe*deltaRTe*tanh((r-rp)/deltaRTe))


#pythran export init_f(float64,float64,float64,float64,int,int,float64,float64,float64,float64,float64,float64,float64,float64,float64,float64)
def init_f(r,theta,z,vPar,m,n, eps,CN0,kN0,deltaRN0,rp,CTi,kTi,deltaRTi,deltaR,R0):
    return fEq(r,vPar,CN0,kN0,deltaRN0,rp,CTi,kTi,deltaRTi)*(1+eps*perturbation(r,theta,z,m,n,rp,deltaR,R0))

#pythran export init_f_flux(float64[:,:],float64,float64[:],float64[:],float64,int,int,float64,float64,float64,float64,float64,float64,float64,float64,float64,float64)
def init_f_flux(surface,r,theta,zVec,vPar,m,n, eps,CN0,kN0,deltaRN0,rp,CTi,kTi,deltaRTi,deltaR,R0):
    for i,q in enumerate(theta):
        for j,z in enumerate(zVec):
            surface[i,j]=fEq(r,vPar,CN0,kN0,deltaRN0,rp,CTi,kTi,deltaRTi)*(1+eps*perturbation(r,q,z,m,n,rp,deltaR,R0))

#pythran export init_f_pol(float64[:,:],float64[:],float64[:],float64,float64,int,int,float64,float64,float64,float64,float64,float64,float64,float64,float64,float64)
def init_f_pol(surface,rVec,theta,z,vPar,m,n, eps,CN0,kN0,deltaRN0,rp,CTi,kTi,deltaRTi,deltaR,R0):
    for i,q in enumerate(theta):
        for j,r in enumerate(rVec):
            surface[i,j]=fEq(r,vPar,CN0,kN0,deltaRN0,rp,CTi,kTi,deltaRTi)*(1+eps*perturbation(r,q,z,m,n,rp,deltaR,R0))

#pythran export init_f_vpar(float64[:,:],float64,float64[:],float64,float64[:],int,int,float64,float64,float64,float64,float64,float64,float64,float64,float64,float64)
def init_f_vpar(surface,r,theta,z,vPar,m,n, eps,CN0,kN0,deltaRN0,rp,CTi,kTi,deltaRTi,deltaR,R0):
    for i,q in enumerate(theta):
        for j,v in enumerate(vPar):
            surface[i,j]=fEq(r,v,CN0,kN0,deltaRN0,rp,CTi,kTi,deltaRTi)*(1+eps*perturbation(r,q,z,m,n,rp,deltaR,R0))

#pythran export feq_vector(float64[:,:],float64[:],float64[:],float64,float64,float64,float64,float64,float64,float64)
def feq_vector(surface,r_vec,vPar,CN0,kN0,deltaRN0,rp,CTi,kTi,deltaRTi):
    for i,r in enumerate(r_vec):
        for j,v in enumerate(vPar):
            surface[i,j]=fEq(r,v,CN0,kN0,deltaRN0,rp,CTi,kTi,deltaRTi)

