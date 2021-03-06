from numpy import exp, sqrt, pi, tanh, cos

#pythran export n0(float64,float64,float64,float64,float64)
#pythran export n0(float64[:],float64,float64,float64,float64)
def n0(r,CN0,kN0,deltaRN0,rp):
    return CN0*exp(-kN0*deltaRN0*tanh((r-rp)/deltaRN0))

def ti(r,Cti,kti,deltaRti,rp):
    return Cti*exp(-kti*deltaRti*tanh((r-rp)/deltaRti))

def perturbation(r,theta,z,m,n,rp,deltaR,R0):
    return exp(-(r-rp)**2/deltaR)*cos(m*theta+n*z/R0)

#pythran export f_eq(float64,float64,float64,float64,float64,float64,float64,float64,float64)
def f_eq(r,vPar,CN0,kN0,deltaRN0,rp,Cti,kti,deltaRti):
    return CN0*exp(-kN0*deltaRN0*tanh((r-rp)/deltaRN0))*exp(-0.5*vPar*vPar/ti(r,Cti,kti,deltaRti,rp))/sqrt(2.0*pi*ti(r,Cti,kti,deltaRti,rp))

#pythran export n0deriv_normalised(float64,float64,float64,float64)
#pythran export n0deriv_normalised(float64[:],float64,float64,float64)
def n0deriv_normalised(r,kN0,rp,deltaRN0):
    return -kN0 * (1 - tanh( (r - rp ) / deltaRN0 )**2)

#pythran export te(float64,float64,float64,float64,float64)
#pythran export te(float64[:],float64,float64,float64,float64)
def te(r,Cte,kte,deltaRte,rp):
    return Cte*exp(-kte*deltaRte*tanh((r-rp)/deltaRte))

#pythran export init_f(float64,float64,float64,float64,int,int,float64,float64,float64,float64,float64,float64,float64,float64,float64,float64)
def init_f(r,theta,z,vPar,m,n, eps,CN0,kN0,deltaRN0,rp,Cti,kti,deltaRti,deltaR,R0):
    return f_eq(r,vPar,CN0,kN0,deltaRN0,rp,Cti,kti,deltaRti)*(1+eps*perturbation(r,theta,z,m,n,rp,deltaR,R0))

#pythran export init_f_flux(float64[:,:],float64,float64[:],float64[:],float64,int,int,float64,float64,float64,float64,float64,float64,float64,float64,float64,float64)
def init_f_flux(surface,r,theta,zVec,vPar,m,n, eps,CN0,kN0,deltaRN0,rp,Cti,kti,deltaRti,deltaR,R0):
    for i,q in enumerate(theta):
        for j,z in enumerate(zVec):
            surface[i,j]=f_eq(r,vPar,CN0,kN0,deltaRN0,rp,Cti,kti,deltaRti)*(1+eps*perturbation(r,q,z,m,n,rp,deltaR,R0))

#pythran export init_f_pol(float64[:,:],float64[:],float64[:],float64,float64,int,int,float64,float64,float64,float64,float64,float64,float64,float64,float64,float64)
def init_f_pol(surface,rVec,theta,z,vPar,m,n, eps,CN0,kN0,deltaRN0,rp,Cti,kti,deltaRti,deltaR,R0):
    for i,q in enumerate(theta):
        for j,r in enumerate(rVec):
            surface[i,j]=f_eq(r,vPar,CN0,kN0,deltaRN0,rp,Cti,kti,deltaRti)*(1+eps*perturbation(r,q,z,m,n,rp,deltaR,R0))

#pythran export init_f_vpar(float64[:,:],float64,float64[:],float64,float64[:],int,int,float64,float64,float64,float64,float64,float64,float64,float64,float64,float64)
def init_f_vpar(surface,r,theta,z,vPar,m,n, eps,CN0,kN0,deltaRN0,rp,Cti,kti,deltaRti,deltaR,R0):
    for i,q in enumerate(theta):
        for j,v in enumerate(vPar):
            surface[i,j]=f_eq(r,v,CN0,kN0,deltaRN0,rp,Cti,kti,deltaRti)*(1+eps*perturbation(r,q,z,m,n,rp,deltaR,R0))

#pythran export feq_vector(float64[:,:],float64[:],float64[:],float64,float64,float64,float64,float64,float64,float64)
def feq_vector(surface,r_vec,vPar,CN0,kN0,deltaRN0,rp,Cti,kti,deltaRti):
    for i,r in enumerate(r_vec):
        for j,v in enumerate(vPar):
            surface[i,j]=f_eq(r,v,CN0,kN0,deltaRN0,rp,Cti,kti,deltaRti)

