from pyccel.decorators import types, pure

@pure
@types('double','double','double','double','double')
def n0(r,CN0,kN0,deltaRN0,rp):
    from numpy import exp, tanh
    return CN0*exp(-kN0*deltaRN0*tanh((r-rp)/deltaRN0))

@pure
@types('double','double','double','double','double')
def ti(r,Cti,kti,deltaRti,rp):
    from numpy import exp, tanh
    return Cti*exp(-kti*deltaRti*tanh((r-rp)/deltaRti))

@pure
@types('double','double','double','int','int','double','double','double')
def perturbation(r,theta,z,m,n,rp,deltaR,R0):
    from numpy import exp, cos
    return exp(-(r-rp)**2/deltaR)*cos(m*theta+n*z/R0)

@pure
@types('double','double','double','double','double','double','double','double','double')
def f_eq(r,vPar,CN0,kN0,deltaRN0,rp,Cti,kti,deltaRti):
    from numpy import exp, sqrt, pi, real
    return n0(r,CN0,kN0,deltaRN0,rp)*exp(-0.5*vPar*vPar/ti(r,Cti,kti,deltaRti,rp))/real(sqrt(2.0*pi*ti(r,Cti,kti,deltaRti,rp)))

@pure
@types('double','double','double','double')
def n0deriv_normalised(r,kN0,rp,deltaRN0):
    from numpy import tanh
    return -kN0 * (1 - tanh( (r - rp ) / deltaRN0 )**2)

@pure
@types('double','double','double','double','double')
def te(r,Cte,kte,deltaRte,rp):
    from numpy import exp, tanh
    return Cte*exp(-kte*deltaRte*tanh((r-rp)/deltaRte))

@pure
@types('double','double','double','double','int','int','double','double',
        'double','double','double','double','double','double','double','double')
def init_f(r,theta,z,vPar,m,n, eps,CN0,kN0,deltaRN0,rp,Cti,kti,deltaRti,deltaR,R0):
    return f_eq(r,vPar,CN0,kN0,deltaRN0,rp,Cti,kti,deltaRti)*(1+eps*perturbation(r,theta,z,m,n,rp,deltaR,R0))

@pure
@types('double[:,:]','double','double[:]','double[:]','double','int','int','double',
        'double','double','double','double','double','double','double','double','double')
def init_f_flux(surface,r,theta,zVec,vPar,m,n, eps,CN0,kN0,deltaRN0,rp,Cti,kti,deltaRti,deltaR,R0):
    for i,q in enumerate(theta):
        for j,z in enumerate(zVec):
            surface[i,j]=f_eq(r,vPar,CN0,kN0,deltaRN0,rp,Cti,kti,deltaRti)*(1+eps*perturbation(r,q,z,m,n,rp,deltaR,R0))

@pure
@types('double[:,:]','double[:]','double[:]','double','double','int','int','double',
        'double','double','double','double','double','double','double','double','double')
def init_f_pol(surface,rVec,theta,z,vPar,m,n, eps,CN0,kN0,deltaRN0,rp,Cti,kti,deltaRti,deltaR,R0):
    for i,q in enumerate(theta):
        for j,r in enumerate(rVec):
            surface[i,j]=f_eq(r,vPar,CN0,kN0,deltaRN0,rp,Cti,kti,deltaRti)*(1+eps*perturbation(r,q,z,m,n,rp,deltaR,R0))

@pure
@types('double[:,:]','double','double[:]','double','double[:]','int','int','double','double',
        'double','double','double','double','double','double','double','double')
def init_f_vpar(surface,r,theta,z,vPar,m,n, eps,CN0,kN0,deltaRN0,rp,Cti,kti,deltaRti,deltaR,R0):
    for i,q in enumerate(theta):
        for j,v in enumerate(vPar):
            surface[i,j]=f_eq(r,v,CN0,kN0,deltaRN0,rp,Cti,kti,deltaRti)*(1+eps*perturbation(r,q,z,m,n,rp,deltaR,R0))

@pure
@types('double[:,:]','double[:]','double[:]','double','double','double','double','double','double','double')
def feq_vector(surface,r_vec,vPar,CN0,kN0,deltaRN0,rp,Cti,kti,deltaRti):
    for i,r in enumerate(r_vec):
        for j,v in enumerate(vPar):
            surface[i,j]=f_eq(r,v,CN0,kN0,deltaRN0,rp,Cti,kti,deltaRti)

