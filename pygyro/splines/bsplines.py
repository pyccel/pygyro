from pyccel.decorators import types

@types('double','int','double[:]','int','double[:]')
def eval_spline_1d_scalar(x,der,knots,degree,coeffs):
    span  =  find_span( knots, degree, x )
    
    from numpy      import empty
    basis  = empty( degree+1, dtype=float )
    if (der==0):
        basis_funs( knots, degree, x, span, basis )
    elif (der==1):
        basis_funs_1st_der( knots, degree, x, span, basis )
    
    y=0.0
    for j in range(degree+1):
        y+=coeffs[span-degree+j]*basis[j]
    return y

@types('double[:]','int','double[:]','int','double[:]','double[:]')
def eval_spline_1d_vector(x,der,knots,degree,coeffs,y):
    from numpy      import empty
    if (der==0):
        for i in range(len(x)):
            span  =  find_span( knots, degree, x[i] )
            basis  = empty( degree+1, dtype=float )
            basis_funs( knots, degree, x[i], span, basis )
            
            y[i]=0.0
            for j in range(degree+1):
                y[i]+=coeffs[span-degree+j]*basis[j]
    elif (der==1):
        for i in range(len(x)):
            span  =  find_span( knots, degree, x[i] )
            basis  = empty( degree+1, dtype=float )
            basis_funs( knots, degree, x[i], span, basis )
            basis_funs_1st_der( knots, degree, x[i], span, basis )
            
            y[i]=0.0
            for j in range(degree+1):
                y[i]+=coeffs[span-degree+j]*basis[j]
    return y
