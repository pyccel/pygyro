from pyccel.decorators  import types

#==============================================================================
@types('double[:]','int','double')
def find_span( knots, degree, x ):
    """
    Determine the knot span index at location x, given the
    B-Splines' knot sequence and polynomial degree. See
    Algorithm A2.1 in [1].

    For a degree p, the knot span index i identifies the
    indices [i-p:i] of all p+1 non-zero basis functions at a
    given location x.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    x : float
        Location of interest.

    Returns
    -------
    span : int
        Knot span index.

    """
    # Knot index at left/right boundary
    low  = degree
    high = 0
    high = len(knots)-1-degree

    # Check if point is exactly on left/right boundary, or outside domain
    if x <= knots[low ]: returnVal = low
    elif x >= knots[high]: returnVal = high-1
    else:
        # Perform binary search
        span = (low+high)//2
        while x < knots[span] or x >= knots[span+1]:
            if x < knots[span]:
               high = span
            else:
               low  = span
            span = (low+high)//2
        returnVal = span

    return returnVal

#==============================================================================
@types('double[:]','int','double','int','double[:]')
def basis_funs( knots, degree, x, span, values ):
    """
    Compute the non-vanishing B-splines at location x,
    given the knot sequence, polynomial degree and knot
    span. See Algorithm A2.2 in [1].

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    x : float
        Evaluation point.

    span : int
        Knot span index.

    Results
    -------
    values : numpy.ndarray
        Values of p+1 non-vanishing B-Splines at location x.

    Notes
    -----
    The original Algorithm A2.2 in The NURBS Book [1] is here
    slightly improved by using 'left' and 'right' temporary
    arrays that are one element shorter.

    """
    from numpy      import empty
    left   = empty( degree  , dtype=float )
    right  = empty( degree  , dtype=float )

    values[0] = 1.0
    for j in range(0,degree):
        left [j] = x - knots[span-j]
        right[j] = knots[span+1+j] - x
        saved    = 0.0
        for r in range(0,j+1):
            temp      = values[r] / (right[r] + left[j-r])
            values[r] = saved + right[r] * temp
            saved     = left[j-r] * temp
        values[j+1] = saved

#==============================================================================
@types('double[:]','int','double','int','double[:]')
def basis_funs_1st_der( knots, degree, x, span, ders ):
    """
    Compute the first derivative of the non-vanishing B-splines
    at location x, given the knot sequence, polynomial degree
    and knot span.

    See function 's_bsplines_non_uniform__eval_deriv' in
    Selalib's source file
    'src/splines/sll_m_bsplines_non_uniform.F90'.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    x : float
        Evaluation point.

    span : int
        Knot span index.

    Results
    -------
    ders : numpy.ndarray
        Derivatives of p+1 non-vanishing B-Splines at location x.

    """
    # Compute nonzero basis functions and knot differences for splines
    # up to degree deg-1
    from numpy      import empty
    values = empty(degree)
    basis_funs( knots, degree-1, x, span, values )

    # Compute derivatives at x using formula based on difference of
    # splines of degree deg-1
    # -------
    # j = 0
    saved = degree * values[0] / (knots[span+1]-knots[span+1-degree])
    ders[0] = -saved
    # j = 1,...,degree-1
    for j in range(1,degree):
        temp    = saved
        saved   = degree * values[j] / (knots[span+j+1]-knots[span+j+1-degree])
        ders[j] = temp - saved
    # j = degree
    ders[degree] = saved

#==============================================================================
@types('double','double[:]','int','double[:]','int')
def eval_spline_1d_scalar(x,knots,degree,coeffs,der=0):
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

#==============================================================================
@types('double[:]','double[:]','int','double[:]','double[:]','int')
def eval_spline_1d_vector(x,knots,degree,coeffs,y,der=0):
    from numpy      import empty
    if (der==0):
        basis  = empty( degree+1, dtype=float )
        for i in range(len(x)):
            span  =  find_span( knots, degree, x[i] )
            basis_funs( knots, degree, x[i], span, basis )
            
            y[i]=0.0
            for j in range(degree+1):
                y[i]+=coeffs[span-degree+j]*basis[j]
    elif (der==1):
        basis  = empty( degree+1, dtype=float )
        for i in range(len(x)):
            span  =  find_span( knots, degree, x[i] )
            basis_funs( knots, degree, x[i], span, basis )
            basis_funs_1st_der( knots, degree, x[i], span, basis )
            
            y[i]=0.0
            for j in range(degree+1):
                y[i]+=coeffs[span-degree+j]*basis[j]

#==============================================================================
@types('double','double','double[:]','int','double[:]','int','double[:,:]','int','int')
def eval_spline_2d_scalar(x,y,kts1,deg1,kts2,deg2,coeffs,der1=0,der2=0):
    span1  =  find_span( kts1, deg1, x )
    span2  =  find_span( kts2, deg2, y )
    
    from numpy      import empty
    basis1  = empty( deg1+1, dtype=float )
    basis2  = empty( deg2+1, dtype=float )
    if (der1==0):
        basis_funs( kts1, deg1, x, span1, basis1 )
    elif (der1==1):
        basis_funs_1st_der( kts1, deg1, x, span1, basis1 )
    if (der2==0):
        basis_funs( kts2, deg2, y, span2, basis2 )
    elif (der2==1):
        basis_funs_1st_der( kts2, deg2, y, span2, basis2 )
    
    theCoeffs = empty([deg1+1,deg2+1])
    theCoeffs[:,:] = coeffs[span1-deg1:span1+1,span2-deg2:span2+1]
    
    z = 0.0
    for i in range(deg1+1):
        theCoeffs[i,0] = theCoeffs[i,0]*basis2[0]
        for j in range(1,deg2+1):
            theCoeffs[i,0] += theCoeffs[i,j]*basis2[j]
        z+=theCoeffs[i,0]*basis1[i]
    return z

#==============================================================================
@types('double[:]','double[:]','double[:]','int','double[:]','int','double[:,:]','double[:,:]','int','int')
def eval_spline_2d_cross(xVec,yVec,kts1,deg1,kts2,deg2,coeffs,z,der1=0,der2=0):
    from numpy      import empty
    basis1  = empty( deg1+1 )
    basis2  = empty( deg2+1 )
    theCoeffs = empty((deg1+1,deg2+1))
    
    nx=len(xVec)
    ny=len(yVec)
    if (der1==0 and der2==0):
        for i in range(nx):
            x=xVec[i]
            span1  =  find_span( kts1, deg1, x )
            basis_funs( kts1, deg1, x, span1, basis1 )
            for j in range(ny):
                y=yVec[j]
                span2  =  find_span( kts2, deg2, y )
                basis_funs( kts2, deg2, y, span2, basis2 )
                
                theCoeffs[:,:] = coeffs[span1-deg1:span1+1,span2-deg2:span2+1]
                
                z[i,j] = 0.0
                for k in range(deg1+1):
                    theCoeffs[k,0] = theCoeffs[k,0]*basis2[0]
                    for l in range(1,deg2+1):
                        theCoeffs[k,0] += theCoeffs[k,l]*basis2[l]
                    z[i,j]+=theCoeffs[k,0]*basis1[k]
    elif (der1==0 and der2==1):
        for i in range(nx):
            x=xVec[i]
            span1  =  find_span( kts1, deg1, x )
            basis_funs( kts1, deg1, x, span1, basis1 )
            for j in range(ny):
                y=yVec[j]
                span2  =  find_span( kts2, deg2, y )
                basis_funs_1st_der( kts2, deg2, y, span2, basis2 )
                
                theCoeffs[:,:] = coeffs[span1-deg1:span1+1,span2-deg2:span2+1]
                
                z[i,j] = 0.0
                for k in range(deg1+1):
                    theCoeffs[k,0] = theCoeffs[k,0]*basis2[0]
                    for l in range(1,deg2+1):
                        theCoeffs[k,0] += theCoeffs[k,l]*basis2[l]
                    z[i,j]+=theCoeffs[k,0]*basis1[k]
    elif (der1==1 and der2==0):
        for i in range(nx):
            x=xVec[i]
            span1  =  find_span( kts1, deg1, x )
            basis_funs_1st_der( kts1, deg1, x, span1, basis1 )
            for j in range(ny):
                y=yVec[j]
                span2  =  find_span( kts2, deg2, y )
                basis_funs( kts2, deg2, y, span2, basis2 )
                
                theCoeffs[:,:] = coeffs[span1-deg1:span1+1,span2-deg2:span2+1]
                
                z[i,j] = 0.0
                for k in range(deg1+1):
                    theCoeffs[k,0] = theCoeffs[k,0]*basis2[0]
                    for l in range(1,deg2+1):
                        theCoeffs[k,0] += theCoeffs[k,l]*basis2[l]
                    z[i,j]+=theCoeffs[k,0]*basis1[k]
    elif (der1==1 and der2==1):
        for i in range(nx):
            x=xVec[i]
            span1  =  find_span( kts1, deg1, x )
            basis_funs_1st_der( kts1, deg1, x, span1, basis1 )
            for j in range(ny):
                y=yVec[j]
                span2  =  find_span( kts2, deg2, y )
                basis_funs_1st_der( kts2, deg2, y, span2, basis2 )
                
                theCoeffs[:,:] = coeffs[span1-deg1:span1+1,span2-deg2:span2+1]
                
                z[i,j] = 0.0
                for k in range(deg1+1):
                    theCoeffs[k,0] = theCoeffs[k,0]*basis2[0]
                    for l in range(1,deg2+1):
                        theCoeffs[k,0] += theCoeffs[k,l]*basis2[l]
                    z[i,j]+=theCoeffs[k,0]*basis1[k]

#==============================================================================
@types('double[:]','double[:]','double[:]','int','double[:]','int','double[:,:]','double[:]','int','int')
def eval_spline_2d_vector(x,y,kts1,deg1,kts2,deg2,coeffs,z,der1=0,der2=0):
    from numpy      import empty
    theCoeffs = empty([deg1+1,deg2+1])
    
    if (der1==0):
        if (der2==0):
            basis1  = empty( deg1+1, dtype=float )
            basis2  = empty( deg2+1, dtype=float )
            for i in range(len(x)):
                span1  =  find_span( kts1, deg1, x[i] )
                span2  =  find_span( kts2, deg2, y[i] )
                basis_funs( kts1, deg1, x[i], span1, basis1 )
                basis_funs( kts2, deg2, y[i], span2, basis2 )
                
                theCoeffs[:,:] = coeffs[span1-deg1:span1+1,span2-deg2:span2+1]
                
                z[i] = 0.0
                for j in range(deg1+1):
                    theCoeffs[j,0] = theCoeffs[j,0]*basis2[0]
                    for k in range(1,deg2+1):
                        theCoeffs[j,0] += theCoeffs[j,k]*basis2[k]
                    z[i]+=theCoeffs[j,0]*basis1[j]
        elif(der2==1):
            basis1  = empty( deg1+1, dtype=float )
            basis2  = empty( deg2+1, dtype=float )
            for i in range(len(x)):
                span1  =  find_span( kts1, deg1, x[i] )
                span2  =  find_span( kts2, deg2, y[i] )
                basis_funs( kts1, deg1, x[i], span1, basis1 )
                basis_funs_1st_der( kts2, deg2, y[i], span2, basis2 )
                
                theCoeffs[:,:] = coeffs[span1-deg1:span1+1,span2-deg2:span2+1]
                
                z[i] = 0.0
                for j in range(deg1+1):
                    theCoeffs[j,0] = theCoeffs[j,0]*basis2[0]
                    for k in range(1,deg2+1):
                        theCoeffs[j,0] += theCoeffs[j,k]*basis2[k]
                    z[i]+=theCoeffs[j,0]*basis1[j]
    elif (der1==1):
        if (der2==0):
            basis1  = empty( deg1+1, dtype=float )
            basis2  = empty( deg2+1, dtype=float )
            for i in range(len(x)):
                span1  =  find_span( kts1, deg1, x[i] )
                span2  =  find_span( kts2, deg2, y[i] )
                basis_funs_1st_der( kts1, deg1, x[i], span1, basis1 )
                basis_funs( kts2, deg2, y[i], span2, basis2 )
                
                theCoeffs[:,:] = coeffs[span1-deg1:span1+1,span2-deg2:span2+1]
                
                z[i] = 0.0
                for j in range(deg1+1):
                    theCoeffs[j,0] = theCoeffs[j,0]*basis2[0]
                    for k in range(1,deg2+1):
                        theCoeffs[j,0] += theCoeffs[j,k]*basis2[k]
                    z[i]+=theCoeffs[j,0]*basis1[j]
        elif(der2==1):
            basis1  = empty( deg1+1, dtype=float )
            basis2  = empty( deg2+1, dtype=float )
            for i in range(len(x)):
                span1  =  find_span( kts1, deg1, x[i] )
                span2  =  find_span( kts2, deg2, y[i] )
                basis_funs_1st_der( kts1, deg1, x[i], span1, basis1 )
                basis_funs_1st_der( kts2, deg2, y[i], span2, basis2 )
                
                theCoeffs[:,:] = coeffs[span1-deg1:span1+1,span2-deg2:span2+1]
                
                z[i] = 0.0
                for j in range(deg1+1):
                    theCoeffs[j,0] = theCoeffs[j,0]*basis2[0]
                    for k in range(1,deg2+1):
                        theCoeffs[j,0] += theCoeffs[j,k]*basis2[k]
                    z[i]+=theCoeffs[j,0]*basis1[j]
