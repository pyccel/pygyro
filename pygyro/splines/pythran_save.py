from numpy import empty

# pythran export cu_find_span(float64, float64, float64)
def cu_find_span(xmin: 'float', dx: 'float', x: 'float'):
    """
    Determine the knot span index at location x, given the
    cell size and start of the domain.

    The knot span index i identifies the
    indices [i-3:i] of all 4 non-zero basis functions at a
    given location x.

    Parameters
    ----------
    xmin : float
        The first break point

    dx : float
        The cell size

    x : float
        Location of interest.

    Returns
    -------
    span : int
        Knot span index.

    offset : float

    """
    normalised_pos = (x-xmin)/dx

    span = int(normalised_pos)
    offset = normalised_pos-span

    return span, offset


# pythran export cu_basis_funs(float64, int, float64, float64[:])
def cu_basis_funs(x : 'float', span : 'int', offset : 'float', values : 'float[:]'):
    """
    Compute the non-vanishing B-splines at location x,
    given the knot sequence, polynomial degree and knot
    span.

    Parameters
    ----------
    x : float
        Evaluation point.

    span : int
        Knot span index.

    offset : float

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
    b = 1-offset
    o = offset

    values[0] = b*b*b
    values[1] = 1 + 3*(1-b*b*(2-b))
    values[2] = 1 + 3*(1-o*o*(2-o))
    values[3] = o*o*o


# pythran export cu_basis_funs_1st_der(float64, int, float64, float64, float64[:])
def cu_basis_funs_1st_der(x: 'float', span: 'int', offset : 'float', dx : 'float', ders: 'float[:]'):
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
    b = 1-offset
    o = offset

    idx = 1/dx

    ders[0] = 3*b*b*idx
    ders[1] = 3*b*idx*(3*b-4)
    ders[2] = 3*o*idx*(4-3*o)
    ders[3] = 3*o*o*idx

# pythran export cu_eval_spline_1d_scalar(float64, float64[:], int, float64[:], int)
def cu_eval_spline_1d_scalar(x: 'float', knots: 'float[:]', degree: 'int', coeffs: 'float[:]', der: 'int') -> 'float':
    """
    TODO
    """
    xmin, dx = knots
    span, offset = cu_find_span(xmin, dx, x)

    basis = empty(4)
    if (der == 0):
        cu_basis_funs(x, span, offset, basis)
    elif (der == 1):
        cu_basis_funs_1st_der(x, span, offset, dx, basis)

    y = 0.0
    for j in range(4):
        y += coeffs[span-3+j]*basis[j]
    return y


# pythran export cu_eval_spline_1d_vector(float64[:], float64[:], int, float64[:], float64[:], int)
def cu_eval_spline_1d_vector(x: 'float[:]', knots: 'float[:]', degree: 'int', coeffs: 'float[:]', y: 'float[:]', der: 'int' = 0):
    """
    TODO
    """
    xmin, dx = knots
    basis = empty(4)

    if (der == 0):
        for i, xi in enumerate(x):
            span, offset = cu_find_span(xmin, dx, xi)
            cu_basis_funs(xi, span, offset, basis)

            y[i] = 0.0
            for j in range(4):
                y[i] += coeffs[span-3+j]*basis[j]
    elif (der == 1):
        for i, xi in enumerate(x):
            span, offset = cu_find_span(xmin, dx, xi)
            cu_basis_funs_1st_der(xi, span, offset, dx, basis)

            y[i] = 0.0
            for j in range(4):
                y[i] += coeffs[span-3+j]*basis[j]


# pythran export cu_eval_spline_2d_scalar(float64, float64, float64[:], int, float64[:], int, float64[:,:], int, int)
def cu_eval_spline_2d_scalar(x: 'float', y: 'float', kts1: 'float[:]', deg1: 'int', kts2: 'float[:]', deg2: 'int',
                          coeffs: 'float[:,:]', der1: 'int' = 0, der2: 'int' = 0) -> 'float':
    """
    TODO
    """
    xmin, dx = kts1
    ymin, dy = kts2

    span1, offset1 = cu_find_span(xmin, dx, x)
    span2, offset2 = cu_find_span(ymin, dy, y)

    basis1 = empty(deg1+1)
    basis2 = empty(deg2+1)

    if (der1 == 0):
        cu_basis_funs(x, span1, offset1, basis1)
    elif (der1 == 1):
        cu_basis_funs_1st_der(x, span1, offset1, dx, basis1)
    if (der2 == 0):
        cu_basis_funs(y, span2, offset2, basis2)
    elif (der2 == 1):
        cu_basis_funs_1st_der(y, span2, offset2, dx, basis2)

    theCoeffs = empty((deg1+1, deg2+1))
    theCoeffs[:, :] = coeffs[span1-deg1:span1+1, span2-deg2:span2+1]

    z = 0.0
    for i in range(deg1+1):
        theCoeffs[i, 0] = theCoeffs[i, 0]*basis2[0]
        for j in range(1, deg2+1):
            theCoeffs[i, 0] += theCoeffs[i, j]*basis2[j]
        z += theCoeffs[i, 0]*basis1[i]
    return z


# pythran export cu_eval_spline_2d_cross(float64[:], float64[:], float64[:], int, float64[:], int, float64[:,:], float[:,:], int, int)
def cu_eval_spline_2d_cross(X: 'float[:]', Y: 'float[:]', kts1: 'float[:]', deg1: 'int', kts2: 'float[:]', deg2: 'int',
                         coeffs: 'float[:,:]', z: 'float[:,:]', der1: 'int' = 0, der2: 'int' = 0):
    """
    TODO
    """
    xmin, dx = kts1
    ymin, dy = kts2

    basis1 = empty(deg1+1)
    basis2 = empty(deg2+1)
    theCoeffs = empty((deg1+1, deg2+1))

    if (der1 == 0 and der2 == 0):
        for i, x in enumerate(X):
            span1, offset1 = cu_find_span(xmin, dx, x)
            cu_basis_funs(x, span1, offset1, basis1)
            for j, y in enumerate(Y):
                span2, offset2 = cu_find_span(ymin, dy, y)
                cu_basis_funs(y, span2, offset2, basis2)

                theCoeffs[:, :] = coeffs[span1 -
                                         deg1:span1+1, span2-deg2:span2+1]

                z[i, j] = 0.0
                for k in range(deg1+1):
                    theCoeffs[k, 0] = theCoeffs[k, 0]*basis2[0]

                    for l in range(1, deg2+1):
                        theCoeffs[k, 0] += theCoeffs[k, l]*basis2[l]
                    z[i, j] += theCoeffs[k, 0]*basis1[k]

    elif (der1 == 0 and der2 == 1):
        for i, x in enumerate(X):
            span1, offset1 = cu_find_span(xmin, dx, x)
            cu_basis_funs(x, span1, offset1, basis1)

            for j, y in enumerate(Y):
                span2, offset2 = cu_find_span(ymin, dy, y)
                cu_basis_funs_1st_der(y, span2, offset2, dx, basis2)

                theCoeffs[:, :] = coeffs[span1 -
                                         deg1:span1+1, span2-deg2:span2+1]

                z[i, j] = 0.0
                for k in range(deg1+1):
                    theCoeffs[k, 0] = theCoeffs[k, 0]*basis2[0]
                    for l in range(1, deg2+1):
                        theCoeffs[k, 0] += theCoeffs[k, l]*basis2[l]
                    z[i, j] += theCoeffs[k, 0]*basis1[k]

    elif (der1 == 1 and der2 == 0):
        for i, x in enumerate(X):
            span1, offset1 = cu_find_span(xmin, dx, x)
            cu_basis_funs_1st_der(x, span1, offset1, dx, basis1)

            for j, y in enumerate(Y):
                span2, offset2 = cu_find_span(ymin, dy, y)
                cu_basis_funs(y, span2, offset2, basis2)

                theCoeffs[:, :] = coeffs[span1 -
                                         deg1:span1+1, span2-deg2:span2+1]

                z[i, j] = 0.0
                for k in range(deg1+1):
                    theCoeffs[k, 0] = theCoeffs[k, 0]*basis2[0]
                    for l in range(1, deg2+1):
                        theCoeffs[k, 0] += theCoeffs[k, l]*basis2[l]
                    z[i, j] += theCoeffs[k, 0]*basis1[k]

    elif (der1 == 1 and der2 == 1):
        for i, x in enumerate(X):
            span1, offset1 = cu_find_span(xmin, dx, x)
            cu_basis_funs_1st_der(x, span1, offset1, dx, basis1)
            for j, y in enumerate(Y):
                span2, offset2 = cu_find_span(ymin, dy, y)
                cu_basis_funs_1st_der(y, span2, offset2, dx, basis2)

                theCoeffs[:, :] = coeffs[span1 -
                                         deg1:span1+1, span2-deg2:span2+1]

                z[i, j] = 0.0
                for k in range(deg1+1):
                    theCoeffs[k, 0] = theCoeffs[k, 0]*basis2[0]
                    for l in range(1, deg2+1):
                        theCoeffs[k, 0] += theCoeffs[k, l]*basis2[l]
                    z[i, j] += theCoeffs[k, 0]*basis1[k]


# pythran export cu_eval_spline_2d_vector(float64[:],float64[:],float64[:],int,float64[:],int,float64[:,:],float64[:],int,int)
def cu_eval_spline_2d_vector(x: 'float[:]', y: 'float[:]', kts1: 'float[:]', deg1: 'int', kts2: 'float[:]', deg2: 'int',
                          coeffs: 'float[:,:]', z: 'float[:]', der1: 'int' = 0, der2: 'int' = 0):
    """
    TODO
    """
    xmin, dx = kts1
    ymin, dy = kts2

    basis1 = empty(deg1+1)
    basis2 = empty(deg2+1)
    theCoeffs = empty((deg1+1, deg2+1))

    if (der1 == 0):
        if (der2 == 0):
            for i in range(len(x)):
                span1, offset1 = cu_find_span(xmin, dx, x[i])
                span2, offset2 = cu_find_span(ymin, dy, y[i])
                cu_basis_funs(x[i], span1, offset1, basis1)
                cu_basis_funs(y[i], span2, offset2, basis2)

                theCoeffs[:, :] = coeffs[span1 -
                                         deg1:span1+1, span2-deg2:span2+1]

                z[i] = 0.0
                for j in range(deg1+1):
                    theCoeffs[j, 0] = theCoeffs[j, 0]*basis2[0]
                    for k in range(1, deg2+1):
                        theCoeffs[j, 0] += theCoeffs[j, k]*basis2[k]
                    z[i] += theCoeffs[j, 0]*basis1[j]
        elif (der2 == 1):
            for i in range(len(x)):
                span1, offset1 = cu_find_span(xmin, dx, x[i])
                span2, offset2 = cu_find_span(ymin, dy, y[i])
                cu_basis_funs(x[i], span1, offset1, basis1)
                cu_basis_funs_1st_der(y[i], span2, offset2, dx, basis2)

                theCoeffs[:, :] = coeffs[span1 -
                                         deg1:span1+1, span2-deg2:span2+1]

                z[i] = 0.0
                for j in range(deg1+1):
                    theCoeffs[j, 0] = theCoeffs[j, 0]*basis2[0]
                    for k in range(1, deg2+1):
                        theCoeffs[j, 0] += theCoeffs[j, k]*basis2[k]
                    z[i] += theCoeffs[j, 0]*basis1[j]
    elif (der1 == 1):
        if (der2 == 0):
            for i in range(len(x)):
                span1, offset1 = cu_find_span(xmin, dx, x[i])
                span2, offset2 = cu_find_span(ymin, dy, y[i])
                cu_basis_funs_1st_der(x[i], span1, offset1, dx, basis1)
                cu_basis_funs(y[i], span2, offset2, basis2)

                theCoeffs[:, :] = coeffs[span1 -
                                         deg1:span1+1, span2-deg2:span2+1]

                z[i] = 0.0
                for j in range(deg1+1):
                    theCoeffs[j, 0] = theCoeffs[j, 0]*basis2[0]
                    for k in range(1, deg2+1):
                        theCoeffs[j, 0] += theCoeffs[j, k]*basis2[k]
                    z[i] += theCoeffs[j, 0]*basis1[j]
        elif (der2 == 1):
            for i in range(len(x)):
                span1, offset1 = cu_find_span(xmin, dx, x[i])
                span2, offset2 = cu_find_span(ymin, dy, y[i])
                cu_basis_funs_1st_der(x[i], span1, offset1, dx, basis1)
                cu_basis_funs_1st_der(y[i], span2, offset2, dx, basis2)

                theCoeffs[:, :] = coeffs[span1 -
                                         deg1:span1+1, span2-deg2:span2+1]

                z[i] = 0.0
                for j in range(deg1+1):
                    theCoeffs[j, 0] = theCoeffs[j, 0]*basis2[0]
                    for k in range(1, deg2+1):
                        theCoeffs[j, 0] += theCoeffs[j, k]*basis2[k]
                    z[i] += theCoeffs[j, 0]*basis1[j]
