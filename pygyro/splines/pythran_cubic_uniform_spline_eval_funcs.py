from numpy import empty

# pythran export cu_find_span(float64, float64, float64, float64)


def cu_find_span(xmin: 'float', xmax: 'float', dx: 'float', x: 'float'):
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

    xmax : float
        The last break point

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

    if x == xmax:
        return span+2, 1.0
    else:
        return span+3, offset


# pythran export cu_basis_funs(int, float64, float64[:])
def cu_basis_funs(span: 'int', offset: 'float', values: 'float[:]'):
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

    values[0] = b*b*b/6
    tmp = 0.5*(1+b*o)
    values[1] = 1/6+b*tmp
    values[2] = 1/6+o*tmp
    values[3] = o*o*o/6


# pythran export cu_basis_funs_1st_der(int, float64, float64, float64[:])
def cu_basis_funs_1st_der(span: 'int', offset: 'float', dx: 'float', ders: 'float[:]'):
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

    coeff = 0.5/dx

    ders[0] = -coeff*b*b
    ders[1] = -coeff * (1+2*b-3*b*b)
    ders[2] = coeff * (1+2*o-3*o*o)
    ders[3] = coeff*o*o

# pythran export cu_eval_spline_1d_scalar(float64, float64[:], int, float64[:], int)


def cu_eval_spline_1d_scalar(x, knots, degree, coeffs, der=0):
    xmin, xmax, dx = knots
    span, offset = cu_find_span(xmin, xmax, dx, x)

    basis = empty(4)
    if (der == 0):
        cu_basis_funs(span, offset, basis)
    elif (der == 1):
        cu_basis_funs_1st_der(span, offset, dx, basis)

    y = 0.0
    for j in range(degree+1):
        y += coeffs[span-3+j]*basis[j]
    return y


# pythran export cu_eval_spline_1d_vector(float64[:], float64[:], int, float64[:], float64[:], int)
def cu_eval_spline_1d_vector(x, knots, degree, coeffs, y, der=0):
    xmin, xmax, dx = knots
    basis = empty(4)

    if (der == 0):
        for i, xi in enumerate(x):
            span, offset = cu_find_span(xmin, xmax, dx, xi)
            cu_basis_funs(span, offset, basis)

            y[i] = 0.0
            for j in range(degree+1):
                y[i] += coeffs[span-3+j]*basis[j]

    elif (der == 1):
        for i, xi in enumerate(x):
            span, offset = cu_find_span(xmin, xmax, dx, xi)
            cu_basis_funs_1st_der(span, offset, dx, basis)

            y[i] = 0.0
            for j in range(degree+1):
                y[i] += coeffs[span-3+j]*basis[j]


# pythran export cu_eval_spline_2d_scalar(float64, float64, float64[:], int, float64[:], int, float64[:,:], int, int)
def cu_eval_spline_2d_scalar(x, y, kts1, deg1, kts2, deg2, coeffs, der1=0, der2=0):
    xmin, xmax, dx = kts1
    ymin, ymax, dy = kts2

    span1, offset1 = cu_find_span(xmin, xmax, dx, x)
    span2, offset2 = cu_find_span(ymin, ymax, dy, y)

    basis1 = empty(4)
    basis2 = empty(4)

    if (der1 == 0):
        cu_basis_funs(span1, offset1, basis1)
    elif (der1 == 1):
        cu_basis_funs_1st_der(span1, offset1, dx, basis1)
    if (der2 == 0):
        cu_basis_funs(span2, offset2, basis2)
    elif (der2 == 1):
        cu_basis_funs_1st_der(span2, offset2, dx, basis2)

    theCoeffs = empty((4, 4))
    theCoeffs[:, :] = coeffs[span1-deg1:span1+1, span2-deg2:span2+1]

    z = 0.0
    for i in range(degree+1):
        theCoeffs[i, 0] = theCoeffs[i, 0]*basis2[0]
        for j in range(1, degree+1):
            theCoeffs[i, 0] += theCoeffs[i, j]*basis2[j]
        z += theCoeffs[i, 0]*basis1[i]
    return z


def cu_eval_spline_2d_cross_00(X, Y, kts1, deg1, kts2, deg2, coeffs, z):
    xmin, xmax, dx = kts1
    ymin, ymax, dy = kts2

    basis1 = empty(4)
    basis2 = empty(4)
    theCoeffs = empty((4, 4))

    for i, x in enumerate(X):
        span1, offset1 = cu_find_span(xmin, xmax, dx, x)
        cu_basis_funs(span1, offset1, basis1)
        for j, y in enumerate(Y):
            span2, offset2 = cu_find_span(ymin, ymax, dy, y)
            cu_basis_funs(span2, offset2, basis2)

            theCoeffs[:, :] = coeffs[span1 -
                                     deg1:span1+1, span2-deg2:span2+1]

            z[i, j] = 0.0
            for k in range(deg1+1):
                theCoeffs[k, 0] = theCoeffs[k, 0]*basis2[0]

                for l in range(1, deg2+1):
                    theCoeffs[k, 0] += theCoeffs[k, l]*basis2[l]
                z[i, j] += theCoeffs[k, 0]*basis1[k]


def cu_eval_spline_2d_cross_01(X, Y, kts1, deg1, kts2, deg2, coeffs, z):
    xmin, xmax, dx = kts1
    ymin, ymax, dy = kts2

    basis1 = empty(4)
    basis2 = empty(4)
    theCoeffs = empty((4, 4))
    for i, x in enumerate(X):
        span1, offset1 = cu_find_span(xmin, xmax, dx, x)
        cu_basis_funs(span1, offset1, basis1)

        for j, y in enumerate(Y):
            span2, offset2 = cu_find_span(ymin, ymax, dy, y)
            cu_basis_funs_1st_der(span2, offset2, dx, basis2)

            theCoeffs[:, :] = coeffs[span1 -
                                     deg1:span1+1, span2-deg2:span2+1]

            z[i, j] = 0.0
            for k in range(deg1+1):
                theCoeffs[k, 0] = theCoeffs[k, 0]*basis2[0]
                for l in range(1, deg2+1):
                    theCoeffs[k, 0] += theCoeffs[k, l]*basis2[l]
                z[i, j] += theCoeffs[k, 0]*basis1[k]


def cu_eval_spline_2d_cross_10(X, Y, kts1, deg1, kts2, deg2, coeffs, z):
    xmin, xmax, dx = kts1
    ymin, ymax, dy = kts2

    basis1 = empty(4)
    basis2 = empty(4)
    theCoeffs = empty((4, 4))
    for i, x in enumerate(X):
        span1, offset1 = cu_find_span(xmin, xmax, dx, x)
        cu_basis_funs_1st_der(span1, offset1, dx, basis1)
        for j, y in enumerate(Y):
            span2, offset2 = cu_find_span(ymin, ymax, dy, y)
            cu_basis_funs(span2, offset2, basis2)

            theCoeffs[:, :] = coeffs[span1 -
                                     deg1:span1+1, span2-deg2:span2+1]

            z[i, j] = 0.0
            for k in range(deg1+1):
                theCoeffs[k, 0] = theCoeffs[k, 0]*basis2[0]
                for l in range(1, deg2+1):
                    theCoeffs[k, 0] += theCoeffs[k, l]*basis2[l]
                z[i, j] += theCoeffs[k, 0]*basis1[k]


def cu_eval_spline_2d_cross_11(X, Y, kts1, deg1, kts2, deg2, coeffs, z):
    xmin, xmax, dx = kts1
    ymin, ymax, dy = kts2

    basis1 = empty(4)
    basis2 = empty(4)
    theCoeffs = empty((4, 4))
    for i, x in enumerate(X):
        span1, offset1 = cu_find_span(xmin, xmax, dx, x)
        cu_basis_funs_1st_der(span1, offset1, dx, basis1)
        for j, y in enumerate(Y):
            span2, offset2 = cu_find_span(ymin, ymax, dy, y)
            cu_basis_funs_1st_der(span2, offset2, dx, basis2)

            theCoeffs[:, :] = coeffs[span1-deg1:span1+1, span2-deg2:span2+1]

            z[i, j] = 0.0
            for k in range(deg1+1):
                theCoeffs[k, 0] = theCoeffs[k, 0]*basis2[0]
                for l in range(1, deg2+1):
                    theCoeffs[k, 0] += theCoeffs[k, l]*basis2[l]


# pythran export cu_eval_spline_2d_cross(float64[:], float64[:], float64[:], int, float64[:], int, float64[:,:], float[:,:], int, int)
def cu_eval_spline_2d_cross(X, Y, kts1, deg1, kts2, deg2, coeffs, z, der1=0, der2=0):
    if der1 == 0 and der2 == 0:
        cu_eval_spline_2d_cross_00(X, Y, kts1, deg1, kts2, deg2, coeffs, z)
    elif der1 == 1 and der2 == 0:
        cu_eval_spline_2d_cross_10(X, Y, kts1, deg1, kts2, deg2, coeffs, z)
    elif der1 == 0 and der2 == 1:
        cu_eval_spline_2d_cross_01(X, Y, kts1, deg1, kts2, deg2, coeffs, z)
    elif der1 == 1 and der2 == 1:
        cu_eval_spline_2d_cross_11(X, Y, kts1, deg1, kts2, deg2, coeffs, z)


def cu_eval_spline_2d_vector_00(x, y, kts1, deg1, kts2, deg2, coeffs, z, der1=0, der2=0):
    xmin, xmax, dx = kts1
    ymin, ymax, dy = kts2

    basis1 = empty(4)
    basis2 = empty(4)
    theCoeffs = empty((4, 4))
    for i, xi in enumerate(x):
        span1, offset1 = cu_find_span(xmin, xmax, dx, xi)
        span2, offset2 = cu_find_span(ymin, ymax, dy, y[i])
        cu_basis_funs(span1, offset1, basis1)
        cu_basis_funs(span2, offset2, basis2)

        theCoeffs[:, :] = coeffs[span1-deg1:span1+1, span2-deg2:span2+1]

        z[i] = 0.0
        for j in range(deg1+1):
            theCoeffs[j, 0] = theCoeffs[j, 0]*basis2[0]
            for k in range(1, deg2+1):
                theCoeffs[j, 0] += theCoeffs[j, k]*basis2[k]
            z[i] += theCoeffs[j, 0]*basis1[j]


def cu_eval_spline_2d_vector_01(x, y, kts1, deg1, kts2, deg2, coeffs, z, der1=0, der2=0):
    xmin, xmax, dx = kts1
    ymin, ymax, dy = kts2

    basis1 = empty(4)
    basis2 = empty(4)
    theCoeffs = empty((4, 4))
    for i, xi in enumerate(x):
        span1, offset1 = cu_find_span(xmin, xmax, dx, xi)
        span2, offset2 = cu_find_span(ymin, ymax, dy, y[i])
        cu_basis_funs(span1, offset1, basis1)
        cu_basis_funs_1st_der(span2, offset2, dy, basis2)

        theCoeffs[:, :] = coeffs[span1-deg1:span1+1, span2-deg2:span2+1]

        z[i] = 0.0
        for j in range(deg1+1):
            theCoeffs[j, 0] = theCoeffs[j, 0]*basis2[0]
            for k in range(1, deg2+1):
                theCoeffs[j, 0] += theCoeffs[j, k]*basis2[k]
            z[i] += theCoeffs[j, 0]*basis1[j]


def cu_eval_spline_2d_vector_10(x, y, kts1, deg1, kts2, deg2, coeffs, z, der1=0, der2=0):
    xmin, xmax, dx = kts1
    ymin, ymax, dy = kts2

    basis1 = empty(4)
    basis2 = empty(4)
    theCoeffs = empty((4, 4))
    for i, xi in enumerate(x):
        span1, offset1 = cu_find_span(xmin, xmax, dx, xi)
        span2, offset2 = cu_find_span(ymin, ymax, dy, y[i])
        cu_basis_funs_1st_der(span1, offset1, dx, basis1)
        cu_basis_funs(span2, offset2, basis2)

        theCoeffs[:, :] = coeffs[span1-deg1:span1+1, span2-deg2:span2+1]

        z[i] = 0.0
        for j in range(deg1+1):
            theCoeffs[j, 0] = theCoeffs[j, 0]*basis2[0]
            for k in range(1, deg2+1):
                theCoeffs[j, 0] += theCoeffs[j, k]*basis2[k]
            z[i] += theCoeffs[j, 0]*basis1[j]


def cu_eval_spline_2d_vector_11(x, y, kts1, deg1, kts2, deg2, coeffs, z, der1=0, der2=0):
    xmin, xmax, dx = kts1
    ymin, ymax, dy = kts2

    basis1 = empty(4)
    basis2 = empty(4)
    theCoeffs = empty((4, 4))
    for i, xi in enumerate(x):
        span1, offset1 = cu_find_span(xmin, xmax, dx, xi)
        span2, offset2 = cu_find_span(ymin, ymax, dy, y[i])
        cu_basis_funs_1st_der(span1, offset1, dx, basis1)
        cu_basis_funs_1st_der(span2, offset2, dy, basis2)

        theCoeffs[:, :] = coeffs[span1-deg1:span1+1, span2-deg2:span2+1]

        z[i] = 0.0
        for j in range(deg1+1):
            theCoeffs[j, 0] = theCoeffs[j, 0]*basis2[0]
            for k in range(1, deg2+1):
                theCoeffs[j, 0] += theCoeffs[j, k]*basis2[k]
            z[i] += theCoeffs[j, 0]*basis1[j]


# pythran export cu_eval_spline_2d_vector(float64[:],float64[:],float64[:],int,float64[:],int,float64[:,:],float64[:],int,int)
def cu_eval_spline_2d_vector(x, y, kts1, deg1, kts2, deg2, coeffs, z, der1=0, der2=0):
    if (der1 == 0 and der2 == 0):
        cu_eval_spline_2d_vector_00(
            x, y, kts1, deg1, kts2, deg2, coeffs, z, der1, der2)
    elif (der1 == 0 and der2 == 1):
        cu_eval_spline_2d_vector_01(
            x, y, kts1, deg1, kts2, deg2, coeffs, z, der1, der2)
    elif (der1 == 1 and der2 == 0):
        cu_eval_spline_2d_vector_10(
            x, y, kts1, deg1, kts2, deg2, coeffs, z, der1, der2)
    elif (der1 == 1 and der2 == 1):
        cu_eval_spline_2d_vector_11(
            x, y, kts1, deg1, kts2, deg2, coeffs, z, der1, der2)
