from numpy import zeros, empty

# pythran export find_span(float64[:], int, float64)


def find_span(knots, degree, x):
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
    low = degree
    high = 0
    high = len(knots)-1-degree

    # Check if point is exactly on left/right boundary, or outside domain
    if x <= knots[low]:
        returnVal = low
    elif x >= knots[high]:
        returnVal = high-1
    else:
        # Perform binary search
        span = (low+high)//2
        while x < knots[span] or x >= knots[span+1]:
            if x < knots[span]:
                high = span
            else:
                low = span
            span = (low+high)//2
        returnVal = span

    return returnVal


# pythran export basis_funs(float64[:], int, float64, int, float64[:])
def basis_funs(knots, degree, x, span, values):
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
    left = zeros(degree)
    right = zeros(degree)

    values[0] = 1.0
    for j in range(0, degree):
        left[j] = x - knots[span-j]
        right[j] = knots[span+1+j] - x
        saved = 0.0
        for r in range(0, j+1):
            temp = values[r] / (right[r] + left[j-r])
            values[r] = saved + right[r] * temp
            saved = left[j-r] * temp
        values[j+1] = saved


# pythran export basis_funs_1st_der(float64[:], int, float64, int, float64[:])
def basis_funs_1st_der(knots, degree, x, span, ders):
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
    values = zeros(degree)
    basis_funs(knots, degree-1, x, span, values)

    # Compute derivatives at x using formula based on difference of
    # splines of degree deg-1
    # -------
    # j = 0
    saved = degree * values[0] / (knots[span+1]-knots[span+1-degree])
    ders[0] = -saved
    # j = 1,...,degree-1
    for j in range(1, degree):
        temp = saved
        saved = degree * values[j] / (knots[span+j+1]-knots[span+j+1-degree])
        ders[j] = temp - saved
    # j = degree
    ders[degree] = saved


# pythran export eval_spline_1d_scalar(float64, float64[:], int, float64[:], int)
def eval_spline_1d_scalar(x, knots, degree, coeffs, der=0):
    span = find_span(knots, degree, x)

    basis = zeros(degree+1)
    if (der == 0):
        basis_funs(knots, degree, x, span, basis)
    elif (der == 1):
        basis_funs_1st_der(knots, degree, x, span, basis)

    y = 0.0
    for j in range(degree+1):
        y += coeffs[span-degree+j]*basis[j]
    return y


# pythran export eval_spline_1d_vector(float64[:], float64[:], int, float64[:], float64[:], int)
def eval_spline_1d_vector(x, knots, degree, coeffs, y, der=0):
    if (der == 0):
        for i, xi in enumerate(x):
            span = find_span(knots, degree, xi)
            basis = zeros(degree+1)
            basis_funs(knots, degree, xi, span, basis)

            y[i] = 0.0
            for j in range(degree+1):
                y[i] += coeffs[span-degree+j]*basis[j]
    elif (der == 1):
        for i, xi in enumerate(x):
            span = find_span(knots, degree, xi)
            basis = zeros(degree+1)
            basis_funs(knots, degree, xi, span, basis)
            basis_funs_1st_der(knots, degree, xi, span, basis)

            y[i] = 0.0
            for j in range(degree+1):
                y[i] += coeffs[span-degree+j]*basis[j]


# pythran export eval_spline_2d_scalar(float64, float64, float64[:], int, float64[:], int, float64[:,:], int, int)
def eval_spline_2d_scalar(x, y, kts1, deg1, kts2, deg2, coeffs, der1=0, der2=0):
    span1 = find_span(kts1, deg1, x)
    span2 = find_span(kts2, deg2, y)

    basis1 = zeros(deg1+1)
    basis2 = zeros(deg2+1)
    if (der1 == 0):
        basis_funs(kts1, deg1, x, span1, basis1)
    elif (der1 == 1):
        basis_funs_1st_der(kts1, deg1, x, span1, basis1)
    if (der2 == 0):
        basis_funs(kts2, deg2, y, span2, basis2)
    elif (der2 == 1):
        basis_funs_1st_der(kts2, deg2, y, span2, basis2)

    theCoeffs = zeros((deg1+1, deg2+1))
    theCoeffs[:] = coeffs[span1-deg1:span1+1, span2-deg2:span2+1]

    z = 0.0
    for i in range(deg1+1):
        theCoeffs[i, 0] = theCoeffs[i, 0]*basis2[0]
        for j in range(1, deg2+1):
            theCoeffs[i, 0] += theCoeffs[i, j]*basis2[j]
        z += theCoeffs[i, 0]*basis1[i]
    return z


def eval_spline_2d_cross_00(X, Y, kts1, deg1, kts2, deg2, coeffs, z):
    basis1 = zeros(deg1+1)
    basis2 = zeros(deg2+1)
    theCoeffs = zeros((deg1+1, deg2+1))

    for i, x in enumerate(X):
        span1 = find_span(kts1, deg1, x)
        basis_funs(kts1, deg1, x, span1, basis1)
        for j, y in enumerate(Y):
            span2 = find_span(kts2, deg2, y)
            basis_funs(kts2, deg2, y, span2, basis2)

            theCoeffs[:, :] = coeffs[span1-deg1:span1+1, span2-deg2:span2+1]

            z[i, j] = 0.0
            for k in range(deg1+1):
                theCoeffs[k, 0] = theCoeffs[k, 0]*basis2[0]
                for l in range(1, deg2+1):
                    theCoeffs[k, 0] += theCoeffs[k, l]*basis2[l]
                z[i, j] += theCoeffs[k, 0]*basis1[k]


def eval_spline_2d_cross_01(X, Y, kts1, deg1, kts2, deg2, coeffs, z):
    basis1 = zeros(deg1+1)
    basis2 = zeros(deg2+1)
    theCoeffs = zeros((deg1+1, deg2+1))
    for i, x in enumerate(X):
        span1 = find_span(kts1, deg1, x)
        basis_funs(kts1, deg1, x, span1, basis1)
        for j, y in enumerate(Y):
            span2 = find_span(kts2, deg2, y)
            basis_funs_1st_der(kts2, deg2, y, span2, basis2)

            theCoeffs[:, :] = coeffs[span1-deg1:span1+1, span2-deg2:span2+1]

            z[i, j] = 0.0
            for k in range(deg1+1):
                theCoeffs[k, 0] = theCoeffs[k, 0]*basis2[0]
                for l in range(1, deg2+1):
                    theCoeffs[k, 0] += theCoeffs[k, l]*basis2[l]
                z[i, j] += theCoeffs[k, 0]*basis1[k]


def eval_spline_2d_cross_10(X, Y, kts1, deg1, kts2, deg2, coeffs, z):
    basis1 = zeros(deg1+1)
    basis2 = zeros(deg2+1)
    theCoeffs = zeros((deg1+1, deg2+1))
    for i, x in enumerate(X):
        span1 = find_span(kts1, deg1, x)
        basis_funs_1st_der(kts1, deg1, x, span1, basis1)
        for j, y in enumerate(Y):
            span2 = find_span(kts2, deg2, y)
            basis_funs(kts2, deg2, y, span2, basis2)

            theCoeffs[:, :] = coeffs[span1-deg1:span1+1, span2-deg2:span2+1]

            z[i, j] = 0.0
            for k in range(deg1+1):
                theCoeffs[k, 0] = theCoeffs[k, 0]*basis2[0]
                for l in range(1, deg2+1):
                    theCoeffs[k, 0] += theCoeffs[k, l]*basis2[l]
                z[i, j] += theCoeffs[k, 0]*basis1[k]


def eval_spline_2d_cross_11(X, Y, kts1, deg1, kts2, deg2, coeffs, z):
    basis1 = zeros(deg1+1)
    basis2 = zeros(deg2+1)
    theCoeffs = zeros((deg1+1, deg2+1))
    for i, x in enumerate(X):
        span1 = find_span(kts1, deg1, x)
        basis_funs_1st_der(kts1, deg1, x, span1, basis1)
        for j, y in enumerate(Y):
            span2 = find_span(kts2, deg2, y)
            basis_funs_1st_der(kts2, deg2, y, span2, basis2)

            theCoeffs[:, :] = coeffs[span1-deg1:span1+1, span2-deg2:span2+1]

            z[i, j] = 0.0
            for k in range(deg1+1):
                theCoeffs[k, 0] = theCoeffs[k, 0]*basis2[0]
                for l in range(1, deg2+1):
                    theCoeffs[k, 0] += theCoeffs[k, l]*basis2[l]
                z[i, j] += 0


# pythran export eval_spline_2d_cross(float64[:], float64[:], float64[:], int, float64[:], int, float64[:,:], float[:,:], int, int)
def eval_spline_2d_cross(X, Y, kts1, deg1, kts2, deg2, coeffs, z, der1=0, der2=0):
    if der1 == 0 and der2 == 0:
        eval_spline_2d_cross_00(X, Y, kts1, deg1, kts2, deg2, coeffs, z)
    elif der1 == 1 and der2 == 0:
        eval_spline_2d_cross_10(X, Y, kts1, deg1, kts2, deg2, coeffs, z)
    elif der1 == 0 and der2 == 1:
        eval_spline_2d_cross_01(X, Y, kts1, deg1, kts2, deg2, coeffs, z)
    elif der1 == 1 and der2 == 1:
        eval_spline_2d_cross_11(X, Y, kts1, deg1, kts2, deg2, coeffs, z)


def eval_spline_2d_vector_00(x, y, kts1, deg1, kts2, deg2, coeffs, z, der1=0, der2=0):
    for i in range(len(x)):
        span1 = find_span(kts1, deg1, x[i])
        span2 = find_span(kts2, deg2, y[i])
        basis1 = empty(deg1+1)
        basis2 = empty(deg2+1)
        basis_funs(kts1, deg1, x[i], span1, basis1)
        basis_funs(kts2, deg2, y[i], span2, basis2)

        theCoeffs = coeffs[span1-deg1:span1+1, span2-deg2:span2+1].copy()

        z[i] = 0.0
        for j in range(deg1+1):
            theCoeffs[j, 0] = theCoeffs[j, 0]*basis2[0]
            for k in range(1, deg2+1):
                theCoeffs[j, 0] += theCoeffs[j, k]*basis2[k]
            z[i] += theCoeffs[j, 0]*basis1[j]


def eval_spline_2d_vector_01(x, y, kts1, deg1, kts2, deg2, coeffs, z, der1=0, der2=0):
    for i in range(len(x)):
        span1 = find_span(kts1, deg1, x[i])
        span2 = find_span(kts2, deg2, y[i])
        basis1 = empty(deg1+1)
        basis2 = empty(deg2+1)
        basis_funs(kts1, deg1, x[i], span1, basis1)
        basis_funs_1st_der(kts2, deg2, y[i], span2, basis2)

        theCoeffs = coeffs[span1-deg1:span1+1, span2-deg2:span2+1].copy()

        z[i] = 0.0
        for j in range(deg1+1):
            theCoeffs[j, 0] = theCoeffs[j, 0]*basis2[0]
            for k in range(1, deg2+1):
                theCoeffs[j, 0] += theCoeffs[j, k]*basis2[k]
            z[i] += theCoeffs[j, 0]*basis1[j]


def eval_spline_2d_vector_10(x, y, kts1, deg1, kts2, deg2, coeffs, z, der1=0, der2=0):
    for i in range(len(x)):
        span1 = find_span(kts1, deg1, x[i])
        span2 = find_span(kts2, deg2, y[i])
        basis1 = empty(deg1+1)
        basis2 = empty(deg2+1)
        basis_funs_1st_der(kts1, deg1, x[i], span1, basis1)
        basis_funs(kts2, deg2, y[i], span2, basis2)

        theCoeffs = coeffs[span1-deg1:span1+1, span2-deg2:span2+1].copy()

        z[i] = 0.0
        for j in range(deg1+1):
            theCoeffs[j, 0] = theCoeffs[j, 0]*basis2[0]
            for k in range(1, deg2+1):
                theCoeffs[j, 0] += theCoeffs[j, k]*basis2[k]
            z[i] += theCoeffs[j, 0]*basis1[j]


def eval_spline_2d_vector_11(x, y, kts1, deg1, kts2, deg2, coeffs, z, der1=0, der2=0):
    for i in range(len(x)):
        span1 = find_span(kts1, deg1, x[i])
        span2 = find_span(kts2, deg2, y[i])
        basis1 = empty(deg1+1)
        basis2 = empty(deg2+1)
        basis_funs_1st_der(kts1, deg1, x[i], span1, basis1)
        basis_funs_1st_der(kts2, deg2, y[i], span2, basis2)

        theCoeffs = coeffs[span1-deg1:span1+1, span2-deg2:span2+1]

        z[i] = 0.0
        for j in range(deg1+1):
            theCoeffs[j, 0] = theCoeffs[j, 0]*basis2[0]
            for k in range(1, deg2+1):
                theCoeffs[j, 0] += theCoeffs[j, k]*basis2[k]
            z[i] += theCoeffs[j, 0]*basis1[j]


# pythran export eval_spline_2d_vector(float64[:],float64[:],float64[:],int,float64[:],int,float64[:,:],float64[:],int,int)
def eval_spline_2d_vector(x, y, kts1, deg1, kts2, deg2, coeffs, z, der1=0, der2=0):
    if (der1 == 0 and der2 == 0):
        eval_spline_2d_vector_00(
            x, y, kts1, deg1, kts2, deg2, coeffs, z, der1, der2)
    elif (der1 == 0 and der2 == 1):
        eval_spline_2d_vector_01(
            x, y, kts1, deg1, kts2, deg2, coeffs, z, der1, der2)
    elif (der1 == 1 and der2 == 0):
        eval_spline_2d_vector_10(
            x, y, kts1, deg1, kts2, deg2, coeffs, z, der1, der2)
    elif (der1 == 1 and der2 == 1):
        eval_spline_2d_vector_11(
            x, y, kts1, deg1, kts2, deg2, coeffs, z, der1, der2)
