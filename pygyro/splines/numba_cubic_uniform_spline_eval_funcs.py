from numba import njit
from numba.pycc import CC
from numba.types import Tuple
from numpy import empty

cc = CC('spline_eval_funcs')


@cc.export('cu_find_span', 'Tuple((i4,f8))(f8, f8, f8, f8)')
@njit
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


@cc.export('cu_basis_funs', '(i4,f8,f8[:])')
@njit
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


@cc.export('cu_basis_funs_1st_der', '(i4,f8,f8,f8[:])')
@njit
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

    idx = 1/dx

    ders[0] = 0.5*b*b*idx
    ders[1] = 0.5*idx*(b*b-1)
    ders[2] = 0.5*idx*(o*o-1)
    ders[3] = 0.5*o*o*idx


@cc.export('cu_eval_spline_1d_scalar', 'f8(f8,f8[:],i4,f8[:],i4)')
@njit
def cu_eval_spline_1d_scalar(x: 'float', knots: 'float[:]', degree: 'int', coeffs: 'float[:]', der: 'int') -> 'float':
    """
    TODO
    """
    xmin, xmax, dx = knots[0], knots[1], knots[2]
    span, offset = cu_find_span(xmin, xmax, dx, x)

    basis = empty(4)
    if (der == 0):
        cu_basis_funs(span, offset, basis)
    elif (der == 1):
        cu_basis_funs_1st_der(span, offset, dx, basis)

    y = 0.0
    for j in range(4):
        y += coeffs[span-3+j]*basis[j]
    return y


@cc.export('cu_eval_spline_1d_vector', '(f8[:],f8[:],i4,f8[:],f8[:],i4)')
@njit
def cu_eval_spline_1d_vector(x: 'float[:]', knots: 'float[:]', degree: 'int', coeffs: 'float[:]', y: 'float[:]', der: 'int' = 0):
    """
    TODO
    """
    xmin, xmax, dx = knots[0], knots[1], knots[2]
    basis = empty(4)

    if (der == 0):
        for i, xi in enumerate(x):
            span, offset = cu_find_span(xmin, xmax, dx, xi)
            cu_basis_funs(span, offset, basis)

            y[i] = 0.0
            for j in range(4):
                y[i] += coeffs[span-3+j]*basis[j]

    elif (der == 1):
        for i, xi in enumerate(x):
            span, offset = cu_find_span(xmin, xmax, dx, xi)
            cu_basis_funs_1st_der(span, offset, dx, basis)

            y[i] = 0.0
            for j in range(4):
                y[i] += coeffs[span-3+j]*basis[j]


@cc.export('cu_eval_spline_2d_scalar', 'f8(f8,f8,f8[:],i4,f8[:],i4,f8[:,:],i4,i4)')
@njit
def cu_eval_spline_2d_scalar(x: 'float', y: 'float', kts1: 'float[:]', deg1: 'int', kts2: 'float[:]', deg2: 'int',
                             coeffs: 'float[:,:]', der1: 'int' = 0, der2: 'int' = 0) -> 'float':
    """
    TODO
    """
    xmin, xmax, dx = kts1[0], kts1[1], kts1[2]
    ymin, ymax, dy = kts2[0], kts2[1], kts2[2]

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
    for i in range(4):
        theCoeffs[i, 0] = theCoeffs[i, 0]*basis2[0]
        for j in range(1, 4):
            theCoeffs[i, 0] += theCoeffs[i, j]*basis2[j]
        z += theCoeffs[i, 0]*basis1[i]
    return z


@cc.export('cu_eval_spline_2d_cross', '(f8[:],f8[:],f8[:],i4,f8[:],i4,f8[:,:],f8[:,:],i4,i4)')
@njit
def cu_eval_spline_2d_cross(X: 'float[:]', Y: 'float[:]', kts1: 'float[:]', deg1: 'int', kts2: 'float[:]', deg2: 'int',
                            coeffs: 'float[:,:]', z: 'float[:,:]', der1: 'int' = 0, der2: 'int' = 0):
    """
    TODO
    """
    xmin, xmax, dx = kts1[0], kts1[1], kts1[2]
    ymin, ymax, dy = kts2[0], kts2[1], kts2[2]

    basis1 = empty(4)
    basis2 = empty(4)
    theCoeffs = empty((4, 4))

    if (der1 == 0 and der2 == 0):
        for i, x in enumerate(X):
            span1, offset1 = cu_find_span(xmin, xmax, dx, x)
            cu_basis_funs(span1, offset1, basis1)
            for j, y in enumerate(Y):
                span2, offset2 = cu_find_span(ymin, ymax, dy, y)
                cu_basis_funs(span2, offset2, basis2)

                theCoeffs[:, :] = coeffs[span1 -
                                         deg1:span1+1, span2-deg2:span2+1]

                z[i, j] = 0.0
                for k in range(4):
                    theCoeffs[k, 0] = theCoeffs[k, 0]*basis2[0]

                    for l in range(1, 4):
                        theCoeffs[k, 0] += theCoeffs[k, l]*basis2[l]
                    z[i, j] += theCoeffs[k, 0]*basis1[k]

    elif (der1 == 0 and der2 == 1):
        for i, x in enumerate(X):
            span1, offset1 = cu_find_span(xmin, xmax, dx, x)
            cu_basis_funs(span1, offset1, basis1)

            for j, y in enumerate(Y):
                span2, offset2 = cu_find_span(ymin, ymax, dy, y)
                cu_basis_funs_1st_der(span2, offset2, dx, basis2)

                theCoeffs[:, :] = coeffs[span1 -
                                         deg1:span1+1, span2-deg2:span2+1]

                z[i, j] = 0.0
                for k in range(4):
                    theCoeffs[k, 0] = theCoeffs[k, 0]*basis2[0]
                    for l in range(1, 4):
                        theCoeffs[k, 0] += theCoeffs[k, l]*basis2[l]
                    z[i, j] += theCoeffs[k, 0]*basis1[k]

    elif (der1 == 1 and der2 == 0):
        for i, x in enumerate(X):
            span1, offset1 = cu_find_span(xmin, xmax, dx, x)
            cu_basis_funs_1st_der(span1, offset1, dx, basis1)

            for j, y in enumerate(Y):
                span2, offset2 = cu_find_span(ymin, ymax, dy, y)
                cu_basis_funs(span2, offset2, basis2)

                theCoeffs[:, :] = coeffs[span1 -
                                         deg1:span1+1, span2-deg2:span2+1]

                z[i, j] = 0.0
                for k in range(4):
                    theCoeffs[k, 0] = theCoeffs[k, 0]*basis2[0]
                    for l in range(1, 4):
                        theCoeffs[k, 0] += theCoeffs[k, l]*basis2[l]
                    z[i, j] += theCoeffs[k, 0]*basis1[k]

    elif (der1 == 1 and der2 == 1):
        for i, x in enumerate(X):
            span1, offset1 = cu_find_span(xmin, xmax, dx, x)
            cu_basis_funs_1st_der(span1, offset1, dx, basis1)
            for j, y in enumerate(Y):
                span2, offset2 = cu_find_span(ymin, ymax, dy, y)
                cu_basis_funs_1st_der(span2, offset2, dx, basis2)

                theCoeffs[:, :] = coeffs[span1 -
                                         deg1:span1+1, span2-deg2:span2+1]

                z[i, j] = 0.0
                for k in range(4):
                    theCoeffs[k, 0] = theCoeffs[k, 0]*basis2[0]
                    for l in range(1, 4):
                        theCoeffs[k, 0] += theCoeffs[k, l]*basis2[l]
                    z[i, j] += theCoeffs[k, 0]*basis1[k]


@cc.export('cu_eval_spline_2d_vector', 'f8[:](f8[:],f8[:],f8[:],i4,f8[:],i4,f8[:,:],f8[:],i4,i4)')
@njit
def cu_eval_spline_2d_vector(x: 'float[:]', y: 'float[:]', kts1: 'float[:]', deg1: 'int', kts2: 'float[:]', deg2: 'int',
                             coeffs: 'float[:,:]', z: 'float[:]', der1: 'int' = 0, der2: 'int' = 0):
    """
    TODO
    """
    xmin, xmax, dx = kts1[0], kts1[1], kts1[2]
    ymin, ymax, dy = kts2[0], kts2[1], kts2[2]

    basis1 = empty(4)
    basis2 = empty(4)
    theCoeffs = empty((4, 4))

    if (der1 == 0):
        if (der2 == 0):
            for i, xi in enumerate(x):
                span1, offset1 = cu_find_span(xmin, xmax, dx, xi)
                span2, offset2 = cu_find_span(ymin, ymax, dy, y[i])
                cu_basis_funs(span1, offset1, basis1)
                cu_basis_funs(span2, offset2, basis2)

                theCoeffs[:, :] = coeffs[span1 -
                                         deg1:span1+1, span2-deg2:span2+1]

                z[i] = 0.0
                for j in range(4):
                    theCoeffs[j, 0] = theCoeffs[j, 0]*basis2[0]
                    for k in range(1, 4):
                        theCoeffs[j, 0] += theCoeffs[j, k]*basis2[k]
                    z[i] += theCoeffs[j, 0]*basis1[j]
        elif (der2 == 1):
            for i, xi in enumerate(x):
                span1, offset1 = cu_find_span(xmin, xmax, dx, xi)
                span2, offset2 = cu_find_span(ymin, ymax, dy, y[i])
                cu_basis_funs(span1, offset1, basis1)
                cu_basis_funs_1st_der(span2, offset2, dx, basis2)

                theCoeffs[:, :] = coeffs[span1 -
                                         deg1:span1+1, span2-deg2:span2+1]

                z[i] = 0.0
                for j in range(4):
                    theCoeffs[j, 0] = theCoeffs[j, 0]*basis2[0]
                    for k in range(1, 4):
                        theCoeffs[j, 0] += theCoeffs[j, k]*basis2[k]
                    z[i] += theCoeffs[j, 0]*basis1[j]
    elif (der1 == 1):
        if (der2 == 0):
            for i, xi in enumerate(x):
                span1, offset1 = cu_find_span(xmin, xmax, dx, xi)
                span2, offset2 = cu_find_span(ymin, ymax, dy, y[i])
                cu_basis_funs_1st_der(span1, offset1, dx, basis1)
                cu_basis_funs(span2, offset2, basis2)

                theCoeffs[:, :] = coeffs[span1 -
                                         deg1:span1+1, span2-deg2:span2+1]

                z[i] = 0.0
                for j in range(4):
                    theCoeffs[j, 0] = theCoeffs[j, 0]*basis2[0]
                    for k in range(1, 4):
                        theCoeffs[j, 0] += theCoeffs[j, k]*basis2[k]
                    z[i] += theCoeffs[j, 0]*basis1[j]
        elif (der2 == 1):
            for i, xi in enumerate(x):
                span1, offset1 = cu_find_span(xmin, xmax, dx, xi)
                span2, offset2 = cu_find_span(ymin, ymax, dy, y[i])
                cu_basis_funs_1st_der(span1, offset1, dx, basis1)
                cu_basis_funs_1st_der(span2, offset2, dy, basis2)

                theCoeffs[:, :] = coeffs[span1 -
                                         deg1:span1+1, span2-deg2:span2+1]

                z[i] = 0.0
                for j in range(4):
                    theCoeffs[j, 0] = theCoeffs[j, 0]*basis2[0]
                    for k in range(1, 4):
                        theCoeffs[j, 0] += theCoeffs[j, k]*basis2[k]
                    z[i] += theCoeffs[j, 0]*basis1[j]
