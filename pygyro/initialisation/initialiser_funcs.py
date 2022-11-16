from pyccel.decorators import pure


@pure
def n0(r: 'float', CN0: 'float', kN0: 'float', deltaRN0: 'float', rp: 'float') -> 'float':
    """
    TODO
    """
    from numpy import exp, tanh
    return CN0 * exp(-kN0 * deltaRN0 * tanh((r - rp) / deltaRN0))


@pure
def Ti(r: 'float', Cti: 'float', kti: 'float', deltaRti: 'float', rp: 'float') -> 'float':
    """
    TODO
    """
    from numpy import exp, tanh
    return Cti * exp(-kti * deltaRti * tanh((r - rp) / deltaRti))


@pure
def perturbation(r: 'float', theta: 'float', z: 'float', m: 'int', n: 'int',
                 rp: 'float', deltaR: 'float', R0: 'float') -> 'float':
    """
    TODO
    """
    from numpy import exp, cos
    return exp(- (r - rp)**2 / deltaR) * cos(m * theta + n * z / R0)


@pure
def f_eq(r: 'float', vPar: 'float', CN0: 'float', kN0: 'float', deltaRN0: 'float',
         rp: 'float', Cti: 'float', kti: 'float', deltaRti: 'float') -> 'float':
    """
    TODO
    """
    from numpy import exp, sqrt, pi, real
    return n0(r, CN0, kN0, deltaRN0, rp) * exp(-0.5 * vPar * vPar / Ti(r, Cti, kti, deltaRti, rp)) \
        / real(sqrt(2.0*pi * Ti(r, Cti, kti, deltaRti, rp)))


def make_f_eq_grid(CN0: 'float', kN0: 'float', deltaRN0: 'float',
                   rp: 'float', Cti: 'float', kti: 'float', deltaRti: 'float',
                   grid_r: 'float[:]', grid_v: 'float[:]', output: 'float[:,:]', first_index: 'int'):
    """
    Parameters
    ----------
        first_index : int
            which index is the first when adressing the array:
             0 : r
             1 : v
    """
    from numpy import exp, sqrt, pi, real

    if first_index == 0:
        for i, r in enumerate(grid_r):
            for j, vPar in enumerate(grid_v):
                output[i, j] = n0(r, CN0, kN0, deltaRN0, rp) \
                    * exp(-0.5 * vPar * vPar / Ti(r, Cti, kti, deltaRti, rp)) \
                    / real(sqrt(2.0*pi * Ti(r, Cti, kti, deltaRti, rp)))
    elif first_index == 1:
        for i, vPar in enumerate(grid_v):
            for j, r in enumerate(grid_r):
                output[i, j] = n0(r, CN0, kN0, deltaRN0, rp) \
                    * exp(-0.5 * vPar * vPar / Ti(r, Cti, kti, deltaRti, rp)) \
                    / real(sqrt(2.0*pi * Ti(r, Cti, kti, deltaRti, rp)))
    else:
        print('Invalid first index')


def make_n0_grid(CN0: 'float', kN0: 'float', deltaRN0: 'float', rp: 'float',
                 grid_r: 'float[:]', output: 'float[:]'):
    """
    TODO
    """

    for i, r in enumerate(grid_r):
        output[i] = n0(r, CN0, kN0, deltaRN0, rp)


@pure
def n0deriv_normalised(r: 'float', kN0: 'float', rp: 'float', deltaRN0: 'float') -> 'float':
    """
    TODO
    """
    from numpy import tanh
    return -kN0 * (1 - tanh((r - rp) / deltaRN0)**2)


@pure
def Te(r: 'float', Cte: 'float', kte: 'float', deltaRte: 'float', rp: 'float') -> 'float':
    """
    TODO
    """
    from numpy import exp, tanh
    return Cte * exp(-kte * deltaRte * tanh((r - rp) / deltaRte))


@pure
def init_f(r: 'float', theta: 'float', z: 'float', vPar: 'float', m: 'int', n: 'int',
           eps: 'float', CN0: 'float', kN0: 'float', deltaRN0: 'float', rp: 'float',
           Cti: 'float', kti: 'float', deltaRti: 'float', deltaR: 'float', R0: 'float') -> 'float':
    """
    TODO
    """
    return f_eq(r, vPar, CN0, kN0, deltaRN0, rp, Cti, kti, deltaRti) \
        * (1 + eps * perturbation(r, theta, z, m, n, rp, deltaR, R0))


@pure
def init_f_flux(surface: 'float[:,:]', r: 'float', theta: 'float[:]', zVec: 'float[:]',
                vPar: 'float', m: 'int', n: 'int', eps: 'float', CN0: 'float', kN0: 'float',
                deltaRN0: 'float', rp: 'float', Cti: 'float', kti: 'float', deltaRti: 'float', deltaR: 'float', R0: 'float'):
    """
    TODO
    """
    for i, q in enumerate(theta):
        for j, z in enumerate(zVec):
            surface[i, j] = f_eq(r, vPar, CN0, kN0, deltaRN0, rp, Cti, kti, deltaRti) \
                * (1 + eps * perturbation(r, q, z, m, n, rp, deltaR, R0))


@pure
def init_f_pol(surface: 'float[:,:]', rVec: 'float[:]', theta: 'float[:]', z: 'float',
               vPar: 'float', m: 'int', n: 'int', eps: 'float', CN0: 'float', kN0: 'float',
               deltaRN0: 'float', rp: 'float', Cti: 'float', kti: 'float', deltaRti: 'float', deltaR: 'float', R0: 'float'):
    """
    TODO
    """
    for i, q in enumerate(theta):
        for j, r in enumerate(rVec):
            surface[i, j] = f_eq(r, vPar, CN0, kN0, deltaRN0, rp, Cti, kti, deltaRti) \
                * (1 + eps * perturbation(r, q, z, m, n, rp, deltaR, R0))


@pure
def init_f_vpar(surface: 'float[:,:]', r: 'float', theta: 'float[:]', z: 'float', vPar: 'float[:]',
                m: 'int', n: 'int', eps: 'float', CN0: 'float', kN0: 'float', deltaRN0: 'float',
                rp: 'float', Cti: 'float', kti: 'float', deltaRti: 'float', deltaR: 'float', R0: 'float'):
    """
    TODO
    """
    for i, q in enumerate(theta):
        for j, v in enumerate(vPar):
            surface[i, j] = f_eq(r, v, CN0, kN0, deltaRN0, rp, Cti, kti, deltaRti) \
                * (1 + eps * perturbation(r, q, z, m, n, rp, deltaR, R0))


@pure
def feq_vector(surface: 'float[:,:]', r_vec: 'float[:]', vPar: 'float[:]', CN0: 'float', kN0: 'float',
               deltaRN0: 'float', rp: 'float', Cti: 'float', kti: 'float', deltaRti: 'float'):
    """
    TODO
    """
    for i, r in enumerate(r_vec):
        for j, v in enumerate(vPar):
            surface[i, j] = f_eq(r, v, CN0, kN0, deltaRN0,
                                 rp, Cti, kti, deltaRti)
