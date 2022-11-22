import numpy as np
from scipy.integrate import trapezoid


def ind_to_tp_ind(ir, it, N_r):
    """
    Convert one-dimensional indices to tensorproduct index

    Parameters
    ----------
        ir : int
            index in r-direction

        it : int
            index in theta-direction

        N_r : int
            number of nodes in r-direction

    Returns
    -------
        tp_ind : int
            tensorproduct index
    """
    tp_ind = ir + it * N_r

    return tp_ind


def neighbour_index(posr, post, ir, it, N_r, N_theta):
    """
    Calculate the tensorproduct index given two one-dimensional indices
    and a positional offset with periodic continuation for both variables

    Parameters
    ----------
        posr : int
            position of the neighbour in r-direction relative to index ir

        post : int
            position of the neighbour in theta-direction relative to index it

        ir : int
            index in r-direction of the node of which we want to compute the neighbour index

        it : int
            index in theta-direction of the node of which we want to compute the neighbour index

        N_r : int
            number of nodes in r-direction

        N_theta : int
            number of nodes in theta-direction

    Returns
    -------
        neigh_ind : int
            index of the neighbour
    """
    neigh_ind = (ir + posr) % N_r + ((it + post) % N_theta) * N_r

    return neigh_ind


def convert_2d_to_flat(arr: np.ndarray):
    """
    Convert a 2d array to a flattened array.

    Parameters
    ----------
        arr : array_like
            numpy array of shape [:, :]

    Returns
    -------
        arr_flat : array_like
            flattened version of arr
    """
    assert hasattr(arr, 'flatten'), 'Array does not have flatten attribute'

    arr_flat = arr.flatten()

    return arr_flat


def convert_flat_to_2d(arr: np.ndarray, N_theta: int = -1, N_r: int = -1):
    """
    Converts a flattened array into 2d shape. At least one of N_theta or N_r must be given.

    Parameters
    ----------
        arr : array_like
            array that will be reshaped into the wanted form

        N_theta : int (optional)
            wanted shape of array for first axis

        N_r : int (optional)
            wanted shape of array for second axis

    Returns
    -------
        arr_2d : array_like
            array with 2 axis of the wanted shape
    """
    assert N_theta != -1 or N_r != -1, 'At least one of the wanted sizes must be given!'
    assert len(arr.shape) == 1, f'Array has more than 1 axis! ({arr.shape})'

    arr_2d = arr.reshape(N_theta, N_r)

    return arr_2d


def compute_int_f(f: np.ndarray,
                  d_theta: float, d_r: float,
                  r_grid: np.ndarray, method: str = 'sum'):
    """
    Compute the integral of f over the whole domain in polar coordinates.
    A uniform grid is assumed.

    Parameters
    ----------
        f : array[float]
            array containing the values of f

        d_theta : float
            grid spacing in angular direction

        d_r : float
            grid spacing in radial direction

        r_grid : array_like
            shape of [N_r], grid points in radial direction

        method : str
            'sum' or 'trapz'; which method to use to compute the integral

    Returns
    -------
        res : float
            the integral over f
    """
    # Check if array is already in 2d format
    if len(f.shape) == 1:
        f = convert_flat_to_2d(f, N_r=len(r_grid))

    assert f.shape[1] == len(r_grid), \
        f'Second axis of f does not have the same length as r grid : {f.shape}[1] != {len(r_grid)}'

    if method == 'sum':
        res = np.sum(f, axis=0)
        res = np.multiply(res, r_grid)
        res = np.sum(res)
        res *= d_theta * d_r

    elif method == 'trapz':
        # We have to append the values of f at theta = 0 for theta = 2*pi in order for the
        # trapezoidal method to work
        f_trap = np.append(f, [f[0, :]], axis=0)

        theta_grid = np.arange(0, 2*np.pi + d_theta/10, d_theta)

        assert f_trap.shape[0] == len(theta_grid), \
            f'First axis of f does not have the same length as theta grid : {f_trap.shape}[0] != {len(theta_grid)}'

        res = trapezoid(np.multiply(trapezoid(f_trap, theta_grid, axis=0),
                                    r_grid), r_grid)

    else:
        raise NotImplementedError(
            f"Integration method {method} not implemented")

    assert isinstance(res, np.float64), f'Wrong type for result {type(res)}'

    return res


def compute_int_f_squared(f: np.ndarray,
                          d_theta: float, d_r: float,
                          r_grid: np.ndarray,
                          method: str = 'sum'):
    """
    Compute the integral of f^2 over the whole domain in polar coordinates.
    A uniform grid is assumed.

    Parameters
    ----------
        f : array[float]
            array containing the values of f

        d_theta : float
            grid spacing in angular direction

        d_r : float
            grid spacing in radial direction

        r_grid : array_like
            shape of [N_r], grid points in radial direction

        method : str
            'sum' or 'trapz'; which method to use to compute the integral

    Returns
    -------
        res : float
            the integral over f**2
    """
    # Check if array is already in 2d format
    if len(f.shape) == 1:
        f = convert_flat_to_2d(f, N_r=len(r_grid))

    assert f.shape[1] == len(r_grid), \
        f'Second axis of f does not have the same length as r grid : {f.shape}[1] != {len(r_grid)}'

    if method == 'sum':
        res = np.sum(f**2, axis=0)
        res = np.multiply(res, r_grid)
        res = np.sum(res)
        res *= d_theta * d_r

    elif method == 'trapz':
        # We have to append the values of f at theta = 0 for theta = 2*pi in order for the
        # trapezoidal method to work
        f_trap = np.append(f, [f[0, :]], axis=0)

        theta_grid = np.arange(0, 2*np.pi + d_theta/10, d_theta)

        assert f_trap.shape[0] == len(theta_grid), \
            f'First axis of f does not have the same length as theta grid : {f_trap.shape}[0] != {len(theta_grid)}'

        res = trapezoid(np.multiply(trapezoid(f_trap**2, theta_grid, axis=0),
                                    r_grid), r_grid)

    else:
        raise NotImplementedError(
            f"Integration method {method} not implemented")

    assert isinstance(res, np.float64), f'Wrong type for result {type(res)}'

    return res


def get_potential_energy(f: np.ndarray, phi: np.ndarray,
                         d_theta: float, d_r: float,
                         r_grid: np.ndarray,
                         method: str = 'sum'):
    """
    Compute the total energy, i.e. the integral of f times phi over the whole
    domain in polar coordinates. A uniform grid is assumed.

    Parameters
    ----------
        f : array_like
            2d array containing the values of f

        phi : array_like
            2d array containing the values of phi

        d_theta : float
            grid spacing in angular direction

        d_r : float
            grid spacing in radial direction

        r_grid : array_like
            shape of [N_r], grid points in radial direction

        method : str
            'sum' or 'trapz'; which method to use to compute the integral

    Returns
    -------
        res : float
            the total energy
    """
    # Check if arrays are already in 2d format
    if len(f.shape) == 1:
        f = convert_flat_to_2d(f, N_r=len(r_grid))

    assert f.shape[1] == len(r_grid), \
        f'Second axis of f does not have the same length as r grid : {f.shape}[1] != {len(r_grid)}'

    if len(phi.shape) == 1:
        phi = convert_flat_to_2d(phi, N_r=len(r_grid))

    assert phi.shape[1] == len(r_grid), \
        f'Second axis of f does not have the same length as r grid : {phi.shape}[1] != {len(r_grid)}'

    if phi.dtype == np.complex128:
        phi = np.real(phi)

    if method == 'sum':
        # point-wise multiplication
        f_phi = np.multiply(f, phi)

        # sum over values in theta direction
        res_t = np.sum(f_phi, axis=0)
        res_t = np.multiply(res_t, r_grid)

        res = np.sum(res_t)

        res *= d_theta * d_r

    elif method == 'trapz':
        # We have to append the values of f at theta = 0 for theta = 2*pi in order for the
        # trapezoidal method to work
        f_trap = np.append(f, [f[0, :]], axis=0)
        phi_trap = np.append(phi, [phi[0, :]], axis=0)

        theta_grid = np.arange(0, 2*np.pi + d_theta/10, d_theta)

        assert f_trap.shape[0] == len(theta_grid), \
            f'First axis of f does not have the same length as theta grid : {f_trap.shape}[0] != {len(theta_grid)}'

        assert phi_trap.shape[0] == len(theta_grid), \
            f'First axis of f does not have the same length as theta grid : {phi_trap.shape}[0] != {len(theta_grid)}'

        f_phi_trap = np.multiply(f_trap, phi_trap)

        res = trapezoid(np.multiply(trapezoid(f_phi_trap, theta_grid, axis=0),
                                    r_grid), r_grid)
    else:
        raise NotImplementedError(
            f"Integration method {method} not implemented")

    assert isinstance(res, np.float64), f'Wrong type for result {type(res)}'

    return res
