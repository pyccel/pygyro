import numpy as np
from mpi4py import MPI
import os
import pytest

from pygyro.initialisation.setups import setupCylindricalGrid
from plotting.energy import KineticEnergy_v2, Mass_f, L2_f, L2_phi, PotentialEnergy_v2
from pygyro import splines as spl
from pygyro.model.layout import LayoutSwapper
from pygyro.model.process_grid import compute_2d_process_grid
from pygyro.model.grid import Grid


"""
Test the plotting.energy module by integrating over different functions both by inserting values
in f and phi.
"""


def get_callables(fun_type: str, fun_factor: str = "1", period: float = 2*np.pi):
    """
    Get callable functions and their primitive functions.

    Parameters
    ----------
    fun_type : str
        which function and its primitive should be returned

    factor : str
        the primitive will be function of fun_type * factor

    period : float
        period of a periodic function
    """
    if fun_factor not in ["1", "x", "x**2"]:
        raise NotImplementedError(
            f'{fun_factor} is an unsupported type of factor!')

    # ========================================
    # ======== non-periodic functions ========
    # ========================================
    if fun_type == "1":
        def fun(x): return 1 + 0 * x
        if fun_factor == "1":
            def Fun(x): return x
        elif fun_factor == "x":
            def Fun(x): return x**2 / 2
        elif fun_factor == "x**2":
            def Fun(x): return x**3 / 3
        else:
            raise NotImplementedError(
                f'The fun_factor {fun_factor} is not implemented!')

    elif fun_type == "x":
        def fun(x): return x
        if fun_factor == "1":
            def Fun(x): return x**2 / 2
        elif fun_factor == "x":
            def Fun(x): return x**3 / 3
        elif fun_factor == "x**2":
            def Fun(x): return x**4 / 4
        else:
            raise NotImplementedError(
                f'The fun_factor {fun_factor} is not implemented!')

    elif fun_type == "x**2":
        def fun(x): return x**2
        if fun_factor == "1":
            def Fun(x): return x**3 / 3
        elif fun_factor == "x":
            def Fun(x): return x**4 / 4
        elif fun_factor == "x**2":
            def Fun(x): return x**5 / 5
        else:
            raise NotImplementedError(
                f'The fun_factor {fun_factor} is not implemented!')

    elif fun_type == "x*sin":
        def fun(x): return x * np.sin(x)
        if fun_factor == "1":
            def Fun(x): return np.sin(x) - x * np.cos(x)
        elif fun_factor == "x":
            def Fun(x): return 2 * x * np.sin(x) - (x**2 - 2) * np.cos(x)
        elif fun_factor == "x**2":
            def Fun(x): return 3 * (x**2 - 2) * np.sin(x) - x * (x**2 - 6) * np.cos(x)
        else:
            raise NotImplementedError(
                f'The fun_factor {fun_factor} is not implemented!')

    # ====================================
    # ======== periodic functions ========
    # ====================================

    elif fun_type == "sin":
        omega = 2 * np.pi / period
        def fun(x): return np.sin(omega * x)
        if fun_factor == "1":
            def Fun(x): return - np.cos(omega * x) / omega
        else:
            raise ValueError(
                'periodic funtions cannot be multiplied with a factor that breaks periodicity!')

    elif fun_type == "cos":
        omega = 2 * np.pi / period
        def fun(x): return np.cos(omega * x)
        if fun_factor == "1":
            def Fun(x): return np.sin(omega * x) / omega
        else:
            raise ValueError(
                'periodic funtions cannot be multiplied with a factor that breaks periodicity!')

    elif fun_type == "sin**2":
        omega = 2 * np.pi / period
        def fun(x): return np.sin(omega * x) ** 2
        if fun_factor == "1":
            def Fun(x): return x / 2 - np.sin(2 * omega * x) / (4 * omega)
        else:
            raise ValueError(
                'periodic funtions cannot be multiplied with a factor that breaks periodicity!')

    elif fun_type == "cos**2":
        omega = 2 * np.pi / period
        def fun(x): return np.cos(omega * x) ** 2
        if fun_factor == "1":
            def Fun(x): return x / 2 + np.sin(2 * omega * x) / (4 * omega)
        else:
            raise ValueError(
                'periodic funtions cannot be multiplied with a factor that breaks periodicity!')

    elif fun_type == "sin*cos":
        omega = 2 * np.pi / period
        def fun(x): return np.sin(omega * x) * np.cos(omega * x)
        if fun_factor == "1":
            def Fun(x): return - np.cos(omega * x)**2 / (2 * omega)
        else:
            raise ValueError(
                'periodic funtions cannot be multiplied with a factor that breaks periodicity!')

    elif fun_type == "sin**2*cos":
        omega = 2 * np.pi / period
        def fun(x): return np.sin(omega * x) ** 2 * np.cos(omega * x)
        if fun_factor == "1":
            def Fun(x): return np.sin(omega * x)**3 / (3 * omega)
        else:
            raise ValueError(
                'periodic funtions cannot be multiplied with a factor that breaks periodicity!')

    elif fun_type == "sin*cos**2":
        omega = 2 * np.pi / period
        def fun(x): return np.sin(omega * x) * np.cos(omega * x) ** 2
        if fun_factor == "1":
            def Fun(x): return - np.cos(omega * x)**3 / (3 * omega)
        else:
            raise ValueError(
                'periodic funtions cannot be multiplied with a factor that breaks periodicity!')

    elif fun_type == "sin**2*cos**2":
        omega = 2 * np.pi / period
        def fun(x): return np.sin(omega * x) ** 2 * np.cos(omega * x) ** 2
        if fun_factor == "1":
            def Fun(x): return x / 8 - np.sin(4 * omega * x) / (32 * omega)
        else:
            raise ValueError(
                'periodic funtions cannot be multiplied with a factor that breaks periodicity!')

    else:
        raise NotImplementedError(
            f'The function {fun_type} is not implemented!')

    return fun, Fun


@pytest.mark.parametrize("fun_type_r", ["1"])
@pytest.mark.parametrize("fun_type_q", ["1"])
@pytest.mark.parametrize("fun_type_z", ["1"])
# @pytest.mark.parametrize("fun_type_q", ["1", "sin", "cos", "sin**2", "cos**2", "sin*cos", "sin**2*cos", "sin*cos**2", "sin**2*cos**2"])
# @pytest.mark.parametrize("fun_type_z", ["1", "sin", "cos", "sin**2", "cos**2", "sin*cos", "sin**2*cos", "sin*cos**2", "sin**2*cos**2"])
@pytest.mark.parametrize("fun_type_v", ["1", "x", "x**2", "x*sin"])
def test_en_kin(fun_type_v, fun_type_z, fun_type_q, fun_type_r, verbose=True):
    """
    Test the class KineticEnergy_v2 in plotting.energy by integrating different functions.

    Parameters
    ----------
    fun_i : callable
        functions that are to be integrated

    Fun_i : callable
        primitive functions of the functions fun_i

    Warning!
    --------
    Fun_r has to be the primitive function of fun_r * r, because of the integration measure!
    Fun_v has to be the primitive function of fun_v * v**2, because of the definition of the
    kinetic energy!
    """
    print()

    # Instantiate MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    constantFile = os.path.dirname(os.path.abspath(__file__)) + "/iota0.json"

    distribFunc, constants, _ = setupCylindricalGrid(constantFile=constantFile,
                                                     layout='v_parallel',
                                                     comm=comm,
                                                     allocateSaveMemory=True)

    distribFunc.setLayout('poloidal')

    eta_grid = distribFunc.eta_grid
    layout = distribFunc.getLayout('poloidal')

    KE_class = KineticEnergy_v2(eta_grid, layout,
                                constants, with_feq=False)

    r = eta_grid[0]
    q = eta_grid[1]
    z = eta_grid[2]
    v = eta_grid[3]

    idx_r = layout.inv_dims_order[0]
    idx_q = layout.inv_dims_order[1]
    idx_z = layout.inv_dims_order[2]
    idx_v = layout.inv_dims_order[3]
    my_r = r[layout.starts[idx_r]:layout.ends[idx_r]]
    my_q = q[layout.starts[idx_q]:layout.ends[idx_q]]
    my_z = z[layout.starts[idx_z]:layout.ends[idx_z]]
    my_v = v[layout.starts[idx_v]:layout.ends[idx_v]]

    # fun_v, Fun_v = get_callables(fun_type_v, fun_factor="x**2")
    fun_v, Fun_v = get_callables(fun_type_v)
    fun_z, Fun_z = get_callables(fun_type_z, period=constants.zMax)
    fun_q, Fun_q = get_callables(fun_type_q)
    # fun_r, Fun_r = get_callables(fun_type_r, fun_factor="x")
    fun_r, Fun_r = get_callables(fun_type_r)

    fun_vals_v = np.array(fun_v(my_v))
    fun_vals_z = np.array(fun_z(my_z))
    fun_vals_q = np.array(fun_q(my_q))
    fun_vals_r = np.array(fun_r(my_r))

    assert np.shape(fun_vals_v) == np.shape(my_v), \
        f'shape of fun_vals_v : {np.shape(fun_vals_v)}, shape of my_v : {np.shape(my_v)}'
    assert np.shape(fun_vals_z) == np.shape(my_z), \
        f'shape of fun_vals_z : {np.shape(fun_vals_z)}, shape of my_z : {np.shape(my_z)}'
    assert np.shape(fun_vals_q) == np.shape(my_q), \
        f'shape of fun_vals_q : {np.shape(fun_vals_q)}, shape of my_q : {np.shape(my_q)}'
    assert np.shape(fun_vals_r) == np.shape(my_r), \
        f'shape of fun_vals_r : {np.shape(fun_vals_r)}, shape of my_r : {np.shape(my_r)}'

    distribFunc._f[:, :, :, :] = 0.
    assert distribFunc._f.shape[idx_v] == my_v.shape[0]
    assert distribFunc._f.shape[idx_z] == my_z.shape[0]
    assert distribFunc._f.shape[idx_q] == my_q.shape[0]
    assert distribFunc._f.shape[idx_r] == my_r.shape[0]
    """
    Layout has to be in poloidal, i.e.:
        (v, z, theta, r)
    """
    shape = [1, 1, 1, 1]
    shape[idx_v] = my_v.size
    fun_vals_v.resize(shape)
    all_vals_v = np.repeat(np.repeat(np.repeat(fun_vals_v,
                                               my_z.size, axis=idx_z),
                                     my_q.size, axis=idx_q),
                           my_r.size, axis=idx_r)

    assert np.all(~np.isnan(fun_vals_v))
    assert np.all(~np.isinf(fun_vals_v))

    shape = [1, 1, 1, 1]
    shape[idx_z] = my_z.size
    fun_vals_z.resize(shape)
    all_vals_z = np.repeat(np.repeat(np.repeat(fun_vals_z,
                                               my_v.size, axis=idx_v),
                                     my_q.size, axis=idx_q),
                           my_r.size, axis=idx_r)

    assert np.all(~np.isnan(fun_vals_z))
    assert np.all(~np.isinf(fun_vals_z))

    shape = [1, 1, 1, 1]
    shape[idx_q] = my_q.size
    fun_vals_q.resize(shape)
    all_vals_q = np.repeat(np.repeat(np.repeat(fun_vals_q,
                                               my_v.size, axis=idx_v),
                                     my_z.size, axis=idx_z),
                           my_r.size, axis=idx_r)

    assert np.all(~np.isnan(fun_vals_q))
    assert np.all(~np.isinf(fun_vals_q))

    shape = [1, 1, 1, 1]
    shape[idx_r] = my_r.size
    fun_vals_r.resize(shape)
    r_mult = np.reshape(my_r, shape)

    assert np.shape(fun_vals_r) == np.shape(r_mult), \
        f'shape of fun_vals_r : {np.shape(fun_vals_r)} , shape of r_mult : {np.shape(r_mult)}'

    all_vals_r = np.repeat(np.repeat(np.repeat(fun_vals_r,
                                               my_v.size, axis=idx_v),
                                     my_z.size, axis=idx_z),
                           my_q.size, axis=idx_q)

    assert np.all(~np.isnan(fun_vals_r))
    assert np.all(~np.isinf(fun_vals_r))

    assert np.shape(distribFunc._f) == np.shape(all_vals_v), \
        f'shape of f : {np.shape(distribFunc._f)}, shape of all_vals_z : {np.shape(all_vals_v)}'
    assert np.shape(distribFunc._f) == np.shape(all_vals_z), \
        f'shape of f : {np.shape(distribFunc._f)}, shape of all_vals_z : {np.shape(all_vals_z)}'
    assert np.shape(distribFunc._f) == np.shape(all_vals_q), \
        f'shape of f : {np.shape(distribFunc._f)}, shape of all_vals_z : {np.shape(all_vals_q)}'
    assert np.shape(distribFunc._f) == np.shape(all_vals_r), \
        f'shape of f : {np.shape(distribFunc._f)}, shape of all_vals_z : {np.shape(all_vals_r)}'

    distribFunc._f[:, :, :, :] = all_vals_v * \
        all_vals_z * all_vals_q * all_vals_r

    # Compute exact result
    res_exact = Fun_v(constants.vMax) - Fun_v(constants.vMin)
    res_exact *= Fun_z(constants.zMax) - Fun_z(constants.zMin)
    res_exact *= Fun_q(2*np.pi) - Fun_q(0.)
    res_exact *= Fun_r(constants.rMax) - Fun_r(constants.rMin)
    # because en_kin has factor 1/2
    res_exact *= 0.5

    res_num_loc = np.zeros(1, dtype=float)
    res_num_glob = np.array([[0.]])

    res_num_loc[0] = KE_class.getKE(distribFunc)
    comm.Reduce(res_num_loc, res_num_glob, op=MPI.SUM, root=0)

    res_num = res_num_glob[0][0]

    if rank == 0 and verbose:
        print(f'numerical result : {res_num}')
        print(f'exact result : {res_exact}')
        if np.abs(res_exact) >= 1e-7:
            print(
                f'relative error : {np.abs((res_num - res_exact) / res_exact) * 100} %')
        else:
            print(f'absolute error : {np.abs(res_num - res_exact)}')

    if rank == 0:
        if np.abs(res_exact) >= 1e-7:
            assert np.abs((res_num - res_exact) / res_exact) < 1e-5
        else:
            assert np.abs(res_num - res_exact) < 1e-10


@pytest.mark.parametrize("fun_v, Fun_v", [(lambda x: 1 + 0 * x, lambda x: x), (lambda x: np.sin(x), lambda x: - np.cos(x))])
@pytest.mark.parametrize("fun_z, Fun_z", [(lambda x: 1 + 0 * x, lambda x: x), (lambda x: np.sin(x), lambda x: - np.cos(x))])
@pytest.mark.parametrize("fun_q, Fun_q", [(lambda x: 1 + 0 * x, lambda x: x), (lambda x: np.sin(x), lambda x: - np.cos(x))])
@pytest.mark.parametrize("fun_r, Fun_r", [(lambda x: 1 + 0 * x, lambda x: x**2 / 2), (np.sin, lambda x: np.sin(x) - x * np.cos(x))])
def test_mass_f(fun_v, fun_z, fun_q, fun_r, Fun_v, Fun_z, Fun_q, Fun_r, verbose=False):
    """
    Test the class Mass_f in plotting.energy by integrating different functions.

    Parameters
    ----------
    fun_i : callable
        functions that are to be integrated

    Fun_i : callable
        primitive functions of the functions fun_i

    Warning!
    --------
    Fun_r has to be the primitive function of fun_r * r, because of the integration measure!
    """

    # Instantiate MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    constantFile = os.path.dirname(os.path.abspath(__file__)) + "/iota0.json"

    distribFunc, constants, _ = setupCylindricalGrid(constantFile=constantFile,
                                                     layout='v_parallel',
                                                     comm=comm,
                                                     allocateSaveMemory=True)

    distribFunc.setLayout('poloidal')

    eta_grid = distribFunc.eta_grid
    layout = distribFunc.getLayout('poloidal')

    Mass_f_class = Mass_f(eta_grid, layout)

    r = eta_grid[0]
    q = eta_grid[1]
    z = eta_grid[2]
    v = eta_grid[3]

    idx_r = layout.inv_dims_order[0]
    idx_q = layout.inv_dims_order[1]
    idx_z = layout.inv_dims_order[2]
    idx_v = layout.inv_dims_order[3]
    my_r = r[layout.starts[idx_r]:layout.ends[idx_r]]
    my_q = q[layout.starts[idx_q]:layout.ends[idx_q]]
    my_z = z[layout.starts[idx_z]:layout.ends[idx_z]]
    my_v = v[layout.starts[idx_v]:layout.ends[idx_v]]

    fun_vals_v = np.array(fun_v(my_v))
    fun_vals_z = np.array(fun_z(my_z))
    fun_vals_q = np.array(fun_q(my_q))
    fun_vals_r = np.array(fun_r(my_r))

    assert np.shape(fun_vals_v) == np.shape(my_v), \
        f'shape of fun_vals_v : {np.shape(fun_vals_v)}, shape of my_v : {np.shape(my_v)}'
    assert np.shape(fun_vals_z) == np.shape(my_z), \
        f'shape of fun_vals_z : {np.shape(fun_vals_z)}, shape of my_z : {np.shape(my_z)}'
    assert np.shape(fun_vals_q) == np.shape(my_q), \
        f'shape of fun_vals_q : {np.shape(fun_vals_q)}, shape of my_q : {np.shape(my_q)}'
    assert np.shape(fun_vals_r) == np.shape(my_r), \
        f'shape of fun_vals_r : {np.shape(fun_vals_r)}, shape of my_r : {np.shape(my_r)}'

    distribFunc._f[:, :, :, :] = 0.
    """
    Layout has to be in poloidal, i.e.:
        (v, z, theta, r)
    """
    shape = [1, 1, 1, 1]
    shape[idx_v] = my_v.size
    fun_vals_v.resize(shape)
    all_vals_v = np.repeat(np.repeat(np.repeat(fun_vals_v,
                                               my_z.size, axis=idx_z),
                                     my_q.size, axis=idx_q),
                           my_r.size, axis=idx_r)

    assert np.all(~np.isnan(fun_vals_v))
    assert np.all(~np.isinf(fun_vals_v))

    shape = [1, 1, 1, 1]
    shape[idx_z] = my_z.size
    fun_vals_z.resize(shape)
    all_vals_z = np.repeat(np.repeat(np.repeat(fun_vals_z,
                                               my_v.size, axis=idx_v),
                                     my_q.size, axis=idx_q),
                           my_r.size, axis=idx_r)

    assert np.all(~np.isnan(fun_vals_z))
    assert np.all(~np.isinf(fun_vals_z))

    shape = [1, 1, 1, 1]
    shape[idx_q] = my_q.size
    fun_vals_q.resize(shape)
    all_vals_q = np.repeat(np.repeat(np.repeat(fun_vals_q,
                                               my_v.size, axis=idx_v),
                                     my_z.size, axis=idx_z),
                           my_r.size, axis=idx_r)

    assert np.all(~np.isnan(fun_vals_q))
    assert np.all(~np.isinf(fun_vals_q))

    shape = [1, 1, 1, 1]
    shape[idx_r] = my_r.size
    fun_vals_r.resize(shape)
    r_mult = np.reshape(my_r, shape)

    assert np.shape(fun_vals_r) == np.shape(r_mult), \
        f'shape of fun_vals_r : {np.shape(fun_vals_r)} , shape of r_mult : {np.shape(r_mult)}'

    all_vals_r = np.repeat(np.repeat(np.repeat(fun_vals_r,
                                               my_v.size, axis=idx_v),
                                     my_z.size, axis=idx_z),
                           my_q.size, axis=idx_q)

    assert np.all(~np.isnan(fun_vals_r))
    assert np.all(~np.isinf(fun_vals_r))

    assert np.shape(distribFunc._f) == np.shape(all_vals_v), \
        f'shape of f : {np.shape(distribFunc._f)}, shape of all_vals_z : {np.shape(all_vals_v)}'
    assert np.shape(distribFunc._f) == np.shape(all_vals_z), \
        f'shape of f : {np.shape(distribFunc._f)}, shape of all_vals_z : {np.shape(all_vals_z)}'
    assert np.shape(distribFunc._f) == np.shape(all_vals_q), \
        f'shape of f : {np.shape(distribFunc._f)}, shape of all_vals_z : {np.shape(all_vals_q)}'
    assert np.shape(distribFunc._f) == np.shape(all_vals_r), \
        f'shape of f : {np.shape(distribFunc._f)}, shape of all_vals_z : {np.shape(all_vals_r)}'

    distribFunc._f[:, :, :, :] = all_vals_v * \
        all_vals_z * all_vals_q * all_vals_r

    # Compute exact result
    res_exact = Fun_v(constants.vMax) - Fun_v(constants.vMin)
    res_exact *= Fun_z(constants.zMax) - Fun_z(constants.zMin)
    res_exact *= Fun_q(2*np.pi) - Fun_q(0.)
    res_exact *= Fun_r(constants.rMax) - Fun_r(constants.rMin)

    res_num_loc = np.zeros(1, dtype=float)
    res_num_glob = np.array([[0.]])

    res_num_loc[0] = Mass_f_class.getMASSF(distribFunc)
    comm.Reduce(res_num_loc, res_num_glob, op=MPI.SUM, root=0)

    res_num = res_num_glob[0][0]

    if rank == 0 and verbose:
        print(f'numerical result : {res_num}')
        print(f'exact result : {res_exact}')
        if res_exact != 0.:
            print(
                f'relative error : {np.abs((res_num - res_exact) / res_exact) * 100} %')
        else:
            print(f'absolute error : {np.abs(res_num - res_exact)}')

    if rank == 0:
        if res_exact != 0.:
            assert np.abs((res_num - res_exact) / res_exact) < 1e-7
        else:
            assert np.abs(res_num - res_exact) < 1e-7


@pytest.mark.parametrize("fun_v, Fun_v", [(lambda x: 1 + 0 * x, lambda x: x)])
@pytest.mark.parametrize("fun_z, Fun_z", [(lambda x: 1 + 0 * x, lambda x: x)])
@pytest.mark.parametrize("fun_q, Fun_q", [(lambda x: 1 + 0 * x, lambda x: x)])
@pytest.mark.parametrize("fun_r, Fun_r", [(lambda x: 1 + 0 * x, lambda x: x**2 / 2)])
def test_l2_f(fun_v, fun_z, fun_q, fun_r, Fun_v, Fun_z, Fun_q, Fun_r, verbose=False):
    """
    Test the class L2_f in plotting.energy by integrating different functions.

    Parameters
    ----------
    fun_i : callable
        functions that are to be integrated

    Fun_i : callable
        primitive functions of the functions fun_i**2

    Warning!
    --------
    Fun_r has to be the primitive function of fun_r**2 * r, because of the integration measure and
    the square in the definition of the l2 norm!
    """

    # Instantiate MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    constantFile = os.path.dirname(os.path.abspath(__file__)) + "/iota0.json"

    distribFunc, constants, _ = setupCylindricalGrid(constantFile=constantFile,
                                                     layout='v_parallel',
                                                     comm=comm,
                                                     allocateSaveMemory=True)

    distribFunc.setLayout('poloidal')

    eta_grid = distribFunc.eta_grid
    layout = distribFunc.getLayout('poloidal')

    L2_f_class = L2_f(eta_grid, layout)

    r = eta_grid[0]
    q = eta_grid[1]
    z = eta_grid[2]
    v = eta_grid[3]

    idx_r = layout.inv_dims_order[0]
    idx_q = layout.inv_dims_order[1]
    idx_z = layout.inv_dims_order[2]
    idx_v = layout.inv_dims_order[3]
    my_r = r[layout.starts[idx_r]:layout.ends[idx_r]]
    my_q = q[layout.starts[idx_q]:layout.ends[idx_q]]
    my_z = z[layout.starts[idx_z]:layout.ends[idx_z]]
    my_v = v[layout.starts[idx_v]:layout.ends[idx_v]]

    fun_vals_v = np.array(fun_v(my_v))
    fun_vals_z = np.array(fun_z(my_z))
    fun_vals_q = np.array(fun_q(my_q))
    fun_vals_r = np.array(fun_r(my_r))

    assert np.shape(fun_vals_v) == np.shape(my_v), \
        f'shape of fun_vals_v : {np.shape(fun_vals_v)}, shape of my_v : {np.shape(my_v)}'
    assert np.shape(fun_vals_z) == np.shape(my_z), \
        f'shape of fun_vals_z : {np.shape(fun_vals_z)}, shape of my_z : {np.shape(my_z)}'
    assert np.shape(fun_vals_q) == np.shape(my_q), \
        f'shape of fun_vals_q : {np.shape(fun_vals_q)}, shape of my_q : {np.shape(my_q)}'
    assert np.shape(fun_vals_r) == np.shape(my_r), \
        f'shape of fun_vals_r : {np.shape(fun_vals_r)}, shape of my_r : {np.shape(my_r)}'

    distribFunc._f[:, :, :, :] = 0.
    """
    Layout has to be in poloidal, i.e.:
        (v, z, theta, r)
    """
    shape = [1, 1, 1, 1]
    shape[idx_v] = my_v.size
    fun_vals_v.resize(shape)
    all_vals_v = np.repeat(np.repeat(np.repeat(fun_vals_v,
                                               my_z.size, axis=idx_z),
                                     my_q.size, axis=idx_q),
                           my_r.size, axis=idx_r)

    assert np.all(~np.isnan(fun_vals_v))
    assert np.all(~np.isinf(fun_vals_v))

    shape = [1, 1, 1, 1]
    shape[idx_z] = my_z.size
    fun_vals_z.resize(shape)
    all_vals_z = np.repeat(np.repeat(np.repeat(fun_vals_z,
                                               my_v.size, axis=idx_v),
                                     my_q.size, axis=idx_q),
                           my_r.size, axis=idx_r)

    assert np.all(~np.isnan(fun_vals_z))
    assert np.all(~np.isinf(fun_vals_z))

    shape = [1, 1, 1, 1]
    shape[idx_q] = my_q.size
    fun_vals_q.resize(shape)
    all_vals_q = np.repeat(np.repeat(np.repeat(fun_vals_q,
                                               my_v.size, axis=idx_v),
                                     my_z.size, axis=idx_z),
                           my_r.size, axis=idx_r)

    assert np.all(~np.isnan(fun_vals_q))
    assert np.all(~np.isinf(fun_vals_q))

    shape = [1, 1, 1, 1]
    shape[idx_r] = my_r.size
    fun_vals_r.resize(shape)
    r_mult = np.reshape(my_r, shape)

    assert np.shape(fun_vals_r) == np.shape(r_mult), \
        f'shape of fun_vals_r : {np.shape(fun_vals_r)} , shape of r_mult : {np.shape(r_mult)}'

    all_vals_r = np.repeat(np.repeat(np.repeat(fun_vals_r,
                                               my_v.size, axis=idx_v),
                                     my_z.size, axis=idx_z),
                           my_q.size, axis=idx_q)

    assert np.all(~np.isnan(fun_vals_r))
    assert np.all(~np.isinf(fun_vals_r))

    assert np.shape(distribFunc._f) == np.shape(all_vals_v), \
        f'shape of f : {np.shape(distribFunc._f)}, shape of all_vals_z : {np.shape(all_vals_v)}'
    assert np.shape(distribFunc._f) == np.shape(all_vals_z), \
        f'shape of f : {np.shape(distribFunc._f)}, shape of all_vals_z : {np.shape(all_vals_z)}'
    assert np.shape(distribFunc._f) == np.shape(all_vals_q), \
        f'shape of f : {np.shape(distribFunc._f)}, shape of all_vals_z : {np.shape(all_vals_q)}'
    assert np.shape(distribFunc._f) == np.shape(all_vals_r), \
        f'shape of f : {np.shape(distribFunc._f)}, shape of all_vals_z : {np.shape(all_vals_r)}'

    distribFunc._f[:, :, :, :] = all_vals_v * \
        all_vals_z * all_vals_q * all_vals_r

    # Compute exact result
    res_exact = Fun_v(constants.vMax) - Fun_v(constants.vMin)
    res_exact *= Fun_z(constants.zMax) - Fun_z(constants.zMin)
    res_exact *= Fun_q(2*np.pi) - Fun_q(0.)
    res_exact *= Fun_r(constants.rMax) - Fun_r(constants.rMin)

    res_num_loc = np.zeros(1, dtype=float)
    res_num_glob = np.array([[0.]])

    res_num_loc[0] = L2_f_class.getL2F(distribFunc)
    comm.Reduce(res_num_loc, res_num_glob, op=MPI.SUM, root=0)

    res_num = res_num_glob[0][0]

    if rank == 0 and verbose:
        print(f'numerical result : {res_num}')
        print(f'exact result : {res_exact}')
        if res_exact != 0.:
            print(
                f'relative error : {np.abs((res_num - res_exact) / res_exact) * 100} %')
        else:
            print(f'absolute error : {np.abs(res_num - res_exact)}')

    if rank == 0:
        if res_exact != 0.:
            assert np.abs((res_num - res_exact) / res_exact) < 1e-7
        else:
            assert np.abs(res_num - res_exact) < 1e-7


@pytest.mark.parametrize("fun_z, Fun_z", [(lambda x: 1 + 0 * x, lambda x: x)])
@pytest.mark.parametrize("fun_q, Fun_q", [(lambda x: 1 + 0 * x, lambda x: x)])
@pytest.mark.parametrize("fun_r, Fun_r", [(lambda x: 1 + 0 * x, lambda x: x**2 / 2)])
def test_l2_phi(fun_z, fun_q, fun_r, Fun_z, Fun_q, Fun_r, verbose=False):
    """
    Test the class L2_phi in plotting.energy by integrating different functions.

    Parameters
    ----------
    fun_i : callable
        functions that are to be integrated

    Fun_i : callable
        primitive functions of the functions fun_i**2

    Warning!
    --------
    Fun_r has to be the primitive function of fun_r**2 * r, because of the integration measure and
    the square in the definition of the l2 norm!
    """

    # Instantiate MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    constantFile = os.path.dirname(os.path.abspath(__file__)) + "/iota0.json"

    distribFunc, constants, _ = setupCylindricalGrid(constantFile=constantFile,
                                                     layout='v_parallel',
                                                     comm=comm,
                                                     allocateSaveMemory=True)

    npts = constants.npts

    degree = constants.splineDegrees[:-1]
    period = [False, True, True]
    domain = [[constants.rMin, constants.rMax],
              [0, 2*np.pi],
              [constants.zMin, constants.zMax]]

    nkts = [n + 1 + d * (int(p) - 1)
            for (n, d, p) in zip(npts, degree, period)]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, d, p)
             for (b, d, p) in zip(breaks, degree, period)]
    bsplines = [spl.BSplines(k, d, p)
                for (k, d, p) in zip(knots, degree, period)]
    eta_grids = [bspl.greville for bspl in bsplines]

    layout_poisson = {'v_parallel_2d': [0, 2, 1],
                      'mode_solve': [1, 2, 0]}
    layout_vpar = {'v_parallel_1d': [0, 2, 1]}
    layout_poloidal = {'poloidal': [2, 1, 0]}

    nprocs = compute_2d_process_grid(npts, size)

    remapperPhi = LayoutSwapper(comm, [layout_poisson, layout_vpar, layout_poloidal],
                                [nprocs, nprocs[0], nprocs[1]], eta_grids,
                                'v_parallel_2d')

    phi = Grid(distribFunc.eta_grid[:3], distribFunc.getSpline(slice(0, 3)),
               remapperPhi, 'mode_solve', comm, dtype=np.complex128)

    phi.setLayout('poloidal')
    distribFunc.setLayout('poloidal')

    eta_grid = distribFunc.eta_grid
    layout = distribFunc.getLayout('poloidal')

    L2_phi_class = L2_phi(eta_grid, layout)

    r = eta_grid[0]
    q = eta_grid[1]
    z = eta_grid[2]

    idx_r = layout.inv_dims_order[0]
    idx_q = layout.inv_dims_order[1]
    idx_z = layout.inv_dims_order[2]
    my_r = r[layout.starts[idx_r]:layout.ends[idx_r]]
    my_q = q[layout.starts[idx_q]:layout.ends[idx_q]]
    my_z = z[layout.starts[idx_z]:layout.ends[idx_z]]

    fun_vals_z = np.array(fun_z(my_z))
    fun_vals_q = np.array(fun_q(my_q))
    fun_vals_r = np.array(fun_r(my_r))

    assert np.shape(fun_vals_z) == np.shape(my_z), \
        f'shape of fun_vals_z : {np.shape(fun_vals_z)}, shape of my_z : {np.shape(my_z)}'
    assert np.shape(fun_vals_q) == np.shape(my_q), \
        f'shape of fun_vals_q : {np.shape(fun_vals_q)}, shape of my_q : {np.shape(my_q)}'
    assert np.shape(fun_vals_r) == np.shape(my_r), \
        f'shape of fun_vals_r : {np.shape(fun_vals_r)}, shape of my_r : {np.shape(my_r)}'

    distribFunc._f[:, :, :, :] = 1.
    phi._f[:, :, :] = 0.
    """
    Layout has to be in poloidal, i.e.:
        (z, theta, r)
    """
    shape = [1, 1, 1, 1]
    shape[idx_z] = my_z.size
    fun_vals_z.resize(shape)
    all_vals_z = np.repeat(np.repeat(fun_vals_z,
                                     my_q.size, axis=idx_q),
                           my_r.size, axis=idx_r).squeeze()

    assert np.all(~np.isnan(fun_vals_z))
    assert np.all(~np.isinf(fun_vals_z))

    shape = [1, 1, 1, 1]
    shape[idx_q] = my_q.size
    fun_vals_q.resize(shape)
    all_vals_q = np.repeat(np.repeat(fun_vals_q,
                                     my_z.size, axis=idx_z),
                           my_r.size, axis=idx_r).squeeze()

    assert np.all(~np.isnan(fun_vals_q))
    assert np.all(~np.isinf(fun_vals_q))

    shape = [1, 1, 1, 1]
    shape[idx_r] = my_r.size
    fun_vals_r.resize(shape)
    r_mult = np.reshape(my_r, shape)

    assert np.shape(fun_vals_r) == np.shape(r_mult), \
        f'shape of fun_vals_r : {np.shape(fun_vals_r)} , shape of r_mult : {np.shape(r_mult)}'

    all_vals_r = np.repeat(np.repeat(fun_vals_r,
                                     my_z.size, axis=idx_z),
                           my_q.size, axis=idx_q).squeeze()

    assert np.all(~np.isnan(fun_vals_r))
    assert np.all(~np.isinf(fun_vals_r))

    assert np.shape(phi._f) == np.shape(all_vals_z), \
        f'shape of f : {np.shape(phi._f)}, shape of all_vals_z : {np.shape(all_vals_z)}'
    assert np.shape(phi._f) == np.shape(all_vals_q), \
        f'shape of f : {np.shape(phi._f)}, shape of all_vals_z : {np.shape(all_vals_q)}'
    assert np.shape(phi._f) == np.shape(all_vals_r), \
        f'shape of f : {np.shape(phi._f)}, shape of all_vals_z : {np.shape(all_vals_r)}'

    phi._f[:, :, :] = all_vals_z * all_vals_q * all_vals_r

    # Compute exact result
    res_exact = Fun_z(constants.zMax) - Fun_z(constants.zMin)
    res_exact *= Fun_q(2*np.pi) - Fun_q(0.)
    res_exact *= Fun_r(constants.rMax) - Fun_r(constants.rMin)

    res_num_loc = np.zeros(1, dtype=float)
    res_num_glob = np.array([[0.]])

    res_num_loc[0] = L2_phi_class.getL2Phi(phi)
    comm.Reduce(res_num_loc, res_num_glob, op=MPI.SUM, root=0)

    res_num = res_num_glob[0][0]

    if rank == 0 and verbose:
        print(f'numerical result : {res_num}')
        print(f'exact result : {res_exact}')
        if res_exact != 0.:
            print(
                f'relative error : {np.abs((res_num - res_exact) / res_exact) * 100} %')
        else:
            print(f'absolute error : {np.abs(res_num - res_exact)}')

    if rank == 0:
        if res_exact != 0.:
            assert np.abs((res_num - res_exact) / res_exact) < 1e-7
        else:
            assert np.abs(res_num - res_exact) < 1e-7


@pytest.mark.parametrize("fun_v, Fun_v", [(lambda x: 1 + 0 * x, lambda x: x)])
@pytest.mark.parametrize("fun_z, Fun_z", [(lambda x: 1 + 0 * x, lambda x: x)])
@pytest.mark.parametrize("fun_q, Fun_q", [(lambda x: 1 + 0 * x, lambda x: x)])
@pytest.mark.parametrize("fun_r, Fun_r", [(lambda x: 1 + 0 * x, lambda x: x**2 / 2)])
def test_en_pot_f(fun_v, fun_z, fun_q, fun_r, Fun_v, Fun_z, Fun_q, Fun_r, verbose=False):
    """
    Test the class PotentialEnergy_v2 in plotting.energy by integrating different functions.
    This test only tests by inserting values into f.

    Parameters
    ----------
    fun_i : callable
        functions that are to be integrated

    Fun_i : callable
        primitive functions of the functions fun_i

    Warning!
    --------
    Fun_r has to be the primitive function of fun_r * r, because of the integration measure!
    """

    # Instantiate MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    constantFile = os.path.dirname(os.path.abspath(__file__)) + "/iota0.json"

    distribFunc, constants, _ = setupCylindricalGrid(constantFile=constantFile,
                                                     layout='v_parallel',
                                                     comm=comm,
                                                     allocateSaveMemory=True)

    npts = constants.npts

    degree = constants.splineDegrees[:-1]
    period = [False, True, True]
    domain = [[constants.rMin, constants.rMax],
              [0, 2*np.pi],
              [constants.zMin, constants.zMax]]

    nkts = [n + 1 + d * (int(p) - 1)
            for (n, d, p) in zip(npts, degree, period)]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, d, p)
             for (b, d, p) in zip(breaks, degree, period)]
    bsplines = [spl.BSplines(k, d, p)
                for (k, d, p) in zip(knots, degree, period)]
    eta_grids = [bspl.greville for bspl in bsplines]

    layout_poisson = {'v_parallel_2d': [0, 2, 1],
                      'mode_solve': [1, 2, 0]}
    layout_vpar = {'v_parallel_1d': [0, 2, 1]}
    layout_poloidal = {'poloidal': [2, 1, 0]}

    nprocs = compute_2d_process_grid(npts, size)

    remapperPhi = LayoutSwapper(comm, [layout_poisson, layout_vpar, layout_poloidal],
                                [nprocs, nprocs[0], nprocs[1]], eta_grids,
                                'v_parallel_2d')

    phi = Grid(distribFunc.eta_grid[:3], distribFunc.getSpline(slice(0, 3)),
               remapperPhi, 'mode_solve', comm, dtype=np.complex128)

    phi.setLayout('poloidal')
    distribFunc.setLayout('poloidal')

    phi._f[:, :, :] = 1.

    eta_grid = distribFunc.eta_grid
    layout = distribFunc.getLayout('poloidal')

    PE_class = PotentialEnergy_v2(eta_grid, layout,
                                  constants, with_feq=False)

    r = eta_grid[0]
    q = eta_grid[1]
    z = eta_grid[2]
    v = eta_grid[3]

    idx_r = layout.inv_dims_order[0]
    idx_q = layout.inv_dims_order[1]
    idx_z = layout.inv_dims_order[2]
    idx_v = layout.inv_dims_order[3]
    my_r = r[layout.starts[idx_r]:layout.ends[idx_r]]
    my_q = q[layout.starts[idx_q]:layout.ends[idx_q]]
    my_z = z[layout.starts[idx_z]:layout.ends[idx_z]]
    my_v = v[layout.starts[idx_v]:layout.ends[idx_v]]

    fun_vals_v = np.array(fun_v(my_v))
    fun_vals_z = np.array(fun_z(my_z))
    fun_vals_q = np.array(fun_q(my_q))
    fun_vals_r = np.array(fun_r(my_r))

    assert np.shape(fun_vals_v) == np.shape(my_v), \
        f'shape of fun_vals_v : {np.shape(fun_vals_v)}, shape of my_v : {np.shape(my_v)}'
    assert np.shape(fun_vals_z) == np.shape(my_z), \
        f'shape of fun_vals_z : {np.shape(fun_vals_z)}, shape of my_z : {np.shape(my_z)}'
    assert np.shape(fun_vals_q) == np.shape(my_q), \
        f'shape of fun_vals_q : {np.shape(fun_vals_q)}, shape of my_q : {np.shape(my_q)}'
    assert np.shape(fun_vals_r) == np.shape(my_r), \
        f'shape of fun_vals_r : {np.shape(fun_vals_r)}, shape of my_r : {np.shape(my_r)}'

    distribFunc._f[:, :, :, :] = 0.
    """
    Layout has to be in poloidal, i.e.:
        (v, z, theta, r)
    """
    shape = [1, 1, 1, 1]
    shape[idx_v] = my_v.size
    fun_vals_v.resize(shape)
    all_vals_v = np.repeat(np.repeat(np.repeat(fun_vals_v,
                                               my_z.size, axis=idx_z),
                                     my_q.size, axis=idx_q),
                           my_r.size, axis=idx_r)

    assert np.all(~np.isnan(fun_vals_v))
    assert np.all(~np.isinf(fun_vals_v))

    shape = [1, 1, 1, 1]
    shape[idx_z] = my_z.size
    fun_vals_z.resize(shape)
    all_vals_z = np.repeat(np.repeat(np.repeat(fun_vals_z,
                                               my_v.size, axis=idx_v),
                                     my_q.size, axis=idx_q),
                           my_r.size, axis=idx_r)

    assert np.all(~np.isnan(fun_vals_z))
    assert np.all(~np.isinf(fun_vals_z))

    shape = [1, 1, 1, 1]
    shape[idx_q] = my_q.size
    fun_vals_q.resize(shape)
    all_vals_q = np.repeat(np.repeat(np.repeat(fun_vals_q,
                                               my_v.size, axis=idx_v),
                                     my_z.size, axis=idx_z),
                           my_r.size, axis=idx_r)

    assert np.all(~np.isnan(fun_vals_q))
    assert np.all(~np.isinf(fun_vals_q))

    shape = [1, 1, 1, 1]
    shape[idx_r] = my_r.size
    fun_vals_r.resize(shape)
    r_mult = np.reshape(my_r, shape)

    assert np.shape(fun_vals_r) == np.shape(r_mult), \
        f'shape of fun_vals_r : {np.shape(fun_vals_r)} , shape of r_mult : {np.shape(r_mult)}'

    all_vals_r = np.repeat(np.repeat(np.repeat(fun_vals_r,
                                               my_v.size, axis=idx_v),
                                     my_z.size, axis=idx_z),
                           my_q.size, axis=idx_q)

    assert np.all(~np.isnan(fun_vals_r))
    assert np.all(~np.isinf(fun_vals_r))

    assert np.shape(distribFunc._f) == np.shape(all_vals_v), \
        f'shape of f : {np.shape(distribFunc._f)}, shape of all_vals_z : {np.shape(all_vals_v)}'
    assert np.shape(distribFunc._f) == np.shape(all_vals_z), \
        f'shape of f : {np.shape(distribFunc._f)}, shape of all_vals_z : {np.shape(all_vals_z)}'
    assert np.shape(distribFunc._f) == np.shape(all_vals_q), \
        f'shape of f : {np.shape(distribFunc._f)}, shape of all_vals_z : {np.shape(all_vals_q)}'
    assert np.shape(distribFunc._f) == np.shape(all_vals_r), \
        f'shape of f : {np.shape(distribFunc._f)}, shape of all_vals_z : {np.shape(all_vals_r)}'

    distribFunc._f[:, :, :, :] = all_vals_v * \
        all_vals_z * all_vals_q * all_vals_r

    # Compute exact result
    res_exact = Fun_v(constants.vMax) - Fun_v(constants.vMin)
    res_exact *= Fun_z(constants.zMax) - Fun_z(constants.zMin)
    res_exact *= Fun_q(2*np.pi) - Fun_q(0.)
    res_exact *= Fun_r(constants.rMax) - Fun_r(constants.rMin)
    # because en_pot has factor 1/2
    res_exact *= 0.5

    res_num_loc = np.zeros(1, dtype=float)
    res_num_glob = np.array([[0.]])

    res_num_loc[0] = PE_class.getPE(distribFunc, phi)
    comm.Reduce(res_num_loc, res_num_glob, op=MPI.SUM, root=0)

    res_num = res_num_glob[0][0]

    if rank == 0 and verbose:
        print(f'numerical result : {res_num}')
        print(f'exact result : {res_exact}')
        if res_exact != 0.:
            print(
                f'relative error : {np.abs((res_num - res_exact) / res_exact) * 100} %')
        else:
            print(f'absolute error : {np.abs(res_num - res_exact)}')

    if rank == 0:
        if res_exact != 0.:
            assert np.abs((res_num - res_exact) / res_exact) < 1e-7
        else:
            assert np.abs(res_num - res_exact) < 1e-7


@pytest.mark.parametrize("fun_z, Fun_z", [(lambda x: 1 + 0 * x, lambda x: x)])
@pytest.mark.parametrize("fun_q, Fun_q", [(lambda x: 1 + 0 * x, lambda x: x)])
@pytest.mark.parametrize("fun_r, Fun_r", [(lambda x: 1 + 0 * x, lambda x: x**2 / 2)])
def test_en_pot_phi(fun_z, fun_q, fun_r, Fun_z, Fun_q, Fun_r, verbose=False):
    """
    Test the class PotentialEnergy_v2 in plotting.energy by integrating different functions.
    This test only tests by inserting values into phi.

    Parameters
    ----------
    fun_i : callable
        functions that are to be integrated

    Fun_i : callable
        primitive functions of the functions fun_i

    Warning!
    --------
    Fun_r has to be the primitive function of fun_r * r, because of the integration measure!
    """

    # Instantiate MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    constantFile = os.path.dirname(os.path.abspath(__file__)) + "/iota0.json"

    distribFunc, constants, _ = setupCylindricalGrid(constantFile=constantFile,
                                                     layout='v_parallel',
                                                     comm=comm,
                                                     allocateSaveMemory=True)

    npts = constants.npts

    degree = constants.splineDegrees[:-1]
    period = [False, True, True]
    domain = [[constants.rMin, constants.rMax],
              [0, 2*np.pi],
              [constants.zMin, constants.zMax]]

    nkts = [n + 1 + d * (int(p) - 1)
            for (n, d, p) in zip(npts, degree, period)]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, d, p)
             for (b, d, p) in zip(breaks, degree, period)]
    bsplines = [spl.BSplines(k, d, p)
                for (k, d, p) in zip(knots, degree, period)]
    eta_grids = [bspl.greville for bspl in bsplines]

    layout_poisson = {'v_parallel_2d': [0, 2, 1],
                      'mode_solve': [1, 2, 0]}
    layout_vpar = {'v_parallel_1d': [0, 2, 1]}
    layout_poloidal = {'poloidal': [2, 1, 0]}

    nprocs = compute_2d_process_grid(npts, size)

    remapperPhi = LayoutSwapper(comm, [layout_poisson, layout_vpar, layout_poloidal],
                                [nprocs, nprocs[0], nprocs[1]], eta_grids,
                                'v_parallel_2d')

    phi = Grid(distribFunc.eta_grid[:3], distribFunc.getSpline(slice(0, 3)),
               remapperPhi, 'mode_solve', comm, dtype=np.complex128)

    phi.setLayout('poloidal')
    distribFunc.setLayout('poloidal')

    eta_grid = distribFunc.eta_grid

    layout = distribFunc.getLayout('poloidal')

    PE_class = PotentialEnergy_v2(eta_grid, layout,
                                  constants, with_feq=False)

    r = eta_grid[0]
    q = eta_grid[1]
    z = eta_grid[2]

    distribFunc._f[:, :, :, :] = 1.
    phi._f[:, :, :] = 0.

    idx_r = layout.inv_dims_order[0]
    idx_q = layout.inv_dims_order[1]
    idx_z = layout.inv_dims_order[2]
    my_r = r[layout.starts[idx_r]:layout.ends[idx_r]]
    my_q = q[layout.starts[idx_q]:layout.ends[idx_q]]
    my_z = z[layout.starts[idx_z]:layout.ends[idx_z]]

    fun_vals_z = np.array(fun_z(my_z))
    fun_vals_q = np.array(fun_q(my_q))
    fun_vals_r = np.array(fun_r(my_r))

    assert np.shape(fun_vals_z) == np.shape(my_z), \
        f'shape of fun_vals_z : {np.shape(fun_vals_z)}, shape of my_z : {np.shape(my_z)}'
    assert np.shape(fun_vals_q) == np.shape(my_q), \
        f'shape of fun_vals_q : {np.shape(fun_vals_q)}, shape of my_q : {np.shape(my_q)}'
    assert np.shape(fun_vals_r) == np.shape(my_r), \
        f'shape of fun_vals_r : {np.shape(fun_vals_r)}, shape of my_r : {np.shape(my_r)}'

    """
    Phi Layout has to be in poloidal, i.e.:
        (z, theta, r)
    """
    shape = [1, 1, 1, 1]
    shape[idx_z] = my_z.size
    fun_vals_z.resize(shape)
    all_vals_z = np.repeat(np.repeat(fun_vals_z,
                                     my_q.size, axis=idx_q),
                           my_r.size, axis=idx_r).squeeze()

    assert np.all(~np.isnan(fun_vals_z))
    assert np.all(~np.isinf(fun_vals_z))

    shape = [1, 1, 1, 1]
    shape[idx_q] = my_q.size
    fun_vals_q.resize(shape)
    all_vals_q = np.repeat(np.repeat(fun_vals_q,
                                     my_z.size, axis=idx_z),
                           my_r.size, axis=idx_r).squeeze()

    assert np.all(~np.isnan(fun_vals_q))
    assert np.all(~np.isinf(fun_vals_q))

    shape = [1, 1, 1, 1]
    shape[idx_r] = my_r.size
    fun_vals_r.resize(shape)
    r_mult = np.reshape(my_r, shape)

    assert np.shape(fun_vals_r) == np.shape(r_mult), \
        f'shape of fun_vals_r : {np.shape(fun_vals_r)} , shape of r_mult : {np.shape(r_mult)}'

    all_vals_r = np.repeat(np.repeat(fun_vals_r,
                                     my_z.size, axis=idx_z),
                           my_q.size, axis=idx_q).squeeze()

    assert np.all(~np.isnan(fun_vals_r))
    assert np.all(~np.isinf(fun_vals_r))

    assert np.shape(phi._f) == np.shape(all_vals_z), \
        f'shape of f : {np.shape(phi._f)}, shape of all_vals_z : {np.shape(all_vals_z)}'
    assert np.shape(phi._f) == np.shape(all_vals_q), \
        f'shape of f : {np.shape(phi._f)}, shape of all_vals_z : {np.shape(all_vals_q)}'
    assert np.shape(phi._f) == np.shape(all_vals_r), \
        f'shape of f : {np.shape(phi._f)}, shape of all_vals_z : {np.shape(all_vals_r)}'

    phi._f[:, :, :] = all_vals_z * all_vals_q * all_vals_r

    # Compute exact result
    res_exact = constants.vMax - constants.vMin
    res_exact *= Fun_z(constants.zMax) - Fun_z(constants.zMin)
    res_exact *= Fun_q(2*np.pi) - Fun_q(0.)
    res_exact *= Fun_r(constants.rMax) - Fun_r(constants.rMin)
    # because en_pot has factor 1/2
    res_exact *= 0.5

    res_num_loc = np.zeros(1, dtype=float)
    res_num_glob = np.array([[0.]])

    res_num_loc[0] = PE_class.getPE(distribFunc, phi)
    comm.Reduce(res_num_loc, res_num_glob, op=MPI.SUM, root=0)

    res_num = res_num_glob[0][0]

    if rank == 0 and verbose:
        print(f'numerical result : {res_num}')
        print(f'exact result : {res_exact}')
        if res_exact != 0.:
            print(
                f'relative error : {np.abs((res_num - res_exact) / res_exact) * 100} %')
        else:
            print(f'absolute error : {np.abs(res_num - res_exact)}')

    if rank == 0:
        if res_exact != 0.:
            assert np.abs((res_num - res_exact) / res_exact) < 1e-7
        else:
            assert np.abs(res_num - res_exact) < 1e-7


def main():
    pass


if __name__ == '__main__':
    main()
