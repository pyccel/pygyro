#$ header metavar includes="/home/emily/Code/selalib/build/modules"
#$ header metavar libraries="sll_splines,pppack,sll_errors,sll_assert"
#$ header metavar libdirs="/home/emily/Code/selalib/build/src/splines/splines_basic/,/home/emily/Code/selalib/build/external/pppack/,/home/emily/Code/selalib/build/src/low_level_utilities/errors/,/home/emily/Code/selalib/build/src/low_level_utilities/assert/"
import numpy as np

@low_level('sll_t_spline_matrix_periodic_banded')
class PeriodicBandedMatrix:
    @low_level('init')
    def __init__(self, n : np.int32, kl : np.int32, ku : np.int32):
        ...

    @low_level('set_element')
    def set_element(self, i : np.int32, j : np.int32, a_ij : float):
        ...

    @low_level('factorize')
    def factorize(self):
        ...

    @low_level('solve_inplace')
    def solve_inplace(self, bx : 'float[:]'):
        ...

    @low_level('free')
    def __del__(self):
        ...
