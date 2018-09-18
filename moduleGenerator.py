from pyccel.epyccel import ContextPyccel
from pyccel.epyccel import epyccel
from pyccel.decorators import types

import pygyro.splines.spline_eval_funcs
from pygyro.splines.mod_context_1 import find_span,basis_funs,basis_funs_1st_der

spline_context = ContextPyccel(name='context_1',context_folder='pygyro.splines',output_folder='pygyro.splines')
spline_context.insert_function(find_span, ['double[:]','int','double'])
spline_context.insert_function(basis_funs, ['double[:]','int','double','int','double[:]'])
spline_context.insert_function(basis_funs_1st_der, ['double[:]','int','double','int','double[:]'])

spline_eval_funcs = epyccel(pygyro.splines.spline_eval_funcs, context=spline_context)

import pygyro.initialisation.initialiser_func
from pygyro.initialisation.mod_initialiser_funcs import fEq, perturbation

init_context = ContextPyccel(name='initialiser_funcs',context_folder='pygyro.initialisation',output_folder='pygyro.initialisation')
init_context.insert_function(fEq, ['double','double','double','double','double','double','double','double','double'])
init_context.insert_function(perturbation, ['double','double','double','int','int','double','double','double'])

initialiser_func = epyccel(pygyro.initialisation.initialiser_func, context=init_context)


import pygyro.advection.accelerated_advection_steps
from pygyro.splines.mod_spline_eval_funcs import eval_spline_2d_cross, eval_spline_2d_scalar
from pygyro.initialisation.mod_initialiser_funcs import fEq

spline_context = ContextPyccel(name='spline_eval_funcs',context_folder='pygyro.splines',output_folder='pygyro.advection')
spline_context.insert_function(eval_spline_2d_cross, ['double[:]','double[:]','double[:]','int','double[:]','int','double[:,:]','double[:,:]','int','int'])
spline_context.insert_function(eval_spline_2d_scalar, ['double','double','double[:]','int','double[:]','int','double[:,:]','int','int'])

init_context = ContextPyccel(name='initialiser_funcs',context_folder='pygyro.initialisation',output_folder='pygyro.advection')
init_context.insert_function(fEq, ['double','double','double','double','double','double','double','double','double'])

aas_func = epyccel(pygyro.advection.accelerated_advection_steps, context=[spline_context,init_context])
