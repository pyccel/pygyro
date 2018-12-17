from pyccel.epyccel import ContextPyccel
from pyccel.epyccel import epyccel
from pyccel.decorators import types

import initialiser_func
from mod_initialiser_funcs import fEq, perturbation

context = ContextPyccel(name='initialiser_funcs')
context.insert_function(fEq, ['double[:]','int','double'])
context.insert_function(perturbation, ['double[:]','int','double','int','double[:]'])

initialiser_func = epyccel(initialiser_func, context=context)
