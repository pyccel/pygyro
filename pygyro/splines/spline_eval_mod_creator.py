from pyccel.epyccel import ContextPyccel
from pyccel.epyccel import epyccel
from pyccel.decorators import types

import spline_eval_funcs
from mod_context_1 import find_span,basis_funs,basis_funs_1st_der

context = ContextPyccel(name='context_1')
context.insert_function(find_span, ['double[:]','int','double'])
context.insert_function(basis_funs, ['double[:]','int','double','int','double[:]'])
context.insert_function(basis_funs_1st_der, ['double[:]','int','double','int','double[:]'])

spline_eval_funcs = epyccel(spline_eval_funcs, context=context)
