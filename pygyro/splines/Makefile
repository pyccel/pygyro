# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

numba: numba_spline_eval_funcs

numba_spline_eval_funcs:
	$(PYTHON) numba_spline_eval_funcs.py

pyccel_generation: mod_spline_eval_funcs.f90 mod_context_1.f90

pyccel: pyccel_spline_eval_funcs

mod_spline_eval_funcs.f90:
	pyccel mod_spline_eval_funcs.py --fflags ' $(FC_FLAGS)'

mod_context_1.f90:
	pyccel mod_context_1.py --fflags ' $(FC_FLAGS)'

pyccel_spline_eval_funcs: spline_eval_funcs$(_OPT).f90 mod_context_1$(_OPT).o
	CC=$(CC) FC=$(FC) f2py -c --opt="$(FC_FLAGS)" -m spline_eval_funcs --fcompiler=$(FF_COMP) $^

mod_context_1$(_OPT).o: mod_context_1$(_OPT).f90
	$(FC) $(FC_FLAGS) -c $^ -o $@

clean:
	rm -f *.o *.so *.mod
