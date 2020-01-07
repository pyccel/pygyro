# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

numba: numba_spline_eval_funcs

pycc: pycc_spline_eval_funcs mod_spline_eval_funcs.o

clean:
	rm -f *.o *.so *.mod
ifeq ($(PYCC_GEN), 1)
	rm -f *.f90
endif

#----------------------------------------------------------
# Numba targets
#----------------------------------------------------------

numba_spline_eval_funcs:
	$(PYTHON) numba_spline_eval_funcs.py

mod_spline_eval_funcs.o: mod_context_1.o mod_spline_eval_funcs.f90
	$(FC) $(FC_FLAGS) -c $^ -o $@


#----------------------------------------------------------
# Pyccel targets
#----------------------------------------------------------

pycc_spline_eval_funcs: spline_eval_funcs.f90 mod_context_1.o
	CC=$(CC) FC=$(FC) f2py -c --opt="$(FC_FLAGS)" -m spline_eval_funcs --fcompiler=$(FF_COMP) $^

ifeq ($(PYCC_GEN), 1)

mod_context_1.o: mod_context_1.py
	pyccel $^ --fflags ' $(FC_FLAGS)'

mod_spline_eval_funcs.f90: mod_spline_eval_funcs.py
	pyccel $^ --fflags ' $(FC_FLAGS)' -t

spline_eval_funcs.f90: spline_eval_funcs.py
	pyccel -f $^

else

mod_context_1.o: mod_context_1.f90
	$(FC) $(FC_FLAGS) -c $^ -o $@

endif
