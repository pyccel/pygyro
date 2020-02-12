# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

numba: numba_spline_eval_funcs

pycc: pycc_spline_eval_funcs

pythran: pythran_spline_eval_funcs

clean:
	rm -f *.o *.so *.mod
	rm -f __pyccel__/*.o __pyccel__/*.so __pyccel__/*.mod
ifeq ($(PYCC_GEN), 1)
	rm -f *.f90
	rm -rf __pyccel__
endif

#----------------------------------------------------------
# Numba targets
#----------------------------------------------------------

numba_spline_eval_funcs:
	$(PYTHON) numba_spline_eval_funcs.py

#----------------------------------------------------------
# Pythran targets
#----------------------------------------------------------

pythran_spline_eval_funcs: pythran_spline_eval_funcs.py
	pythran $(PYTHRAN_FLAGS) $^ -o spline_eval_funcs.so


#----------------------------------------------------------
# Pyccel targets
#----------------------------------------------------------

ifeq ($(PYCC_GEN), 1)

pycc_spline_eval_funcs: spline_eval_funcs.py
	pyccel $^ --fflags ' $(FC_FLAGS)'

else

pycc_spline_eval_funcs: __pyccel__/f2py_spline_eval_funcs.f90 __pyccel__/spline_eval_funcs.o
	CC=$(CC) FC=$(FC) f2py -c --opt="$(FC_FLAGS)" -m spline_eval_funcs --fcompiler=$(FF_COMP) $^

__pyccel__/spline_eval_funcs.o: __pyccel__/spline_eval_funcs.f90
	cd __pyccel__; $(FC) $(FC_FLAGS) -c spline_eval_funcs.f90 -o spline_eval_funcs.o

endif
