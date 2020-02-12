# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

numba: numba_initialiser_func

pycc: pycc_initialiser_funcs

pythran: pythran_initialiser_funcs

clean:
	rm -f *.o *.so *.mod
	rm -f __pyccel__/*.o __pyccel__/*.so __pyccel__/*.mod
ifeq ($(PYCC_GEN), 1)
	rm -f *.f90
	rm -rf __pyccel__
endif


#----------------------------------------------------------
# Pythran targets
#----------------------------------------------------------

pythran_initialiser_funcs: pythran_initialiser_funcs.py
	pythran $(PYTHRAN_FLAGS) $^ -o initialiser_func.so

#----------------------------------------------------------
# Numba targets
#----------------------------------------------------------

numba_initialiser_func: numba_mod_initialiser_funcs.py
	$(PYTHON) $^

#----------------------------------------------------------
# Pyccel targets
#----------------------------------------------------------



ifeq ($(PYCC_GEN), 1)

pycc_initialiser_funcs: initialiser_funcs.py
	pyccel $^ --fflags ' $(FC_FLAGS)'

else

pycc_initialiser_funcs: __pyccel__/initialiser_funcs.o __pyccel__/f2py_initialiser_funcs.f90
	CC=$(CC) FC=$(FC) f2py -c --opt="$(FC_FLAGS)" -m initialiser_funcs --fcompiler=$(FF_COMP) $^

__pyccel__/initialiser_funcs.o: __pyccel__/initialiser_funcs.f90
	cd __pyccel__; $(FC) $(FC_FLAGS) -c initialiser_funcs.f90 -o initialiser_funcs.o

endif

