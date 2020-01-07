# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

numba: numba_initialiser_func

pycc: pycc_initialiser_func mod_initialiser_funcs.o

clean:
	rm -f *.o *.so *.mod
ifeq ($(PYCC_GEN), 1)
	rm -f *.f90
endif

#----------------------------------------------------------
# Pyccel targets
#----------------------------------------------------------

numba_initialiser_func:
	$(PYTHON) numba_mod_initialiser_funcs.py

pycc_initialiser_func: initialiser_func.f90 mod_initialiser_funcs.o
	CC=$(CC) FC=$(FC) f2py -c --opt="$(FC_FLAGS)" -m initialiser_func --fcompiler=$(FF_COMP) $^


ifeq ($(PYCC_GEN), 1)

mod_initialiser_funcs.o: mod_initialiser_funcs.py
	pyccel mod_initialiser_funcs.py --fflags ' $(FC_FLAGS)'

initialiser_func.f90: initialiser_func.py
	pyccel initialiser_func.py -f --fflags ' $(FC_FLAGS)'

else

mod_initialiser_funcs.o: mod_initialiser_funcs.f90
	$(FC) $(FC_FLAGS) -c $^ -o $@

endif

