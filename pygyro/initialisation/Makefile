# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean cython_initialiser_func

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

numba: numba_initialiser_func

pycc: pycc_initialiser_func mod_initialiser_funcs.o

cython: cython_initialiser_func

clean:
	rm -f *.o *.so *.mod
ifeq ($(PYCC_GEN), 1)
	rm -f *.f90
endif
ifeq ($(CYTHON_GEN), 1)
ifeq ($(USE_CPP), 1)
	rm -f *.cpp
else
	rm -f *.c
endif
endif

#----------------------------------------------------------
# Numba targets
#----------------------------------------------------------

numba_initialiser_func:
	$(PYTHON) numba_mod_initialiser_funcs.py

#----------------------------------------------------------
# Pyccel targets
#----------------------------------------------------------

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


#----------------------------------------------------------
# Cython targets
#----------------------------------------------------------

cython_initialiser_func: initialiser_func.so mod_initialiser_funcs.so

mod_initialiser_funcs.so: cython_mod_initialiser_funcs.o
	$(CYTH_CC) $(CXX_FLAGS) -shared $^ -o $@

initialiser_func.so: cython_initialiser_func.o
	$(CYTH_CC) $(CXX_FLAGS) -shared $^ -o $@

ifeq ($(CYTHON_GEN),1)

ifeq ($(USE_CPP),1)
initialiser_func.cpp: initialiser_func.pyx
	cython -3 $^ --cplus -o $@

mod_initialiser_funcs.cpp: mod_initialiser_funcs.pyx
	cython -3 $^ --cplus -o $@
else
initialiser_func.c: initialiser_func.pyx
	cython -3 $^ -o $@

mod_initialiser_funcs.c: mod_initialiser_funcs.pyx
	cython -3 $^ -o $@

endif
endif


ifeq ($(USE_CPP),1)
cython_mod_initialiser_funcs.o: mod_initialiser_funcs.cpp
else
cython_mod_initialiser_funcs.o: mod_initialiser_funcs.c
endif
	$(CYTH_CC) $(CXX_FLAGS) -I/usr/include/python3.6m -c $^ -o $@

ifeq ($(USE_CPP),1)
cython_initialiser_func.o: initialiser_func.cpp
else
cython_initialiser_func.o: initialiser_func.c
endif
	$(CYTH_CC) $(CXX_FLAGS) -I/usr/include/python3.6m -c $^ -o $@
