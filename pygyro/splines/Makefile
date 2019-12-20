# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean numba_spline_eval_funcs pycc_spline_eval_funcs cython_spline_eval_funcs

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

numba: numba_spline_eval_funcs

pycc: pycc_spline_eval_funcs mod_spline_eval_funcs.o

cython: cython_spline_eval_funcs

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



#----------------------------------------------------------
# Cython targets
#----------------------------------------------------------

cython_spline_eval_funcs: mod_context_1.so spline_eval_funcs.so mod_spline_eval_funcs.so

mod_context_1.so: cython_mod_context_1.o
	$(CYTH_CC) $(CXX_FLAGS) -shared $^ -o $@

spline_eval_funcs.so: cython_spline_eval_funcs.o
	$(CYTH_CC) $(CXX_FLAGS) -shared $^ -o $@

mod_spline_eval_funcs.so: cython_mod_spline_eval_funcs.o
	$(CYTH_CC) $(CXX_FLAGS) -shared $^ -o $@

ifeq ($(CYTHON_GEN),1)

ifeq ($(USE_CPP),1)
mod_context_1.cpp: mod_context_1.pyx
	cython -3 $^ --cplus -o $@

spline_eval_funcs.cpp: spline_eval_funcs.pyx
	cython -3 $^ --cplus -o $@

mod_spline_eval_funcs.cpp: mod_spline_eval_funcs.pyx
	cython -3 $^ --cplus -o $@
else
mod_context_1.c: mod_context_1.pyx
	cython -3 $^ -o $@

mod_spline_eval_funcs.c: mod_spline_eval_funcs.pyx
	cython -3 $^ -o $@

mod_spline_eval_funcs.c: mod_spline_eval_funcs.pyx
	cython -3 $^ -o $@
endif
endif

ifeq ($(USE_CPP),1)
cython_mod_context_1.o: mod_context_1.cpp
else
cython_mod_context_1.o: mod_context_1.c
endif
	$(CYTH_CC) $(CXX_FLAGS) -I/usr/include/python3.6m -c $^ -o $@


ifeq ($(USE_CPP),1)
cython_spline_eval_funcs.o: spline_eval_funcs.cpp
else
cython_spline_eval_funcs.o: spline_eval_funcs.c
endif
	$(CYTH_CC) $(CXX_FLAGS) -I/usr/include/python3.6m -c $^ -o $@


ifeq ($(USE_CPP),1)
cython_mod_spline_eval_funcs.o: mod_spline_eval_funcs.cpp
else
cython_mod_spline_eval_funcs.o: mod_spline_eval_funcs.c
endif
	$(CYTH_CC) $(CXX_FLAGS) -I/usr/include/python3.6m -c $^ -o $@
