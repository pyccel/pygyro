# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean cython_accelerated_advection_steps

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

numba: numba_accelerated_advection_steps

pycc: pycc_accelerated_advection_steps

cython: cython_accelerated_advection_steps

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

numba_accelerated_advection_steps:
	$(PYTHON) numba_accelerated_advection_steps.py

#----------------------------------------------------------
# Pyccel targets
#----------------------------------------------------------

pycc_accelerated_advection_steps: accelerated_advection_steps.f90 ../initialisation/mod_initialiser_funcs.o ../splines/mod_spline_eval_funcs.o ../splines/mod_context_1.o
	CC=$(CC) FC=$(FC) f2py -c --opt="$(FC_FLAGS)" -m accelerated_advection_steps --fcompiler=$(FF_COMP) $^ -I../initialisation -I../splines


ifeq ($(PYCC_GEN), 1)
accelerated_advection_steps.f90: accelerated_advection_steps.py
	pyccel -f $^
endif



#----------------------------------------------------------
# Cython targets
#----------------------------------------------------------

cython_accelerated_advection_steps: accelerated_advection_steps.so

accelerated_advection_steps.so: cython_accelerated_advection_steps.o
	$(CYTH_CC) $(CXX_FLAGS) -shared $^ -o $@

ifeq ($(CYTHON_GEN),1)

ifeq ($(USE_CPP),1)
accelerated_advection_steps.cpp: accelerated_advection_steps.pyx
	cython -3 $^ --cplus -o $@
else
accelerated_advection_steps.cpp: accelerated_advection_steps.pyx
	cython -3 $^ -o $@
endif
endif

ifeq ($(USE_CPP),1)
cython_accelerated_advection_steps.o: accelerated_advection_steps.cpp
else
cython_accelerated_advection_steps.o: accelerated_advection_steps.c
endif
	$(CYTH_CC) $(CXX_FLAGS) -I/usr/include/python3.6m -c $^ -o $@


