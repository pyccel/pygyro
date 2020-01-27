# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

numba: numba_accelerated_advection_steps

pycc: pycc_accelerated_advection_steps

pythran: pythran_accelerated_advection_steps

clean:
	rm -f *.o *.so *.mod
ifeq ($(PYCC_GEN), 1)
	rm -f *.f90
endif


#----------------------------------------------------------
# Pythran targets
#----------------------------------------------------------

pythran_accelerated_advection_steps: pythran_deps/pythran_accelerated_advection_steps.py
	pythran $^ -o accelerated_advection_steps.so

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


