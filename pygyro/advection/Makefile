# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

all: accelerated_advection_steps$(SO_EXT)

.ACC.$(ACC):
	@rm -f .ACC.*
	@touch $@

ifeq ($(ACC)$(PYCC_GEN), pycc0)

DEPS := __pyccel__/f2py_accelerated_advection_steps.f90 __pyccel__/accelerated_advection_steps.o ../initialisation/__pyccel__/initialiser_funcs.o ../splines/__pyccel__/spline_eval_funcs.o

accelerated_advection_steps$(SO_EXT): $(DEPS) .ACC.$(ACC)
	CC=$(CC) FC=$(FC) f2py -c --opt="$(FC_FLAGS)" -m accelerated_advection_steps --fcompiler=$(FF_COMP) $(DEPS) -I../initialisation/__pyccel__ -I../splines/__pyccel__

__pyccel__/accelerated_advection_steps.o: __pyccel__/accelerated_advection_steps.f90 ../splines/__pyccel__/spline_eval_funcs.o
	cd __pyccel__; $(FC) $(FC_FLAGS) -c accelerated_advection_steps.f90 -o accelerated_advection_steps.o -I../../initialisation/__pyccel__ -I../../splines/__pyccel__

else

ifneq ($(ACC), pythran)
accelerated_advection_steps$(SO_EXT): $(NAME_PREFIX)accelerated_advection_steps.py .ACC.$(ACC)
	$(TOOL) $< $(TOOL_FLAGS)
else
accelerated_advection_steps$(SO_EXT): pythran_deps/$(NAME_PREFIX)accelerated_advection_steps.py .ACC.$(ACC)
	$(TOOL) $< $(TOOL_FLAGS) -o $@
endif
endif


clean:
	rm -f *.o *.so *.mod
	rm -f __pyccel__/*.o __pyccel__/*.so __pyccel__/*.mod
ifeq ($(PYCC_GEN), 1)
	rm -f *.f90
	rm -rf __pyccel__
endif

