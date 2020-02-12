# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

all: spline_eval_funcs$(SO_EXT)

.ACC.$(ACC):
	@rm -f .ACC.*
	@touch $@

ifeq ($(ACC)$(PYCC_GEN), pycc0)

spline_eval_funcs$(SO_EXT): __pyccel__/f2py_spline_eval_funcs.f90 __pyccel__/spline_eval_funcs.o .ACC.$(ACC)
	CC=$(CC) FC=$(FC) f2py -c --opt="$(FC_FLAGS)" -m spline_eval_funcs --fcompiler=$(FF_COMP) __pyccel__/f2py_spline_eval_funcs.f90 __pyccel__/spline_eval_funcs.o

__pyccel__/spline_eval_funcs.o: __pyccel__/spline_eval_funcs.f90
	cd __pyccel__; $(FC) $(FC_FLAGS) -c spline_eval_funcs.f90 -o spline_eval_funcs.o

else

spline_eval_funcs$(SO_EXT): $(NAME_PREFIX)spline_eval_funcs.py .ACC.$(ACC)
ifneq ($(ACC), pythran)
	$(TOOL) $< $(TOOL_FLAGS)
else
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

