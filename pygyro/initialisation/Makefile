# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

all: initialiser_funcs$(SO_EXT)

.ACC.$(ACC):
	@rm -f .ACC.*
	@touch $@

ifeq ($(ACC)$(PYCC_GEN), pycc0)

initialiser_funcs$(SO_EXT): __pyccel__/f2py_initialiser_funcs.f90 __pyccel__/initialiser_funcs.o .ACC.$(ACC)
	CC=$(CC) FC=$(FC) f2py -c --opt="$(FC_FLAGS)" -m initialiser_funcs --fcompiler=$(FF_COMP) __pyccel__/f2py_initialiser_funcs.f90 __pyccel__/initialiser_funcs.o

__pyccel__/initialiser_funcs.o: __pyccel__/initialiser_funcs.f90
	cd __pyccel__; $(FC) $(FC_FLAGS) -c initialiser_funcs.f90 -o initialiser_funcs.o

else

initialiser_funcs$(SO_EXT): $(NAME_PREFIX)initialiser_funcs.py .ACC.$(ACC)
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

