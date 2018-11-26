# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

#initialiser_func: initialiser_func.so
#
#initialiser_func.so: initialiser_func.f90 mod_initialiser_funcs$(_OPT).o
#	CC=$(CC) FC=$(FC) f2py -c --opt="$(FC_FLAGS)" -m initialiser_func $^

pyccel_generation: mod_initialiser_funcs.f90

mod_initialiser_funcs.f90:
	pyccel mod_initialiser_funcs.py --fflags ' $(FC_FLAGS)'

pyccel: mod_initialiser_funcs$(_OPT).o initialiser_func

initialiser_func: initialiser_func.f90 mod_initialiser_funcs$(_OPT).o
	CC=$(CC) FC=$(FC) f2py -c --opt="$(FC_FLAGS)" -m initialiser_func --fcompiler=$(FF_COMP) $^

mod_initialiser_funcs$(_OPT).o: mod_initialiser_funcs$(_OPT).f90
	$(FC) $(FC_FLAGS) -c $^ -o $@

clean:
	rm -f *.o *.so *.mod
