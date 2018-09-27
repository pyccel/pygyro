# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

#initialiser_func: initialiser_func.so
#
#initialiser_func.so: initialiser_func.f90 mod_initialiser_funcs_opt.o
#	CC=$(CC) FC=$(FC) f2py -c --opt="$(FC_FLAGS)" -m initialiser_func $^

initialiser_func: initialiser_func.f90 mod_initialiser_funcs_opt.o
	CC=$(CC) FC=$(FC) f2py -c --opt="$(FC_FLAGS)" -m initialiser_func $^

mod_initialiser_funcs_opt.o: mod_initialiser_funcs_opt.f90
	$(FC) $(FC_FLAGS) -c $^ -o $@

clean:
	rm -f *.o *.so *.mod
