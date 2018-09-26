# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

#initialiser_func: initaliser_func.so
#
#initialiser_func.so: initialiser_func.f90 mod_initialiser_funcs.o
#	CC=$(CC) FC=$(FC) f2py -c --opt="$(FC_FLAGS)" -m initialiser_func $^

initialiser_func: initialiser_func.f90 mod_initialiser_funcs.o
	CC=$(CC) FC=$(FC) f2py -c --opt="$(FC_FLAGS)" -m initialiser_func $^

mod_initialiser_funcs.o: mod_initialiser_funcs.f90
	$(FC) $(FC_FLAGS) -c $^ -o $@

clean:
	rm -f *.o *.so *.mod
