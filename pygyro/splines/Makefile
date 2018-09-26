# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

#spline_eval_funcs: spline_eval_funcs.so
#
#spline_eval_funcs.so: spline_eval_funcs_opt.f90 mod_context_1_opt.o
#	CC=$(CC) FC=$(FC) f2py -c --opt="$(FC_FLAGS)" -m spline_eval_funcs $^

spline_eval_funcs: spline_eval_funcs_opt.f90 mod_context_1_opt.o
	CC=$(CC) FC=$(FC) f2py -c --opt="$(FC_FLAGS)" -m spline_eval_funcs $^

mod_context_1_opt.o: mod_context_1_opt.f90
	$(FC) $(FC_FLAGS) -c $^ -o $@

clean:
	rm -f *.o *.so *.mod
