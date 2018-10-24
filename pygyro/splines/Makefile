# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

#spline_eval_funcs: spline_eval_funcs.so
#
#spline_eval_funcs.so: spline_eval_funcs$(_OPT).f90 mod_context_1$(_OPT).o
#	CC=$(CC) FC=$(FC) f2py -c --opt="$(FC_FLAGS)" -m spline_eval_funcs $^

spline_eval_funcs: spline_eval_funcs$(_OPT).f90 mod_context_1$(_OPT).o
	CC=$(CC) FC=$(FC) f2py -c --opt="$(FC_FLAGS)" -m spline_eval_funcs --fcompiler=$(FF_COMP) $^

mod_context_1$(_OPT).o: mod_context_1$(_OPT).f90
	$(FC) $(FC_FLAGS) -c $^ -o $@

clean:
	rm -f *.o *.so *.mod
