# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

#accelerated_advection_steps: accelerated_advection_steps.so
#
#accelerated_advection_steps.so: accelerated_advection_steps$(_OPT).f90 ../advection/mod_initialiser_funcs$(_OPT).o ../splines/mod_spline_eval_funcs$(_OPT).o
#	CC=$(CC) FC=$(FC) f2py -c --opt="$(FC_FLAGS)" -m accelerated_advection_steps $^

accelerated_advection_steps: accelerated_advection_steps$(_OPT).f90 ../initialisation/mod_initialiser_funcs$(_OPT).o mod_spline_eval_funcs$(_OPT).o
	CC=$(CC) FC=$(FC) f2py -c --opt="$(FC_FLAGS)" -m accelerated_advection_steps --fcompiler=$(FF_COMP) $^ -I../initialisation -I../splines

mod_spline_eval_funcs$(_OPT).o: ../splines/mod_spline_eval_funcs$(_OPT).f90
	$(FC) $(FC_FLAGS) -c $^ -o $@

clean:
	rm -f *.o *.so *.mod
