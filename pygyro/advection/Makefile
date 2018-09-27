# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

#accelerated_advection_steps: accelerated_advection_steps.so
#
#accelerated_advection_steps.so: accelerated_advection_steps_opt.f90 ../advection/mod_initialiser_funcs_opt.o ../splines/mod_spline_eval_funcs_opt.o
#	CC=$(CC) FC=$(FC) f2py -c --opt="$(FC_FLAGS)" -m accelerated_advection_steps $^

accelerated_advection_steps: accelerated_advection_steps_opt.f90 ../initialisation/mod_initialiser_funcs_opt.o mod_spline_eval_funcs_opt.o
	CC=$(CC) FC=$(FC) f2py -c --opt="$(FC_FLAGS)" -m accelerated_advection_steps $^ -I../initialisation -I../splines

mod_spline_eval_funcs_opt.o: ../splines/mod_spline_eval_funcs_opt.f90
	$(FC) $(FC_FLAGS) -c $^ -o $@

clean:
	rm -f *.o *.so *.mod
