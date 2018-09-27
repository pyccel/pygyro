# Export variables to children Makefiles
export EXPORTED_VARS $(EXPORTED_VARS)

# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

spline_eval_funcs:
	$(MAKE) -C splines $@

initialiser_func:
	$(MAKE) -C initialisation $@

accelerated_advection_steps:
	$(MAKE) -C advection $@

clean:
	$(MAKE) -C splines $@
	$(MAKE) -C initialisation $@
	$(MAKE) -C advection $@
