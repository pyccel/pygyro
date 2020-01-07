# Export variables to children Makefiles
export EXPORTED_VARS $(EXPORTED_VARS)

# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean


#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

numba:
	$(MAKE) -C splines $@
	$(MAKE) -C advection $@
	$(MAKE) -C initialisation $@

pycc:
	echo "pycc " $@
	$(MAKE) -C splines $@
	$(MAKE) -C initialisation $@
	$(MAKE) -C advection $@

$(ACC)_spline_eval_funcs:
	$(MAKE) -C splines $@

$(ACC)_initialiser_func:
	$(MAKE) -C initialisation $@

$(ACC)_accelerated_advection_steps: pycc

clean:
	$(MAKE) -C splines $@
	$(MAKE) -C initialisation $@
	$(MAKE) -C advection $@
