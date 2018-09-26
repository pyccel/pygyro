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

clean:
	$(MAKE) -C splines $@
