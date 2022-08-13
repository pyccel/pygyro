# Export variables to children Makefiles
export EXPORTED_VARS $(EXPORTED_VARS)

# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean splines initialisation advection arakawa


#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

$(ACC):
	$(MAKE) -C splines
	$(MAKE) -C initialisation
	$(MAKE) -C advection
	$(MAKE) -C arakawa

splines:
	$(MAKE) -C $@

initialisation:
	$(MAKE) -C $@

advection: $(ACC)

clean:
	$(MAKE) -C splines $@
	$(MAKE) -C initialisation $@
	$(MAKE) -C advection $@
	$(MAKE) -C arakawa $@
