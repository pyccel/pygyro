#----------------------------------------------------------
# Compiler options
#----------------------------------------------------------

CC       := gcc
FC       := gfortran
FC_FLAGS := -Wall -O3 -fPIC -fstack-arrays

#----------------------------------------------------------
# Export all relevant variables to children Makefiles
#----------------------------------------------------------

EXPORTED_VARS = CC FC FC_FLAGS
export EXPORTED_VARS $(EXPORTED_VARS)

#----------------------------------------------------------
# Make options
#----------------------------------------------------------

# Clear unfinished targets
.DELETE_ON_ERROR:

# Define the phony targets
.PHONY: all clean help

# List of main targets
ALL = \
	spline_eval_funcs \
	initialiser_func

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

all: $(ALL)

$(ALL):
	$(MAKE) -C pygyro $@

clean:
	$(MAKE) -C pygyro $@

help:
	@echo "Available targets: $(ALL)"
