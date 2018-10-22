#----------------------------------------------------------
# DEFAULT MAKE FLAGS
#----------------------------------------------------------

# Use GNU or intel compilers? [gnu|intel]
COMP := gnu

# Use manually optimized Fortran files? [1|0]
MOPT := 1

#----------------------------------------------------------
# Compiler options
#----------------------------------------------------------

ifeq ($(COMP), gnu)
	CC       := gcc
	FC       := gfortran
	FC_FLAGS := -Wall -O3 -fPIC -fstack-arrays
else \
ifeq ($(COMP), intel)
	CC       := icc
	FC       := ifort
	FC_FLAGS := -O3 -xHost -ip -fpic
endif

#----------------------------------------------------------
# Pyccel-generated Fortran files:
# use originals or manually-optimized version?
#----------------------------------------------------------

ifeq ($(MOPT), 1)
	_OPT := _opt
else
	_OPT :=
endif

#----------------------------------------------------------
# Export all relevant variables to children Makefiles
#----------------------------------------------------------

EXPORTED_VARS = CC FC FC_FLAGS _OPT
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
	initialiser_func  \
	accelerated_advection_steps

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
