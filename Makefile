#----------------------------------------------------------
# DEFAULT MAKE FLAGS
#----------------------------------------------------------

# Acceleration method? [none|numba|pycc]
ACC := pycc

# Use GNU or intel compilers? [gnu|intel]
COMP := gnu

# Use pyccel to generate files? [1|0]
PYCC_GEN := 0

#----------------------------------------------------------
# Compiler options
#----------------------------------------------------------

ifeq ($(COMP), gnu)
	CC       := gcc
	FC       := gfortran
	FC_FLAGS := -Wall -O3 -fPIC -fstack-arrays
        FF_COMP  := gnu95
else \
ifeq ($(COMP), intel)
	CC       := icc
	FC       := ifort
	FC_FLAGS := -O3 -xHost -ip -fpic
        FF_COMP  := intelem
endif

python_version_full := $(wordlist 2,4,$(subst ., ,$(shell python --version 2>&1)))
python_version_major := $(word 1,${python_version_full})
ifeq ($(python_version_major),3)
	PYTHON := python
else
	PYTHON := python3
endif

#----------------------------------------------------------
# Defaut command:
# use pure python, pyccel, numba?
#----------------------------------------------------------

ifeq ($(ACC), none)
	TYPE := clean
else
	ifeq ($(ACC), pycc)
		TYPE := pycc
	else
		ifeq ($(ACC), numba)
			TYPE := numba
		else
			TYPE := clean
		endif
	endif
endif

#----------------------------------------------------------
# Export all relevant variables to children Makefiles
#----------------------------------------------------------

EXPORTED_VARS = CC FC FC_FLAGS FF_COMP PYCC_GEN PYTHON ACC
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
	$(TYPE)_spline_eval_funcs \
	$(TYPE)_initialiser_func  \
	$(TYPE)_accelerated_advection_steps

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

all: $(ALL)

$(ALL): 
	$(MAKE) -C pygyro $(TYPE)

pycc:
	$(MAKE) -C pygyro $@

numba:
	$(MAKE) -C pygyro $@

clean:
	$(MAKE) -C pygyro $@

help:
	@echo "Available targets: $(ALL)"
