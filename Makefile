#----------------------------------------------------------
# DEFAULT MAKE FLAGS
#----------------------------------------------------------

# Acceleration method? [none|numba|pycc|cython]
ACC := pycc

# Use GNU or intel compilers? [gnu|intel]
COMP := gnu

# Use pyccel to generate files? [1|0]
PYCC_GEN := 0

# Use cython to generate files? [1|0]
CYTHON_GEN := 1

# Use C++ files instead of C files for cython
USE_CPP := 1

#----------------------------------------------------------
# Compiler options
#----------------------------------------------------------

ifeq ($(COMP), gnu)
	CC        := gcc
	CXX       := g++
	CXX_FLAGS := -Wall -O3 -fPIC
	FC        := gfortran
	FC_FLAGS  := -Wall -O3 -fPIC -fstack-arrays
        FF_COMP   := gnu95
else \
ifeq ($(COMP), intel)
	CC        := icc
	CXX       := icc
	CXX_FLAGS := -O3 -xHost -ip -fpic
	FC        := ifort
	FC_FLAGS  := -O3 -xHost -ip -fpic
        FF_COMP   := intelem
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
			ifeq ($(ACC), cython)
				TYPE := cython
			else
				TYPE := clean
			endif
		endif
	endif
endif

ifeq ($(USE_CPP),1)
	CYTH_CC := $(CXX)
else
	CYTH_CC := $(CC)
endif

#----------------------------------------------------------
# Export all relevant variables to children Makefiles
#----------------------------------------------------------

EXPORTED_VARS = CC CXX CXX_FLAGS FC FC_FLAGS FF_COMP PYCC_GEN PYTHON ACC USE_CPP CYTH_CC CYTHON_GEN
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

all:
	$(MAKE) -C pygyro $(TYPE)

$(ALL): 
	$(MAKE) -C pygyro $@

pyccel:
	echo $(TYPE)
	$(MAKE) -C pygyro pycc

numba:
	$(MAKE) -C pygyro $@

cython:
	$(MAKE) -C pygyro $@

clean:
	$(MAKE) -C pygyro $@

help:
	@echo "Available targets: $(ALL)"
