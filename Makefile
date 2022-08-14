#----------------------------------------------------------
# DEFAULT MAKE FLAGS
#----------------------------------------------------------

# Acceleration method? [none|numba|pycc|pythran]
ACC := pycc

# Use GNU or intel compilers? [GNU|intel]
COMP := GNU

# Target language
LANGUAGE := fortran

#PYTHRAN_FLAGS := -DUSE_XSIMD -fopenmp -march=native
PYTHRAN_FLAGS := 

#----------------------------------------------------------
# Compiler options
#----------------------------------------------------------

ifeq ($(COMP), GNU)
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

PYTHON := python3

SO_EXT := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

#----------------------------------------------------------
# Defaut command:
# use pure python, pyccel, numba?
#----------------------------------------------------------

ifeq ($(ACC), pycc)
	TOOL := pyccel
	TOOL_FLAGS := --compiler=$(COMP) --flags ' $(FC_FLAGS)' --language=$(LANGUAGE)
	NAME_PREFIX := 
else
	ifeq ($(ACC), numba)
		TOOL := $(PYTHON)
		TOOL_FLAGS :=
		NAME_PREFIX := numba_
	else
		ifeq ($(ACC), pythran)
			TOOL := pythran
			TOOL_FLAGS = $(PYTHRAN_FLAGS)
			NAME_PREFIX := pythran_
		else
			TOOL := pyccel
			TOOL_FLAGS := --flags ' $(FC_FLAGS)' --language= --language=$(LANGUAGE)
			NAME_PREFIX := 
		endif
	endif
endif

#----------------------------------------------------------
# Export all relevant variables to children Makefiles
#----------------------------------------------------------

EXPORTED_VARS = CC FC FC_FLAGS FF_COMP PYCC_GEN PYTHON ACC PYTHRAN_FLAGS SO_EXT TOOL TOOL_FLAGS NAME_PREFIX
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
	splines \
	initialisation \
	advection \
	arakawa

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

all:
	$(MAKE) -C pygyro $(TYPE)

$(ALL): 
	$(MAKE) -C pygyro $@

pycc:
	$(MAKE) -C pygyro $@

numba:
	$(MAKE) -C pygyro $@

pythran:
	$(MAKE) -C pygyro $@

clean:
	$(MAKE) -C pygyro $@

help:
	@echo "Available targets: $(ALL)"
