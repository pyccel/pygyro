#----------------------------------------------------------
# DEFAULT MAKE FLAGS
#----------------------------------------------------------

# Acceleration method? [none|numba|pycc]
ACC := pycc

# Use GNU or intel compilers? [gnu|intel]
COMP := gnu

# Use manually optimized Fortran files? [1|0]
MOPT := 0

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
# Pyccel-generated Fortran files:
# use originals or manually-optimized version?
#----------------------------------------------------------

ifeq ($(MOPT), 1)
	_OPT := _opt
else
	_OPT :=
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
			echo "Option not recognised!"
		endif
	endif
endif

#----------------------------------------------------------
# Export all relevant variables to children Makefiles
#----------------------------------------------------------

EXPORTED_VARS = CC FC FC_FLAGS FF_COMP _OPT PYCC_GEN
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

$(ALL): $(TYPE)

pyccel_generation:
	$(MAKE) -C pygyro $@

ifeq ($(PYCC_GEN), 1)
pycc: pyccel_generation
	$(PYTHON) moduleGenerator.py
else
pycc:
endif
	echo $(PYCC_GEN)
	$(MAKE) -C pygyro pyccel

clean:
	$(MAKE) -C pygyro $@

help:
	@echo "Available targets: $(ALL)"
