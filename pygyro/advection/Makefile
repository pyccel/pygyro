# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

numba: accelerated_advection_steps

accelerated_advection_steps:
	$(PYTHON) numba_accelerated_advection_steps.py

clean:
	rm -f *.so
