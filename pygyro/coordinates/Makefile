# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

numba: #numba_accelerated_advection_steps

#numba_accelerated_advection_steps:
#	$(PYTHON) numba_accelerated_advection_steps.py

pyccel_generation: pyccel pyccel_coordinates.f90

pyccel: coordinates

pyccel_coordinates.f90: coordinates.py
	pyccel -t coordinates.py

coordinates: coordinates.f90
	CC=$(CC) FC=$(FC) f2py -c --opt="$(FC_FLAGS)" -m coordinates --fcompiler=$(FF_COMP) $^ 

clean:
	rm -f *.o *.so *.mod
