# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

numba: spline_eval_funcs

spline_eval_funcs:
	$(PYTHON) numba_spline_eval_funcs.py

clean:
	rm -f *.so
