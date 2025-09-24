# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

all: spline_eval_funcs$(SO_EXT) cubic_uniform_spline_eval_funcs$(SO_EXT) splines$(SO_EXT) accelerated_spline_interpolators$(SO_EXT) sll_m_spline_matrix_periodic_banded$(SO_EXT)

.ACC.$(ACC):
	@rm -f .ACC.*
	@touch $@

spline_eval_funcs$(SO_EXT): $(NAME_PREFIX)spline_eval_funcs.py .ACC.$(ACC)
ifneq ($(ACC), pythran)
	$(TOOL) $< $(TOOL_FLAGS)
else
	$(TOOL) $< $(TOOL_FLAGS) -o $@
endif

cubic_uniform_spline_eval_funcs$(SO_EXT): $(NAME_PREFIX)cubic_uniform_spline_eval_funcs.py .ACC.$(ACC)
ifneq ($(ACC), pythran)
	$(TOOL) $< $(TOOL_FLAGS)
else
	$(TOOL) $< $(TOOL_FLAGS) -o $@
endif

splines$(SO_EXT): $(NAME_PREFIX)splines.py .ACC.$(ACC) spline_eval_funcs$(SO_EXT) cubic_uniform_spline_eval_funcs$(SO_EXT)
ifneq ($(ACC), pythran)
	$(TOOL) $< $(TOOL_FLAGS)
else
	$(TOOL) $< $(TOOL_FLAGS) -o $@
endif

sll_m_spline_matrix_periodic_banded$(SO_EXT): sll_m_spline_matrix_periodic_banded.pyi
	pyccel-wrap $< $(TOOL_FLAGS)

accelerated_spline_interpolators$(SO_EXT): $(NAME_PREFIX)accelerated_spline_interpolators.py .ACC.$(ACC) splines$(SO_EXT) sll_m_spline_matrix_periodic_banded$(SO_EXT)
ifneq ($(ACC), pythran)
	$(TOOL) $< $(TOOL_FLAGS)
else
	$(TOOL) $< $(TOOL_FLAGS) -o $@
endif


clean:
	rm -f *.o *.so *.mod .ACC.pycc .lock_acquisition.lock
	pyccel-clean

