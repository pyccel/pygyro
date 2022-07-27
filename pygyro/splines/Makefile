# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

all: spline_eval_funcs$(SO_EXT)

.ACC.$(ACC):
	@rm -f .ACC.*
	@touch $@

spline_eval_funcs$(SO_EXT): $(NAME_PREFIX)spline_eval_funcs.py .ACC.$(ACC)
ifneq ($(ACC), pythran)
	$(TOOL) $< $(TOOL_FLAGS)
else
	$(TOOL) $< $(TOOL_FLAGS) -o $@
endif


clean:
	rm -f *.o *.so *.mod .ACC.pycc .lock_acquisition.lock
	pyccel-clean

