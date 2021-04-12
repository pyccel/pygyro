# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

all: accelerated_advection_steps$(SO_EXT)

.ACC.$(ACC):
	@rm -f .ACC.*
	@touch $@

DEPS := ../initialisation/$(NAME_PREFIX)initialiser_funcs.py ../splines/$(NAME_PREFIX)spline_eval_funcs.py

ifneq ($(ACC), pythran)
accelerated_advection_steps$(SO_EXT): $(NAME_PREFIX)accelerated_advection_steps.py $(DEPS) .ACC.$(ACC)
	$(TOOL) $< $(TOOL_FLAGS)
else
accelerated_advection_steps$(SO_EXT): pythran_deps/$(NAME_PREFIX)accelerated_advection_steps.py $(DEPS) .ACC.$(ACC)
	$(TOOL) $< $(TOOL_FLAGS) -o $@
endif


clean:
	rm -f *.o *.so *.mod
	pyccel-clean
