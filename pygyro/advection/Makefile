# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
advection_path := $(abspath $(mkfile_path)/..)
pygyro_path := $(abspath $(advection_path)/..)

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

all: accelerated_advection_steps$(SO_EXT)

.ACC.$(ACC):
	@rm -f .ACC.*
	@touch $@

DEPS := ../initialisation/$(NAME_PREFIX)initialiser_funcs.py ../splines/$(NAME_PREFIX)spline_eval_funcs.py ../splines/$(NAME_PREFIX)cubic_uniform_spline_eval_funcs.py

ifneq ($(ACC), numba)
ifneq ($(ACC), pythran)
accelerated_advection_steps$(SO_EXT): $(NAME_PREFIX)accelerated_advection_steps.py $(DEPS) .ACC.$(ACC)
	$(TOOL) $< $(TOOL_FLAGS)
else
accelerated_advection_steps$(SO_EXT): pythran_deps/$(NAME_PREFIX)accelerated_advection_steps.py $(DEPS) .ACC.$(ACC)
	$(TOOL) $< $(TOOL_FLAGS) -o $@
endif
else
accelerated_advection_steps$(SO_EXT): $(NAME_PREFIX)accelerated_advection_steps.py $(DEPS) .ACC.$(ACC)
	echo $(pygyro_path); PYTHONPATH=$(PYTHONPATH):$(pygyro_path) $(TOOL) $< $(TOOL_FLAGS)
endif


clean:
	rm -f *.o *.so *.mod .ACC.pycc .lock_acquisition.lock
	pyccel-clean
