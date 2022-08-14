# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

all: utilities_pyccel$(SO_EXT) matrix_assembly_pyccel$(SO_EXT)

.ACC.$(ACC):
	@rm -f .ACC.*
	@touch $@

DEPS := ./$(NAME_PREFIX)utilities_pyccel.py

ifneq ($(ACC), numba)
ifneq ($(ACC), pythran)
# pyccel
utilities_pyccel$(SO_EXT): $(NAME_PREFIX)utilities_pyccel.py .ACC.$(ACC)
	$(TOOL) $< $(TOOL_FLAGS)
matrix_assembly_pyccel$(SO_EXT): $(NAME_PREFIX)matrix_assembly_pyccel.py $(DEPS) .ACC.$(ACC)
	$(TOOL) $< $(TOOL_FLAGS)
else
utilities_pyccel$(SO_EXT): pythran_deps/$(NAME_PREFIX)utilities_pyccel.py .ACC.$(ACC)
	$(TOOL) $< $(TOOL_FLAGS) -o $@
matrix_assembly_pyccel$(SO_EXT): pythran_deps/$(NAME_PREFIX)matrix_assembly_pyccel.py $(DEPS) .ACC.$(ACC)
	$(TOOL) $< $(TOOL_FLAGS) -o $@
endif
else
# numba
utilities_pyccel$(SO_EXT): $(NAME_PREFIX)utilities_pyccel.py .ACC.$(ACC)
	echo $(pygyro_path); PYTHONPATH=$(PYTHONPATH):$(pygyro_path) $(TOOL) $< $(TOOL_FLAGS)
matrix_assembly_pyccel$(SO_EXT): $(NAME_PREFIX)matrix_assembly_pyccel.py $(DEPS) .ACC.$(ACC)
	echo $(pygyro_path); PYTHONPATH=$(PYTHONPATH):$(pygyro_path) $(TOOL) $< $(TOOL_FLAGS)
endif


clean:
	rm -f *.o *.so *.mod .ACC.pycc .lock_acquisition.lock
	pyccel-clean
