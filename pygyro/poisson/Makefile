# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

all: poisson_tools$(SO_EXT)

.ACC.$(ACC):
	@rm -f .ACC.*
	@touch $@

poisson_tools$(SO_EXT): $(NAME_PREFIX)poisson_tools.py .ACC.$(ACC)
ifneq ($(ACC), pythran)
	$(TOOL) $< $(TOOL_FLAGS)
else
	$(TOOL) $< $(TOOL_FLAGS) -o $@
endif


clean:
	rm -f *.o *.so *.mod .ACC.pycc .lock_acquisition.lock
	pyccel-clean
