# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

all: initialiser_funcs$(SO_EXT)

.ACC.$(ACC):
	@rm -f .ACC.*
	@touch $@

initialiser_funcs$(SO_EXT): $(NAME_PREFIX)initialiser_funcs.py .ACC.$(ACC)
ifneq ($(ACC), pythran)
	$(TOOL) $< $(TOOL_FLAGS)
else
	$(TOOL) $< $(TOOL_FLAGS) -o $@
endif


clean:
	rm -f *.o *.so *.mod
	pyccel-clean
