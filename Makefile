PYTHON ?= python
DEVICE ?= auto

.PHONY: run test

run:
	$(PYTHON) -m cfrbm.user_based ubased.json --device $(DEVICE)

test:
	$(PYTHON) -m unittest discover -v
