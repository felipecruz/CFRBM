PYTHON ?= python
DEVICE ?= auto
ORIENTATION ?= user

.PHONY: run test

run:
	$(PYTHON) -m cfrbm.runner ubased.json --orientation $(ORIENTATION) --device $(DEVICE)

test:
	$(PYTHON) -m unittest discover -v
