VENV = .venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip3
STREAMLIT = $(VENV)/bin/streamlit

# if .env file exists, include it
ifneq (,$(wildcard .env))
	include .env
	export
endif

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

o1: $(VENV)/bin/activate
	$(STREAMLIT) run o1.py

o2: $(VENV)/bin/activate
	$(STREAMLIT) run o2.py

clean:
	rm -rf __pycache__
	rm -rf $(VENV)