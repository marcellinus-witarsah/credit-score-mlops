#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = credit-score-mlops
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

.PHONY: requirements_web_app
requirements_web_app:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r ./app/requirements.txt

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 credit_score_mlops
	isort --check --diff --profile black credit_score_mlops
	black --check --config pyproject.toml credit_score_mlops

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml credit_score_mlops

## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"

## Create a ipykernel
.PHONY: create_ipykernel
create_ipykernel: requirements
	$(PYTHON_INTERPRETER) -m pip install ipykernel
	$(PYTHON_INTERPRETER) -m ipykernel install --user --name $(PROJECT_NAME) --display-name "$(PROJECT_NAME) (Python $(PYTHON_VERSION))"
	@echo ">>> ipykernel created"


## Create a documentation using numpydoc format
.PHONY: pyment_generate_doc
pyment_generate_doc:
	pyment -w -o $(DOC_FORMAT) $(PYTHON_FILE)
	@echo ">>> $(DOC_FORMAT) documentation generated"


## Update requirements.text
.PHONY: update_requirements
update_requirements:
	echo '-e .' > requirements.txt
	pip-chill >> requirements.txt
	@echo ">>> requirements.txt updated"

## Push web application to the HuggingFace Spaces
.PHONY: hf-login
hf-login:
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF) --add-to-git-credential

.PHONY: hf-push-web-app
hf-push-web-app:
	huggingface-cli upload marcellinus-witarsah/credit-score-app-v2 ./app/app.py --repo-type=space --commit-message="Push app.py file"
	huggingface-cli upload marcellinus-witarsah/credit-score-app-v2 ./app/requirements.txt --repo-type=space --commit-message="Push requirements.txt"

.PHONY: deploy
deploy: hf-login hf-push-web-app
#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make Dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) credit_score_mlops/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
