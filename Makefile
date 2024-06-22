#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME=mlflow-practice
PYTHON_VERSION=3.10
PYTHON_INTERPRETER=python
POSTGRES_DB=mlflow
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_URI=postgresql://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@$(POSTGRES_HOST):$(POSTGRES_PORT)/$(POSTGRES_DB)
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Start mlfow tracking server
.PHONY: mlflow-server
mlflow-server:
	echo $(POSTGRES_URI)
	mlflow server \
		--backend-store-uri $(POSTGRES_URI) \
		--artifacts-destination s3://mlruns \
		--host 0.0.0.0 \
		--port 5000

.PHONY: tracking-storage
tracking-storage:
	echo $(POSTGRES_DB)
	POSTGRES_DB=$(POSTGRES_DB) \
	POSTGRES_USER=$(POSTGRES_USER) \
	POSTGRES_PASSWORD=$(POSTGRES_PASSWORD) \
	AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) \
	AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) \
	docker compose up -d

.PHONY: mlflow-plus-storage
mlflow-plus-storage: tracking-storage mlflow-server

## Install Python Dependencies
.PHONY: requirements
requirements:
	conda env update --name $(PROJECT_NAME) --file environment.yml --prune
	

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 mlflow_practice
	isort --check --diff --profile black mlflow_practice
	black --check --config pyproject.toml mlflow_practice

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml mlflow_practice


## Download Data from storage system
.PHONY: sync_data_down
sync_data_down:
	aws s3 sync s3://mlflow/data/\
		data/ 
	

## Upload Data to storage system
.PHONY: sync_data_up
sync_data_up:
	aws s3 sync s3://mlflow/data/ data/\
		 --profile $(PROFILE)
	



## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	conda env create --name $(PROJECT_NAME) -f environment.yml
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make Dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) mlflow_practice/data/make_dataset.py


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
	@python -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
