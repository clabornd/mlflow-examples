# mlflow-practice

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Tutorials and personal practice using mlflow with actual cloud storage and a postgres db.  I also throw in some [hydra](https://github.com/facebookresearch/hydra) for more tool learning, but this could easily be recreated with simple argparse.

## Setup

### Requirements

1. Create the conda environment or otherwise install the dependencies in environment.yml.

```bash
conda env create -f environment.yml
```

2. Install docker and docker compose:  https://docs.docker.com/compose/install/

## Running Locally

### Spin up services

First, spin up the minio and postgres services:

```bash
docker-compose up -d

# using make
make tracking-storage
```

Start the tracking server:

```bash
mlflow server \
    --backend-store-uri postgresql://user:password@localhost:5432/mlflowdb \
    --artifacts-destination s3://mlruns \
    --host 0.0.0.0 \
    --port 5000

# using make
make mlflow-server
```

Or you can start both using make:

```bash
make mlflow-plus-storage
```

See `docker-compose.yaml` and `Makefile` to see environment variables that control the services.  You can edit them by setting any of the environment variables:

```bash
export POSTGRES_USER=user2
export POSTGRES_PASSWORD=password2

make mlflow-plus-storage
```

### Run Example Experiment

First, set a few environment variables, preferably from some serets file, but here it is in bash:

```bash
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 # where the minio service is running
export AWS_ACCESS_KEY_ID=minioadmin # minio username
export AWS_SECRET_ACCESS_KEY=minioadmin # minio password
export MLFLOW_TRACKING_URI=http://localhost:5000 # where the mlflow server is running
```

Then run the example experiment:

```bash
python src/basic-example.py
```

The parameters are controlled by [hydra](https://github.com/facebookresearch/hydra).  It is a way to create structured configs, where for instance you have models, dataloaders, etc. that can all be configured in a schmogasbord of ways.  You can, for example change from the default of training a random forest to training an SVM:

```bash
python src/basic-example.py model=svm

# with a different hyperparameter
python src/basic-example.py model=svm model.C=0.1

# or change the experiment name
python src/basic-example.py experiment_name=my_experiment
```
