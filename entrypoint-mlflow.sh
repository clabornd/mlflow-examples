#!/bin/bash

echo ${POSTGRES_URI}
mlflow server \
    --backend-store-uri ${POSTGRES_URI} \
    --artifacts-destination s3://${AWS_BUCKET} \
    --host 0.0.0.0 \
    --port 5000