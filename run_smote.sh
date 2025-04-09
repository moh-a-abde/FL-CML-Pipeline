#!/bin/bash
set -e

# Define default partition ID
PARTITION_ID=0
# Check if a partition ID was provided as an argument
if [ "$#" -ge 1 ]; then
  PARTITION_ID=$1
fi

echo "Starting SMOTE-enhanced client with partition ID: $PARTITION_ID"

# Run the SMOTE client with specified partition ID
python smote_client.py \
  --partition-id ${PARTITION_ID} \
  --train-method bagging \
  --num-partitions 2 \
  --partitioner-type exponential

echo "SMOTE client execution completed" 