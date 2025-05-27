#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Ensure the global feature processor is created before starting server and clients
# This guarantees consistent preprocessing across all clients and the server
echo "Step 1: Ensuring global feature processor is created..."
python create_global_processor.py \
    --data-file "data/received/final_dataset.csv" \
    --output-dir "outputs" \
    --force

if [ $? -ne 0 ]; then
    echo "Error creating global feature processor. Exiting."
    exit 1
fi

echo "✓ Global feature processor ready at outputs/global_feature_processor.pkl"
echo ""

echo "Step 2: Starting federated learning server..."
python3 server.py --pool-size=5 --num-rounds=5 --num-clients-per-round=5 --centralised-eval &
SERVER_PID=$!
sleep 30  # Sleep for 30s to give the server enough time to start

echo "Step 3: Starting federated learning clients..."

# Start regular client (partition 0)
echo "Starting regular client (partition 0)"
python3 client.py --partition-id=0 --num-partitions=5 --partitioner-type=exponential &
CLIENT_PIDS[0]=$!

# Start regular client (partition 1)
echo "Starting regular client (partition 1)"
python3 client.py --partition-id=1 --num-partitions=5 --partitioner-type=exponential &
CLIENT_PIDS[1]=$!

# Start regular client (partition 2)
echo "Starting regular client (partition 2)"
python3 client.py --partition-id=2 --num-partitions=5 --partitioner-type=exponential &
CLIENT_PIDS[2]=$!

# Start regular client (partition 3)
echo "Starting regular client (partition 3)"
python3 client.py --partition-id=3 --num-partitions=5 --partitioner-type=exponential &
CLIENT_PIDS[3]=$!

# Start regular client (partition 4)
echo "Starting regular client (partition 4)"
python3 client.py --partition-id=4 --num-partitions=5 --partitioner-type=exponential &
CLIENT_PIDS[4]=$!

echo "All clients started. Waiting for federated learning to complete..."

# Enable CTRL+C to stop all background processes
trap "echo 'Stopping all processes...'; kill -- -$$" SIGINT SIGTERM

# Wait for server to complete (it will finish after all rounds)
wait $SERVER_PID

# Wait for all client processes to complete
for pid in "${CLIENT_PIDS[@]}"; do
    if kill -0 $pid 2>/dev/null; then
        wait $pid
    fi
done

echo ""
echo "✓ Federated learning completed successfully!"
echo "Results saved to: outputs/"