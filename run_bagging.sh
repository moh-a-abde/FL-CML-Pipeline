#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

echo "Starting server"
python3 server.py --pool-size=2 --num-rounds=50 --num-clients-per-round=2 &
sleep 30  # Sleep for 30s to give the server enough time to start

# Start regular client (partition 0)
echo "Starting regular client (partition 0)"
python3 client.py --partition-id=0 --num-partitions=2 --partitioner-type=exponential &

# Start regular client (partition 1)
echo "Starting regular client (partition 1)"
python3 client.py --partition-id=1 --num-partitions=5 --partitioner-type=exponential &

# Start regular client (partition 2)
echo "Starting regular client (partition 2)"
python3 client.py --partition-id=2 --num-partitions=5 --partitioner-type=exponential &

# Start regular client (partition 3)
echo "Starting regular client (partition 3)"
python3 client.py --partition-id=3 --num-partitions=5 --partitioner-type=exponential &

# Start regular client (partition 4)
echo "Starting regular client (partition 4)"
python3 client.py --partition-id=4 --num-partitions=5 --partitioner-type=exponential &


# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait