#!/bin/bash

# Run Python scripts in the background and store their process IDs (PIDs)
python3 data/receiving_data.py &
PID1=$!

python3 data/livepreprocessing_socket.py &
PID2=$!

# Wait for 1 minute
sleep 30

# Kill the Python scripts
kill $PID1 $PID2

# Run Git commands and GitHub workflow
git pull
./commit.sh
gh workflow run cml.yaml

# Wait for 5 minutess
sleep 300
git pull
