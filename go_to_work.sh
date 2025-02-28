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

# Wait for 5 minutes
sleep 300

# Run Git commands again
git pull
./commit.sh

# Open the evaluation results folder
#xdg-open /path/to/eval_results  # For Linux
# open /path/to/eval_results    # Uncomment this if you're on macOS

