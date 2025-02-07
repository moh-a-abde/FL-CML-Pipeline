import subprocess
import time
import os

def run_script(script_path):
    """Run a Python script from the given path."""
    return subprocess.Popen(['python3', script_path])

def wait_for_new_file(directory):
    """Wait for a new file to be created in the directory."""
    existing_files = set(os.listdir(directory))
    while True:
        time.sleep(5)  # Check every 5 seconds
        current_files = set(os.listdir(directory))
        new_files = current_files - existing_files
        if new_files:
            print(f"New file detected: {new_files}")
            return new_files.pop()  # Return the first detected new file

def main():
    # Paths to your scripts
    preprocessing_script = '/home/mohamed/Desktop/test_repo/data/livepreprocessing_socket.py'
    receiving_script = '/home/mohamed/Desktop/test_repo/data/receiving_data.py'
    data_directory = '/home/mohamed/Desktop/test_repo/data'

    # Start both scripts
    preprocessing_process = run_script(preprocessing_script)
    receiving_process = run_script(receiving_script)

    print("Both scripts are running...")

    try:
        # Wait for a new file to appear in the /data directory
        new_file = wait_for_new_file(data_directory)

        # Run the 'act -j run' command once the new file is detected
        subprocess.run(['act', '-j', 'run', '--container-architecture', 'linux/amd64', '-v'])
        print(f"'act -j run' executed after detecting {new_file}")

    except KeyboardInterrupt:
        print("Process interrupted by user.")
    finally:
        # Terminate the running processes
        preprocessing_process.terminate()
        receiving_process.terminate()
        print("Processes terminated.")

if __name__ == "__main__":
    main()
