#!/usr/bin/env python3
"""
Training Progress Tracker
Monitors the current training process in real-time
"""

import time
import os
import sys
from datetime import datetime, timedelta

def track_training():
    """Track training progress by monitoring console output"""
    print("=" * 60)
    print("NBA ROCKSTAR MODEL - TRAINING TRACKER")
    print("=" * 60)
    print(f"Started monitoring at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    # Look for running Python processes
    import subprocess
    try:
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'],
                              capture_output=True, text=True)
        python_procs = [line for line in result.stdout.split('\n') if 'python.exe' in line]
        print(f"\nFound {len(python_procs)} Python processes running")
        for proc in python_procs:
            print(f"  {proc.strip()}")
    except:
        pass

    print("\n" + "=" * 60)
    print("PROGRESS MONITORING")
    print("=" * 60)
    print("\nPress Ctrl+C to stop monitoring\n")

    last_progress = None
    start_time = datetime.now()

    # Try to find the training window and monitor it
    # Since we can't directly access the other terminal, we'll check for model files
    model_dir = "models"

    while True:
        try:
            # Check if model files are being updated
            if os.path.exists(model_dir):
                files = os.listdir(model_dir)
                pkl_files = [f for f in files if f.endswith('.pkl')]

                if pkl_files:
                    # Get most recently modified file
                    file_times = [(f, os.path.getmtime(os.path.join(model_dir, f)))
                                 for f in pkl_files]
                    most_recent = max(file_times, key=lambda x: x[1])
                    mod_time = datetime.fromtimestamp(most_recent[1])
                    time_diff = datetime.now() - mod_time

                    current_status = (
                        f"\rLast updated: {most_recent[0]} | "
                        f"Modified: {mod_time.strftime('%H:%M:%S')} | "
                        f"Age: {int(time_diff.total_seconds())}s ago | "
                        f"Elapsed: {str(datetime.now() - start_time).split('.')[0]}"
                    )

                    if current_status != last_progress:
                        print(current_status, end='', flush=True)
                        last_progress = current_status

            time.sleep(5)  # Check every 5 seconds

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(5)

if __name__ == "__main__":
    track_training()
