# outer_script.py
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_hello_world(file_list):
    cmd = f"python hello_world.py {file_list}"
    result = subprocess.run(cmd, shell=True, text=True)

    # Return True if the task succeeded, False otherwise
    return (result.returncode == 0, file_list)

def read_completed_tasks(log_file):
    if not os.path.exists(log_file):
        return []

    with open(log_file, "r") as f:
        completed_tasks = f.read().splitlines()

    return set(completed_tasks)

def main():
    file_lists = [f"file_list{i}.txt" for i in range(10)]  # Replace with your actual file_lists
    max_parallel_tasks = 3
    log_file = "completed_tasks.log"

    completed_tasks = read_completed_tasks(log_file)
    remaining_file_lists = [file_list for file_list in file_lists if file_list not in completed_tasks]
    print(f"{len(remaining_file_lists)} left to process out of {len(file_lists)} ")

    with ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
        futures = [executor.submit(run_hello_world, file_list) for file_list in remaining_file_lists]

        for future in as_completed(futures):
            success, file_list = future.result()
            if success:
                print(f"Task for {file_list} completed successfully.")
                with open(log_file, "a") as f:
                    f.write(file_list + "\n")
            else:
                print(f"Task for {file_list} failed.")

if __name__ == "__main__":
    main()

