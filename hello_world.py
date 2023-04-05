# hello_world.py
import sys
import time

def main():
    file_list = sys.argv[1]
    time.sleep(10)
    print(f"Hello, World! from {file_list}")

if __name__ == "__main__":
    main()

