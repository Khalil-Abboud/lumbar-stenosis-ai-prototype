import subprocess
import sys


print("Step 1: Extracting image features...")
subprocess.run(
    [sys.executable, "main.py"],
    check=True
)

print()
print("Step 2: Running online classifier demo...")
subprocess.run(
    [sys.executable, "classifier_demo.py"],
    check=True
)

print()
print("Pipeline completed successfully.")