import os
import shlex
import subprocess
import runpod

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

def handler(event):
    """
    Minimal RunPod worker that calls your main.py with parameters from the job payload.
    Expected payload structure:
    {
      "input": {
        "root_path": "./",
        "data_path": "data.csv",
        "root_cause": "A",
        "trigger_point": "B",
        "cuda": "cpu"
      }
    }
    """
    job_input = event.get("input", {})

    # Extract arguments (use defaults if missing)
    root_path = job_input.get("root_path", "./")
    data_path = job_input.get("data_path", "data.csv")
    root_cause = job_input.get("root_cause", "root_cause")
    trigger_point = job_input.get("trigger_point", "None")
    cuda = job_input.get("cuda", "cpu")

    # Build the command line for main.py
    command = [
        "python", "main.py",
        "--root_path", root_path,
        "--data_path", data_path,
        "--root_cause", root_cause,
        "--trigger_point", trigger_point,
        "--cuda", cuda
    ]

    # Run main.py and capture output
    process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True
    )

    # Return structured logs
    return {
        "command": " ".join(shlex.quote(c) for c in command),
        "return_code": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }

# Start the serverless worker
runpod.serverless.start({"handler": handler})
