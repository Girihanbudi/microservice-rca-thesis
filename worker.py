import os
import shlex
import subprocess
import runpod

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def list_all_paths(base="/", max_files=1000):
    """
    Walks through all directories starting at `base` and returns up to `max_files` full paths.
    Keeps output manageable for JSON return.
    """
    paths = []
    for root, dirs, files in os.walk(base):
        for name in dirs + files:
            paths.append(os.path.join(root, name))
            if len(paths) >= max_files:
                return paths
    return paths


def handler(event):
    """
    Minimal RunPod worker that calls your main.py with parameters from the job payload.
    Also returns a listing of all visible file paths to help locate mounted volumes or CSVs.
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

    # --- Debug: list all paths ---
    all_paths = list_all_paths("/")
    # -----------------------------

    # Run main.py and capture output
    process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True
    )

    # Return structured logs + full file listing
    return {
        "command": " ".join(shlex.quote(c) for c in command),
        "return_code": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "all_paths": all_paths,  # 👈 Shows every file RunPod can see
    }


# Start the serverless worker
runpod.serverless.start({"handler": handler})
