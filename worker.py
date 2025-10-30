"""
Runpod serverless worker for the RUN root cause analysis pipeline.

This worker launches the CLI command
    python main.py --root_path <...> --data_path <...> --root_cause <...> --trigger_point <...>
using parameters supplied in the Runpod job payload. The command output is captured and returned
to the caller so jobs can inspect the program logs.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from typing import Any, Dict, List

import runpod


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Defaults mirror the values documented in README.md so a minimal payload works out of the box.
DEFAULT_ARGS = {
    "root_path": "./",
    "data_path": "data",
    "root_cause": "root_cause",
    "trigger_point": "trigger_point",
    "cuda": "cuda:0",
}


def _build_command(payload: Dict[str, Any]) -> List[str]:
    """
    Translate a Runpod payload into the CLI invocation for `main.py`.

    The payload can either contain parameters at the top level, or inside an `input` field. An
    optional `extra_args` field (list[str]) allows advanced overrides to be appended to the command.
    """
    job_input = payload.get("input", payload)

    # Merge defaults with user-supplied values while keeping unspecified arguments optional.
    cmd_args = {**DEFAULT_ARGS, **{k: v for k, v in job_input.items() if isinstance(v, str)}}

    required_fields = ("root_cause", "data_path", "root_path", "trigger_point")
    missing = [field for field in required_fields if not cmd_args.get(field)]
    if missing:
        raise ValueError(f"Missing required argument(s): {', '.join(missing)}")

    command = [
        "python",
        "main.py",
        "--root_path",
        cmd_args["root_path"],
        "--data_path",
        cmd_args["data_path"],
        "--root_cause",
        cmd_args["root_cause"],
        "--trigger_point",
        cmd_args["trigger_point"],
    ]

    cuda_value = cmd_args.get("cuda")
    if isinstance(cuda_value, str) and cuda_value:
        command.extend(["--cuda", cuda_value])
    if "epochs" in job_input and isinstance(job_input["epochs"], int):
        command.extend(["--epochs", str(job_input["epochs"])])
    if "learning_rate" in job_input and isinstance(job_input["learning_rate"], (int, float)):
        command.extend(["--learning_rate", str(job_input["learning_rate"])])
    if "optimizer" in job_input and isinstance(job_input["optimizer"], str):
        command.extend(["--optimizer", job_input["optimizer"]])
    if "num_workers" in job_input and isinstance(job_input["num_workers"], int):
        command.extend(["--num_workers", str(job_input["num_workers"])])

    extra_args = job_input.get("extra_args", [])
    if extra_args:
        if not isinstance(extra_args, list) or not all(isinstance(arg, str) for arg in extra_args):
            raise TypeError("`extra_args` must be a list of strings.")
        command.extend(extra_args)

    return command


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the RUN pipeline and return structured output for Runpod.
    """
    command = _build_command(event)

    process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    return {
        "command": " ".join(shlex.quote(part) for part in command),
        "return_code": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }


runpod.serverless.start({"handler": handler})
