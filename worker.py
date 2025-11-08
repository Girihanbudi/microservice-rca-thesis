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
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import runpod
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError
from runpod.serverless.utils import download_files_from_urls


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
JOB_STORAGE_ROOT = Path(os.getenv("RUNPOD_INPUT_ROOT", "/tmp/runpod_inputs"))

# Defaults mirror the values documented in README.md so a minimal payload works out of the box.
DEFAULT_ARGS = {
    "root_path": "./",
    "data_path": "data",
    "root_cause": "root_cause",
    "trigger_point": "trigger_point",
    "cuda": "cuda:0",
}

S3_ENDPOINT_DEFAULT = "https://s3api-us-ks-2.runpod.io"
S3_REGION_DEFAULT = "us-ks-2"


def _job_identifier(event: Dict[str, Any]) -> str:
    """
    Return a stable identifier for the current Runpod job.
    """
    return str(event.get("id") or event.get("jobId") or event.get("job_id") or "runpod-job")


def _discover_network_volume_root() -> Optional[Path]:
    """
    Try to locate the mounted Runpod network volume, if attached.
    """
    candidate_paths = [
        os.getenv("RUNPOD_VOLUME_PATH"),
        os.getenv("RUNPOD_MOUNT_PATH"),
        os.getenv("RUNPOD_NETWORK_VOLUME"),
        "/runpod-volume",
    ]

    for path_value in candidate_paths:
        if not path_value:
            continue
        candidate = Path(path_value)
        if candidate.is_dir():
            return candidate
    return None


def _find_network_volume_file(filename: Optional[str] = None) -> Optional[Path]:
    """
    Return the path to `filename` on the Runpod network volume, if it exists.
    """
    volume_root = _discover_network_volume_root()
    if volume_root is None:
        return None

    target_name = filename or os.getenv("RUNPOD_DATA_FILENAME") or "data.csv"
    candidate = volume_root / target_name
    if candidate.is_file():
        return candidate
    return None


def _safe_destination(base_dir: Path, requested_name: str) -> Path:
    """
    Resolve a user-supplied filename into a safe, relative destination.
    """
    candidate = Path(requested_name)
    if candidate.is_absolute():
        raise ValueError("Absolute paths are not permitted for uploaded files.")

    cleaned_parts = [part for part in candidate.parts if part not in ("", ".", "..")]
    if not cleaned_parts:
        raise ValueError("Invalid filename for uploaded file.")

    destination = base_dir.joinpath(*cleaned_parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _parse_s3_location(candidate: Optional[str]) -> Optional[Tuple[str, str]]:
    """
    Parse an S3-style path into (bucket, key) if the path explicitly references S3.
    Supports `s3://`, `s3a://`, and `runpod://` schemes. If no scheme is supplied, an explicit
    bucket must be provided via the `S3_BUCKET` environment variable; otherwise `None` is returned.
    """
    if not isinstance(candidate, str):
        return None

    value = candidate.strip()
    if not value:
        return None

    normalized = value.replace("\\", "/")
    lowered = normalized.lower()
    scheme_present = False
    for prefix in ("s3://", "s3a://", "runpod://"):
        if lowered.startswith(prefix):
            normalized = normalized[len(prefix) :]
            scheme_present = True
            break

    env_bucket = os.getenv("S3_BUCKET")
    if not scheme_present and env_bucket is None:
        return None

    normalized = normalized.lstrip("/")
    if not normalized:
        if scheme_present:
            raise ValueError("S3 path must include both bucket and object key.")
        raise ValueError("S3 object key missing and no default bucket configured.")

    bucket: Optional[str]
    key: str

    if scheme_present:
        if "/" not in normalized:
            raise ValueError("S3 path must include both bucket and object key.")
        bucket, key = normalized.split("/", 1)
    else:
        bucket = env_bucket
        key = normalized

    bucket = bucket or env_bucket
    if not bucket:
        raise ValueError("S3 bucket not specified; set S3_BUCKET or include it in the path.")

    key = key.strip("/")
    if not key:
        raise ValueError("S3 object key missing from path.")

    return bucket, key


def _resolve_s3_dataset(job_input: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """
    Determine whether the payload references data stored on the Runpod S3-compatible volume.
    Returns (bucket, key) when an S3 location is detected, otherwise None.
    """
    data_path = job_input.get("data_path")
    if isinstance(data_path, str):
        try:
            parsed = _parse_s3_location(data_path)
        except ValueError as exc:
            raise ValueError(f"Invalid S3 data_path '{data_path}': {exc}") from exc
        if parsed:
            return parsed

    root_path = job_input.get("root_path")
    if isinstance(root_path, str) and isinstance(data_path, str):
        try:
            root_parsed = _parse_s3_location(root_path)
        except ValueError as exc:
            raise ValueError(f"Invalid S3 root_path '{root_path}': {exc}") from exc
        if root_parsed:
            bucket, base_key = root_parsed
            relative_key = data_path.strip().replace("\\", "/").lstrip("/")
            combined_parts = [segment for segment in (base_key.strip("/"), relative_key) if segment]
            if not combined_parts:
                raise ValueError("Combined S3 root_path and data_path result in an empty object key.")
            combined_key = "/".join(combined_parts)
            return bucket, combined_key

    return None


@lru_cache(maxsize=1)
def _get_s3_client() -> boto3.client:
    """
    Lazily construct and cache a boto3 S3 client using Runpod storage credentials.
    """
    access_key = os.getenv("S3_ACCESS_KEY")
    secret_key = os.getenv("S3_SECRECT_KEY")
    if not access_key or not secret_key:
        raise RuntimeError(
            "S3 credentials are not configured. Set S3_ACCESS_KEY and S3_SECRECT_KEY environment variables."
        )

    endpoint_url = os.getenv("S3_ENDPOINT_URL", S3_ENDPOINT_DEFAULT)
    region_name = os.getenv("S3_REGION", S3_REGION_DEFAULT)

    return boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint_url,
        region_name=region_name,
        config=BotoConfig(s3={"addressing_style": "path"}),
    )


def _download_s3_dataset(event: Dict[str, Any], bucket: str, key: str) -> Path:
    """
    Download an object from the Runpod S3-compatible endpoint for this job and return its local path.
    """
    destination_root = JOB_STORAGE_ROOT / _job_identifier(event) / "s3"
    destination = _safe_destination(destination_root, key)

    client = _get_s3_client()
    try:
        client.download_file(bucket, key, str(destination))
    except (ClientError, BotoCoreError) as exc:
        raise RuntimeError(f"Failed to download s3://{bucket}/{key}: {exc}") from exc

    return destination


def _prepare_s3_data(event: Dict[str, Any]) -> Optional[str]:
    """
    Download the dataset from the Runpod S3 volume when referenced in the payload.
    Updates the payload so subsequent processing uses the local copy.
    """
    job_input = event.get("input", event)
    dataset_location = _resolve_s3_dataset(job_input)
    if not dataset_location:
        return None

    bucket, key = dataset_location
    local_path = _download_s3_dataset(event, bucket, key)
    job_input["root_path"] = str(local_path.parent)
    job_input["data_path"] = local_path.name
    job_input["s3_source"] = f"s3://{bucket}/{key}"
    return str(local_path)


def _prepare_job_files(event: Dict[str, Any]) -> Dict[str, str]:
    """
    Download files attached to the Runpod payload and return their local paths.
    """
    job_input = event.get("input", event)
    file_entries = job_input.get("files")
    if not isinstance(file_entries, list) or not file_entries:
        return {}

    job_identifier = _job_identifier(event)
    input_root = JOB_STORAGE_ROOT / job_identifier / "inputs"
    input_root.mkdir(parents=True, exist_ok=True)

    urls = [entry.get("url") for entry in file_entries if isinstance(entry, Dict) and entry.get("url")]
    if not urls:
        return {}

    downloaded_paths = download_files_from_urls(job_identifier, urls)

    saved_files: Dict[str, str] = {}
    for entry, downloaded in zip(file_entries, downloaded_paths):
        if not isinstance(entry, Dict):
            continue
        if not downloaded:
            continue

        requested_name = entry.get("name") or os.path.basename(downloaded)
        try:
            destination = _safe_destination(input_root, requested_name)
        except ValueError:
            destination = input_root / os.path.basename(downloaded)
            destination.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(downloaded, destination)
        saved_files[requested_name] = str(destination)

    if saved_files:
        if str(job_input.get("root_path")) in (None, "", ".", "./"):
            job_input["root_path"] = str(input_root)
        job_input["files"] = saved_files

    return saved_files


def _build_command(payload: Dict[str, Any]) -> List[str]:
    """
    Translate a Runpod payload into the CLI invocation for `main.py`.

    The payload can either contain parameters at the top level, or inside an `input` field. An
    optional `extra_args` field (list[str]) allows advanced overrides to be appended to the command.
    """
    job_input = payload.get("input", payload)

    defaults = DEFAULT_ARGS.copy()
    network_volume_file = _find_network_volume_file()
    if network_volume_file is not None:
        defaults["root_path"] = str(network_volume_file.parent)
        defaults["data_path"] = network_volume_file.name

    # Merge defaults with user-supplied values while keeping unspecified arguments optional.
    string_inputs = {k: v for k, v in job_input.items() if isinstance(v, str)}
    cmd_args = {**defaults, **string_inputs}

    data_path_value = string_inputs.get("data_path")
    if isinstance(data_path_value, str) and data_path_value:
        absolute_candidate = Path(data_path_value)
        if absolute_candidate.is_absolute():
            cmd_args["root_path"] = str(absolute_candidate.parent)
            cmd_args["data_path"] = absolute_candidate.name
        else:
            root_path_value = string_inputs.get("root_path")
            if root_path_value in (None, "", ".", "./"):
                volume_root = _discover_network_volume_root()
                if volume_root is not None:
                    resolved = volume_root / absolute_candidate
                    if resolved.is_file():
                        cmd_args["root_path"] = str(resolved.parent)
                        cmd_args["data_path"] = resolved.name

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
    saved_files = _prepare_job_files(event)
    s3_data_file = _prepare_s3_data(event)
    if s3_data_file:
        saved_files.setdefault("s3_data_file", s3_data_file)
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
        "downloaded_files": saved_files,
        "s3_data_file": s3_data_file,
    }


runpod.serverless.start({"handler": handler})
