# RUN (AAAI 2024)

[![Runpod](https://api.runpod.io/badge/Girihanbudi/microservice-rca-thesis)](https://console.runpod.io/hub/Girihanbudi/microservice-rca-thesis)

Official code of the [paper](https://arxiv.org/abs/2402.01140) Root Cause Analysis In Microservice Using Neural Granger Causal Discovery .

## Introduction

We propose a novel approach for root cause analysis using neural Granger causal discovery with contrastive learning.

* Enhance the backbone encoder by integrating contextual information from time series
* Leverage a time series forecasting model to conduct neural Granger causal discovery
* In addition, we incorporate Pagerank with a personalization vector to efficiently recommend the top-k root causes

## Reproduce RUN

```
// create new virtual env
python3 -m venv ./path-to-new-venv

// virtual env activation
.\venv\Scripts\activate

// Installation
pip install -r requirements.txt

// run program
python main.py --root_path ./ --data_path data --root_cause root_cause --trigger_point trigger_point

## Runpod Serverless

The repository includes a `worker.py` entry point so the pipeline can be triggered from a Runpod
Serverless endpoint. The worker launches the CLI command shown above and returns the captured logs.

1. Build your Runpod image from the project root so `main.py` and `worker.py` are on the container
   filesystem.
2. Set the handler to `worker.py`.
3. Submit jobs with the parameters required by `main.py`. A minimal payload looks like:

```json
{
  "input": {
    "root_path": "./",
    "data_path": "data/simple_data_2.csv",
    "root_cause": "service_a",
    "trigger_point": "service_b"
  }
}
```

Optional fields (`cuda`, `epochs`, `learning_rate`, `optimizer`, `num_workers`, `extra_args`) are
passed through to the CLI when provided.
The worker defaults to `--cuda cuda:0`; override with `"cuda": "cpu"` or another device string if needed.

### Build With Docker

A ready-to-use `Dockerfile` is provided. Build and push it before creating the Runpod endpoint:

```bash
docker build -t <your-registry>/<repo>:run .
docker push <your-registry>/<repo>:run
```

Configure the Serverless endpoint to pull that image and execute the default command (`python -u worker.py`).

```

```
