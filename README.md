# RUN (AAAI 2024)

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
```
