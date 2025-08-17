# Breaking the Aggregation Bottleneck in Federated Recommendation: A Personalized Model Merging Approach

## Introduction

Federated recommendation (FR) facilitates collaborative training by aggregating local models from massive devices, enabling client-specific personalization while ensuring privacy. However, we empirically and theoretically demonstrate that server-side aggregation can undermine client-side personalization, leading to suboptimal performance, which we term the aggregation bottleneck. This issue stems from the inherent heterogeneity across numerous clients in FR, which drives the globally aggregated model to deviate from local optima. To this end, we propose FedEM, which elastically merges the global and local models to compensate for impaired personalization. Unlike existing personalized federated recommendation (pFR) methods, FedEM (1) investigates the aggregation bottleneck in FR through theoretical insights, rather than relying on heuristic analysis; (2) leverages off-the-shelf local models rather than designing additional mechanisms to boost personalization. Extensive experiments on real-world datasets demonstrate that our method preserves client personalization during collaborative training, outperforming state-of-the-art baselines.

## Requirements

The code is built on `Python=3.8` and `Pytorch=2.0`.

The other necessary Python libraries are as follows:
    
* coloredlogs>=15.0.1
* cvxpy==1.4.2
* numpy>=1.24.3
* pandas>=2.0.3
* scikit_learn>=1.0.2

To install these, please run the following commands:

  `pip install -r requirements.txt`
  
## Code Structure

The structure of our project is presented in a tree form as follows:

```
FedEM  # The root of project.
│   README.md
│   requirements.txt
│   train.py # The entry function file includes the main hyperparameter configurations.
|
└───datasets  # The used datasets in this work.
│   │   filmtrust   
|   │       ratings.dat
|   |   ...
|   |
└───model  # The main components in FR tasks.
│   │  engine.py # It includes the server aggregation and local training processes.
│   │  loss.py # Task-specific loss for local clients.
│   │  model.py # Defined backbone model and adapter network architecture.
│   │  tools.py # Similarity-based aggregation optimization process.
|   |
└───utils  # Other commonly used tools.
|   │   data.py # Codes related to data loading and preprocessing.
|   │   metrics.py # The evaluation metrics used in this work.
|   │   utils.py # Other utility functions.
```

## Parameters Settings

The main hyper-parameters for tuning are as follows:

`alpha`: the weight of model similarity for aggregation, the default value is `0.9`.

`adapter`: the specific number of layers and units used in MLPs for adapter, the default value is `[32, 16, 8, 1]`.


## Quick Start

Please change the used dataset and hyper-parameters in `train.py`.

`mkdir logs`

`python train.py --dataset='filmtrust' --lr_embedding=1e-1 --lr_adapter=1e-1 --alpha=0.9`

