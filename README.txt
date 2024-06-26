# Hyperbolic Convolution via Kernel Point Aggregation

Code for Hyperbolic Convolution via Kernel Point Aggregation.

## Training

The training scripts are in `scripts\`. You can use them by

```bash
bash scripts/experiment/dataset.experiment.sh
```

## File Descriptions

`data/`: Datasets
`distribution/`: Hyperbolic Distributions
`kernels/`: Kernel generation
`layers/`: Hyperbolic layers
`manifolds/`: Manifold calculations
`models/`: GNN models
`optim/`: Optimization on manifolds
`scripts/`: Training scripts
`utils/`: Utility files
`train.py`: Training scripts

Our HKConv is located in `layers/hyp_layers.py`.
