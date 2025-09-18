# AIMNet-X2D

AIMNet-X2D is a Graph Neural Network-based model for molecular property prediction with multi-task learning capabilities, designed to scale from small to large datasets. The architecture is capable of scaling to foundation model level, allowing researchers to create their own molecular foundation models with limited compute resources or scale efficiently when more hardware is available.

**Stay tuned for our upcoming paper with detailed results and methodology!**

## Features

- Multi-task learning for molecular properties
- Size-extensive additive (SAE) normalization for energy properties
- In-memory and iterable (streaming) dataset loading options for scalability
- HDF5 support for large datasets
- Multi-GPU training support via DistributedDataParallel
- Attention-based graph pooling
- Multi-hop message passing
- Stereochemical feature inclusion
- Partial charges prediction
- Molecule embedding extraction
- Foundation model scaling capabilities

## Installation

### Setup

1. Clone the repository:
```bash
git clone https://github.com/isayevlab/aimnet-x2d.git
cd aimnet-x2d
```

2. Create a conda environment:
```bash
conda create -n aimnet-x2d python=3.12
conda activate aimnet-x2d
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

After installation, see [USAGE.md](USAGE.md) for detailed instructions on how to run AIMNet-X2D.

## Project Structure

```
aimnet-x2d/
├── main.py                    # Main entry point
├── src/                       # Modular source code
│   ├── config/               # Configuration management
│   │   ├── args.py           # Argument parsing
│   │   ├── validation.py     # Input validation
│   │   ├── experiment.py     # Experiment tracking
│   │   └── paths.py          # Path management
│   ├── datasets/             # Data handling
│   │   ├── molecular.py      # PyTorch Geometric datasets
│   │   ├── features.py       # Molecular feature computation
│   │   ├── loaders.py        # Data loaders
│   │   └── io.py             # File I/O operations
│   ├── models/               # Model architecture
│   │   ├── gnn.py            # Main GNN model
│   │   ├── layers.py         # Neural network layers
│   │   ├── pooling.py        # Graph pooling mechanisms
│   │   ├── losses.py         # Loss functions
│   │   └── normalizers.py    # Data normalization
│   ├── training/             # Training pipeline
│   │   ├── trainer.py        # Training loops
│   │   ├── evaluator.py      # Model evaluation
│   │   ├── predictor.py      # Prediction methods
│   │   └── extractors.py     # Feature extraction
│   ├── inference/            # Inference pipeline
│   │   ├── engine.py         # Inference orchestration
│   │   ├── pipeline.py       # Processing pipeline
│   │   ├── uncertainty.py    # Uncertainty estimation
│   │   └── embeddings.py     # Embedding extraction
│   ├── data/                 # Data preprocessing
│   │   └── preprocessing.py  # SAE & scaling pipelines
│   ├── main/                 # Execution management
│   │   ├── runner.py         # Main execution logic
│   │   ├── cli.py            # Command-line interface
│   │   ├── hyperopt.py       # Hyperparameter optimization
│   │   └── utils.py          # Execution utilities
│   └── utils/                # General utilities
│       ├── distributed.py    # Multi-GPU utilities
│       ├── optimization.py   # Training optimizations
│       ├── activation.py     # Activation functions
│       └── random.py         # Reproducibility tools
├── sample-data/              # Example datasets
├── requirements.txt          # Python dependencies
├── USAGE.md                  # Detailed usage guide
└── README.md                 # This file
```


## Authors
Rohit Nandakumar, Roman Zubatyuk, Olexandr Isayev

## License

[MIT License]
