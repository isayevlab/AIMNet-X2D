# AIMNet-X2D

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![RDKit](https://img.shields.io/badge/Powered%20by-RDKit-3838ff.svg?logo=data:image/svg+xml;base64,PHN2ZyBpZD0ibWFpbiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB2aWV3Qm94PSIwIDAgMTAwIDEwMCI+PGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgcj0iNTAiIHN0eWxlPSJmaWxsOiMzODM4ZmY7Ii8+PC9zdmc+)](https://www.rdkit.org/)

AIMNet-X2D is a Graph Neural Network-based model for molecular property prediction with multi-task learning capabilities. Designed to scale from small datasets to foundation model level, it enables researchers to create their own molecular foundation models with limited compute resources or scale efficiently when more hardware is available.

**Stay tuned for our upcoming paper with detailed results and methodology!**

## Why AIMNet-X2D?

- **Scalable Architecture**: Train on your laptop or scale to multi-GPU clusters seamlessly
- **Foundation Model Ready**: Build domain-specific molecular foundation models with the same codebase
- **Memory Efficient**: HDF5 streaming for datasets that don't fit in RAM
- **Production Optimized**: Multi-GPU inference, uncertainty quantification, and embedding extraction
- **Scientifically Rigorous**: SAE normalization for size-extensive properties, stereochemistry support

## Features

**Core Capabilities**
- Multi-task learning for simultaneous prediction of multiple molecular properties
- Self-Atomic Energy (SAE) normalization for energy properties
- Multi-hop message passing with BFS-based graph traversal
- Attention-based graph pooling with learnable aggregation

**Scalability & Performance**
- In-memory and iterable (streaming) dataset loading for datasets of any size
- HDF5 support with automatic caching for large-scale workflows
- Multi-GPU training via DistributedDataParallel (DDP)
- Optimized inference pipeline with chunked processing

**Advanced Features**
- Stereochemical feature encoding (chiral centers, E/Z bonds)
- Partial charges prediction using Gasteiger method
- Molecule embedding extraction for transfer learning
- Uncertainty quantification (evidential regression, MC Dropout)
- Foundation model scaling capabilities

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train a model on the included QM9 dataset
python main.py \
  --data_path sample-data/qm9/qm9_whole.csv \
  --target_column homo \
  --task_type regression \
  --epochs 50 \
  --model_save_path models/my_model.pth

# Run inference
python main.py \
  --inference_csv molecules.csv \
  --model_save_path models/my_model.pth \
  --inference_output predictions.csv
```

For detailed usage examples, see [USAGE.md](USAGE.md).

## Installation

### Prerequisites

- Python 3.12+
- CUDA 12.1+ (for GPU support)
- 8GB+ RAM (for sample datasets; more for larger datasets)

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

### CPU-Only Installation

If you don't have a GPU or want to run on CPU:

```bash
# Install PyTorch CPU version first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Then install other dependencies (remove torch lines from requirements.txt first)
pip install -r requirements.txt
```

## Project Structure

```
aimnet-x2d/
├── main.py                    # Main entry point
├── create_inference_hdf5.py   # HDF5 preprocessing for inference
├── src/                       # Modular source code
│   ├── config/               # Configuration management
│   │   ├── args.py           # CLI argument parsing
│   │   ├── validation.py     # Input validation
│   │   ├── experiment.py     # Experiment tracking
│   │   └── paths.py          # Path management utilities
│   ├── datasets/             # Data handling
│   │   ├── molecular.py      # PyTorch Geometric datasets
│   │   ├── features.py       # Molecular feature computation
│   │   ├── loaders.py        # DataLoader creation
│   │   └── io.py             # File I/O operations
│   ├── models/               # Model architecture
│   │   ├── gnn.py            # Main GNN model
│   │   ├── layers.py         # Shell convolution layers
│   │   ├── pooling.py        # Graph pooling mechanisms
│   │   ├── losses.py         # Loss functions (evidential, weighted)
│   │   └── normalizers.py    # Data normalization layers
│   ├── training/             # Training pipeline
│   │   ├── trainer.py        # Training loops with DDP support
│   │   ├── evaluator.py      # Model evaluation metrics
│   │   ├── predictor.py      # Prediction methods
│   │   └── extractors.py     # Embedding extraction
│   ├── inference/            # Inference pipeline
│   │   ├── engine.py         # Inference orchestration
│   │   ├── pipeline.py       # Streaming processing
│   │   ├── uncertainty.py    # MC Dropout uncertainty
│   │   └── embeddings.py     # Embedding extraction
│   ├── data/                 # Data preprocessing
│   │   └── preprocessing.py  # SAE & scaling pipelines
│   ├── main/                 # Execution management
│   │   ├── runner.py         # Main execution orchestration
│   │   ├── cli.py            # CLI interface
│   │   ├── hyperopt.py       # Hyperparameter optimization
│   │   └── utils.py          # Execution utilities
│   └── utils/                # General utilities
│       ├── distributed.py    # Multi-GPU coordination
│       ├── optimization.py   # Training optimizations
│       ├── activation.py     # Activation function factory
│       └── random.py         # Reproducibility utilities
├── sample-data/              # Example QM9 dataset (~134k molecules)
├── tests/                    # Unit tests
├── requirements.txt          # Python dependencies
├── USAGE.md                  # Detailed usage guide
├── CLAUDE.md                 # AI assistant instructions
└── README.md                 # This file
```

## Key Concepts

### SAE Normalization

Self-Atomic Energy (SAE) normalization accounts for properties that scale with molecular size:

- **Use SAE for**: Energies, enthalpies, heat capacities (extrinsic properties)
- **Don't use SAE for**: logP, HOMO-LUMO gap, dipole moment (intrinsic properties)

```bash
python main.py --data_path data.csv --target_column u0_atom --calculate_sae
```

### Large Dataset Handling

Two modes for different dataset sizes:

- **In-memory** (default): Fast, requires dataset to fit in RAM
- **Iterable/HDF5** (`--iterable_dataset`): Streaming from disk for massive datasets

```bash
# For datasets > available RAM
python main.py --train_data huge_dataset.csv --iterable_dataset --train_hdf5 data/train.h5
```

### Multi-GPU Training

Leverage multiple GPUs with PyTorch DistributedDataParallel:

```bash
torchrun --nproc_per_node=4 main.py --data_path data.csv --num_gpu_devices 4
```

## Documentation

- **[USAGE.md](USAGE.md)**: Comprehensive usage examples and command reference
- **[CLAUDE.md](CLAUDE.md)**: Instructions for AI coding assistants

## Troubleshooting

### Common Issues

**Out of Memory Errors**
- Reduce `--batch_size` (try 32, 16, or 8)
- Use `--iterable_dataset` for HDF5 streaming
- Enable gradient checkpointing with `--gradient_checkpointing`

**CUDA Out of Memory**
- Reduce batch size
- Reduce `--hidden_dim` or `--num_shells`
- Use mixed precision training with `--mixed_precision`

**Slow Data Loading**
- Increase `--num_workers` (typically 4-8)
- Use HDF5 preprocessing for inference workloads
- Ensure data files are on fast storage (SSD, not NFS)

**HDF5 Files Not Regenerating**
- Delete existing HDF5 files manually to force regeneration
- Use `--epochs 0` to only create HDF5 files without training

### Debug Mode

Enable detailed logging:

```bash
AIMNET_DEBUG=1 python main.py [arguments]
```

## Performance Tips

1. **Use HDF5 for large datasets**: Pre-process once, train many times
2. **Multi-GPU inference**: Use `torchrun` with `--num_gpu_devices` for faster predictions
3. **Optimize workers**: Set `--num_workers` to your CPU core count (but not higher)
4. **Batch size tuning**: Larger batches are faster but use more memory; find the sweet spot
5. **Mixed precision**: Add `--mixed_precision` for 2x speedup on modern GPUs (A100, RTX 30xx+)

## Citation

If you use AIMNet-X2D in your research, please cite our upcoming paper:

```bibtex
@article{aimnetx2d2025,
  title={AIMNet-X2D: Scalable Graph Neural Networks for Molecular Property Prediction},
  author={Nandakumar, Rohit and Zubatyuk, Roman and Isayev, Olexandr},
  journal={In preparation},
  year={2025}
}
```

## Contributing

We welcome contributions! Areas of interest:

- Additional pooling mechanisms
- New loss functions for specialized tasks
- Performance optimizations
- Documentation improvements
- Bug reports and fixes

Please open an issue or pull request on [GitHub](https://github.com/isayevlab/aimnet-x2d).

## Support

- **Issues**: [GitHub Issues](https://github.com/isayevlab/aimnet-x2d/issues)
- **Discussions**: [GitHub Discussions](https://github.com/isayevlab/aimnet-x2d/discussions)
- **Email**: [Isayev Lab](https://isayevlab.org/)

## Authors

Rohit Nandakumar, Roman Zubatyuk, Olexandr Isayev

[Isayev Laboratory](https://isayevlab.org/), Carnegie Mellon University

## License

[MIT License](LICENSE)

## Acknowledgments

Built with:
- [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [RDKit](https://www.rdkit.org/) for molecular featurization
- [HDF5](https://www.hdfgroup.org/) for efficient data storage
