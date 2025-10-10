# AIMNet-X2D Usage Guide

This document provides instructions on how to use AIMNet-X2D for molecular property prediction.

## Data Format

Your input data should be in CSV format with:
- A SMILES column (default column name: "smiles")
- One or more target property columns

## Basic Usage Examples

### Single-Task Regression

Train a model to predict a single property:

```bash
python main.py \
  --data_path sample-data/qm9/qm9_whole.csv \
  --smiles_column smiles \
  --target_column homo \
  --task_type regression \
  --train_split 0.8 \
  --val_split 0.1 \
  --test_split 0.1 \
  --model_save_path models/single_task_model.pth \
  --batch_size 64 \
  --epochs 50 \
  --early_stopping
```

### Single-Task Regression with SAE Normalization 

Train a model with SAE normalization for an extrinsic property:

```bash
python main.py \
  --data_path sample-data/qm9/qm9_whole.csv \
  --smiles_column smiles \
  --target_column u0_atom \
  --task_type regression \
  --calculate_sae \
  --train_split 0.8 \
  --val_split 0.1 \
  --test_split 0.1 \
  --model_save_path models/single_task_sae_model.pth \
  --batch_size 64 \
  --epochs 50 \
  --early_stopping
```

**When to use SAE normalization:**
- Use SAE for **extrinsic properties** like energies, enthalpies, or other size-extensive properties that scale with molecule size
- Do NOT use SAE for **intrinsic properties** like logP, HOMO-LUMO gap, or other properties that are inherent to the molecule regardless of size

Applying SAE to intrinsic properties may hurt model performance. The normalization is specifically designed to account for size-extensivity in properties that scale with the number of atoms.


### Activating Optional Features

To activate our optional features (partial charges and stereochemistry), add the --use_partial_charges and/or --use_stereochemistry flags

```bash
python main.py \
  --data_path sample-data/qm9/qm9_whole.csv \
  --smiles_column smiles \
  --target_column homo \
  --task_type regression \
  --train_split 0.8 \
  --val_split 0.1 \
  --test_split 0.1 \
  --model_save_path models/single_task_model.pth \
  --batch_size 64 \
  --epochs 50 \
  --early_stopping \
  --use_partial_charges \
  --use_stereochemistry
```


### Multi-Task Learning with QM9 Dataset with custom train/val/test split

Predict multiple properties simultaneously:

```bash
python main.py \
  --train_data sample-data/qm9/sample-splits/train.csv \
  --val_data sample-data/qm9/sample-splits/val.csv \
  --test_data sample-data/qm9/sample-splits/test.csv \
  --task_type multitask \
  --multi_target_columns mu,alpha,homo,lumo,gap,r2,zpve,cv,u0_atom,u298_atom,h298_atom,g298_atom \
  --calculate_sae \
  --sae_subtasks 7,8,9,10,11 \
  --model_save_path models/qm9_model.pth \
  --batch_size 64 \
  --epochs 100 \
  --early_stopping
```

**Important Note About SAE Subtasks:** 

When using `--calculate_sae` with `--task_type multitask`, you MUST specify `--sae_subtasks` as a comma-separated list of 0-indexed column indices. For example, `--sae_subtasks 7,8,9,10,11` means apply SAE normalization to the 8th, 9th, 10th, 11th, and 12th target columns (0-indexed).

### Using Iterable Dataset for Large Datasets

For datasets that don't fit in memory:

```bash
python main.py \
  --train_data sample-data/qm9/sample-splits/train.csv \
  --val_data sample-data/qm9/sample-splits/val.csv \
  --test_data sample-data/qm9/sample-splits/test.csv \
  --task_type multitask \
  --multi_target_columns mu,alpha,homo,lumo,gap \
  --iterable_dataset \
  --train_hdf5 data/train.hdf5 \
  --val_hdf5 data/val.hdf5 \
  --test_hdf5 data/test.hdf5 \
  --model_save_path models/large_dataset.pth \
  --batch_size 64 \
  --epochs 50
```

**Important Note About Iterable Datasets:** If the HDF5 files specified by `--train_hdf5`, `--val_hdf5`, and `--test_hdf5` already exist, the preprocessing steps will be skipped and the existing HDF5 files will be used. This can save time but may cause unexpected results if you've changed your data processing parameters. If you need to regenerate the HDF5 files, delete the existing ones first.

### Transfer Learning

Use a pre-trained model to start from for a new target property:

```bash
python main.py \
  --train_data sample-data/qm9/sample-splits/train.csv \
  --val_data sample-data/qm9/sample-splits/val.csv \
  --test_data sample-data/qm9/sample-splits/test.csv \
  --task_type multitask \
  --multi_target_columns alpha,homo,lumo \
  --transfer_learning models/qm9_model.pth \
  --freeze_pretrained \
  --model_save_path models/transfer_model.pth \
  --batch_size 64 \
  --epochs 20 \
  --early_stopping
```

For more advanced transfer learning with layer-wise learning rate decay:

```bash
python main.py \
  --train_data sample-data/qm9/sample-splits/train.csv \
  --val_data sample-data/qm9/sample-splits/val.csv \
  --test_data sample-data/qm9/sample-splits/test.csv \
  --task_type multitask \
  --multi_target_columns alpha,homo,lumo \
  --transfer_learning models/qm9_model.pth \
  --layer_wise_lr_decay \
  --lr_decay_factor 0.8 \
  --model_save_path models/transfer_model_decay.pth \
  --batch_size 64 \
  --epochs 30 \
  --early_stopping
```

## Distributed Training

For training on multiple GPUs, use `torchrun` (recommended over the deprecated `torch.distributed.launch`):

```bash
torchrun --nproc_per_node=4 main.py \
  --train_data sample-data/qm9/sample-splits/train.csv \
  --val_data sample-data/qm9/sample-splits/val.csv \
  --test_data sample-data/qm9/sample-splits/test.csv \
  --task_type multitask \
  --multi_target_columns mu,alpha,homo,lumo,gap \
  --num_gpu_devices 4 \
  --model_save_path models/multi_gpu.pth \
  --batch_size 64 \
  --epochs 50
```

### Distributed Training with Iterable Datasets

When using distributed training with iterable datasets, follow a two-step process:

**Step 1:** First, create the HDF5 files without distributed training (set epochs to 0):

```bash
python main.py \
  --train_data sample-data/qm9/sample-splits/train.csv \
  --val_data sample-data/qm9/sample-splits/val.csv \
  --test_data sample-data/qm9/sample-splits/test.csv \
  --task_type multitask \
  --multi_target_columns mu,alpha,homo,lumo,gap \
  --iterable_dataset \
  --train_hdf5 data/train.hdf5 \
  --val_hdf5 data/val.hdf5 \
  --test_hdf5 data/test.hdf5 \
  --epochs 0  # Only create HDF5 files, don't train
```

**Step 2:** Then, run distributed training using the pre-created HDF5 files:

```bash
torchrun --nproc_per_node=4 main.py \
  --train_data sample-data/qm9/sample-splits/train.csv \
  --val_data sample-data/qm9/sample-splits/val.csv \
  --test_data sample-data/qm9/sample-splits/test.csv \
  --task_type multitask \
  --multi_target_columns mu,alpha,homo,lumo,gap \
  --iterable_dataset \
  --train_hdf5 data/train.hdf5 \
  --val_hdf5 data/val.hdf5 \
  --test_hdf5 data/test.hdf5 \
  --num_gpu_devices 4 \
  --model_save_path models/multi_gpu_iterable.pth \
  --batch_size 64 \
  --epochs 50
```

This approach ensures the HDF5 files are created once and then reused for distributed training.

## Inference Mode

### Single-GPU Chunked Inference

For prediction on new molecules:

```bash
python main.py \
  --inference_csv sample-data/qm9/qm9_whole.csv \
  --inference_output results/predictions.csv \
  --model_save_path models/trained_model.pth
```

### Multi-GPU Distributed Chunked Inference

For faster inference on large datasets using multiple GPUs:

```bash
torchrun --nproc_per_node=4 main.py \
  --inference_csv sample-data/qm9/qm9_whole.csv \
  --inference_output results/large_predictions.csv \
  --model_save_path models/trained_model.pth \
  --num_gpu_devices 4 \
  --stream_batch_size 500 \
  --stream_chunk_size 5000
```

This will automatically split the inference workload across the GPUs and combine the results into a single output file.


### HDF5 Inference
For maximum performance, especially with very large datasets or on High-Performance Computing (HPC) systems, it is highly recommended to pre-process your inference data into a specialized HDF5 file.

This pipeline provides a significant speedup by separating the slow, CPU-intensive work (SMILES parsing and graph featurization) from the fast, GPU-intensive work (model prediction).

First, train your model. 
```bash
python main.py \
  --data_path sample-data/qm9/qm9_whole.csv \
  --target_column homo \
  --epochs 50 \
  --model_save_path models/homo_model.pth
```

Second, use the create_inference_hdf5.py script to convert your input CSV into an inference-ready HDF5 file. This script reads your trained model to ensure the molecular features are computed with the correct parameters (e.g., number of shells). In this case, we're just going to inference the whole qm9 dataset for demonstration purposes.
```bash
python create_inference_hdf5.py \
  --model_path models/homo_model.pth \
  --input_csv sample-data/qm9/qm9_whole.csv \
  --output_hdf5 data/new_molecules_inference.h5 \
  --smiles_column smiles \
  --num_workers 20
```

Lastly, run inference. There are two ways to do this (Single GPU or Multi GPU)

#### Single GPU
```bash
python main.py \
  --inference_hdf5 data/new_molecules_inference.h5 \
  --model_save_path models/homo_model.pth \
  --inference_output results/predictions_hdf5_inference.csv
```

#### Multi GPU
```bash
torchrun --nproc_per_node=3 main.py \
  --inference_hdf5 data/new_molecules_inference.h5 \
  --num_gpu_devices 3 \
  --model_save_path models/homo_model.pth \
  --inference_output results/predictions_hdf5_inference_ddp.csv
```


### Uncertainty Estimation

For training with uncertainty quantification, use evidential loss:

```bash
python main.py \
  --data_path sample-data/qm9/qm9_whole.csv \
  --smiles_column smiles \
  --target_column homo \
  --task_type regression \
  --train_split 0.8 \
  --val_split 0.1 \
  --test_split 0.1 \
  --model_save_path models/single_task_model_with_evidential.pth \
  --loss_function evidential \
  --batch_size 64 \
  --epochs 50 \
  --early_stopping
```

For inference with uncertainty quantification (works for all loss functions with the exception of evidential loss):

```bash
python main.py \
  --inference_csv sample-data/qm9/qm9_whole.csv \
  --inference_output results/predictions_with_uncertainty.csv \
  --model_save_path models/trained_model.pth \
  --mc_samples 30
```


## Key Command Line Arguments

### Data Loading

| Argument | Description |
|----------|-------------|
| `--data_path` | Path to single CSV file |
| `--train_data` | CSV file for train set |
| `--val_data` | CSV file for validation set |
| `--test_data` | CSV file for test set |
| `--smiles_column` | Column name for SMILES strings |
| `--target_column` | Column name for target value in single-task mode |
| `--multi_target_columns` | Comma-separated list of target columns for multi-task mode |
| `--train_split` | Fraction for training set (when using `--data_path`) |
| `--val_split` | Fraction for validation set (when using `--data_path`) |
| `--test_split` | Fraction for test set (when using `--data_path`) |

### Model Architecture

| Argument | Description |
|----------|-------------|
| `--hidden_dim` | Hidden dimension size |
| `--num_shells` | Number of hops for message passing |
| `--num_message_passing_layers` | Number of message passing layers |
| `--ffn_hidden_dim` | Feed-forward network hidden dimension |
| `--ffn_num_layers` | Number of feed-forward layers |
| `--pooling_type` | Type of graph pooling (attention, mean, max, sum) |
| `--embedding_dim` | Embedding dimension for atom features |
| `--shell_conv_num_mlp_layers` | Number of MLP layers in shell convolution |
| `--shell_conv_dropout` | Dropout rate for shell convolution |
| `--attention_num_heads` | Number of attention heads |
| `--attention_temperature` | Initial temperature for attention |
| `--activation_type` | Type of activation function (relu, leakyrelu, elu, gelu, silu) |

### Training Parameters

| Argument | Description |
|----------|-------------|
| `--task_type` | Type of task (regression, multitask) |
| `--learning_rate` | Learning rate |
| `--epochs` | Number of training epochs |
| `--batch_size` | Batch size |
| `--early_stopping` | Enable early stopping |
| `--patience` | Early stopping patience |
| `--lr_scheduler` | Learning rate scheduler |
| `--num_workers` | Number of data loading worker processes |
| `--model_save_path` | Where to save the trained model |

### Advanced Features

| Argument | Description |
|----------|-------------|
| `--calculate_sae` | Enable SAE normalization |
| `--sae_subtasks` | Comma-separated list of subtask indices for SAE (0-indexed) |
| `--iterable_dataset` | Use HDF5+IterableDataset for large data |
| `--train_hdf5` | Path to train HDF5 file |
| `--val_hdf5` | Path to validation HDF5 file |
| `--test_hdf5` | Path to test HDF5 file |
| `--use_partial_charges` | Enable partial charge calculations |
| `--use_stereochemistry` | Enable stereochemical features |
| `--output_partial_charges` | Path to save partial charges CSV |
| `--num_gpu_devices` | Number of GPU devices for DDP |
| `--enable_wandb` | Enable Weights & Biases logging |
| `--hyperparameter_file` | YAML file with hyperparameter configuration |
| `--num_trials` | Number of trials for hyperparameter search |
| `--save_embeddings` | Extract and save molecular embeddings |
| `--embeddings_output_path` | Path for saved embeddings |

### Transfer Learning Parameters

| Argument | Description |
|----------|-------------|
| `--transfer_learning` | Path to pretrained model for transfer learning |
| `--freeze_pretrained` | Freeze pretrained layers except output layer |
| `--freeze_layers` | Comma-separated list of layer patterns to freeze |
| `--unfreeze_layers` | Comma-separated list of layer patterns to explicitly unfreeze |
| `--layer_wise_lr_decay` | Enable layer-wise learning rate decay |
| `--lr_decay_factor` | Decay factor for layer-wise learning rate |