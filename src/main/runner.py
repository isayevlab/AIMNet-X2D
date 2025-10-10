"""
Main execution runner for AIMNet-X2D.

This module contains the core execution logic for training, evaluation,
and inference operations.
"""

import os
import time
import copy
from typing import Dict, Any, Optional, Tuple

import torch
import torch.distributed as dist
import numpy as np

# Local imports
from utils import set_seed, is_main_process
from config import validate_args
from datasets import (
    load_dataset_simple,
    load_dataset_multitask,
    split_dataset,
    precompute_all_and_filter,
    precompute_and_write_hdf5_parallel_chunked,
    create_pyg_dataloader,
    create_iterable_pyg_dataloader,
    PyGSMILESDataset,
    ATOM_TYPES,
    DEGREES,
    HYBRIDIZATIONS,
)
from models import GNN, WeightedL1Loss, WeightedMSELoss, EvidentialLoss, WeightedEvidentialLoss
from training import train_gnn, evaluate, extract_embeddings_main
from inference import inference_main
from data.preprocessing import PreprocessingConfig, preprocess_molecular_data

from data.preprocessing import (
    PreprocessingConfig, 
    preprocess_molecular_data, 
    PreprocessingPipeline, 
    StandardScaler, 
    SAENormalizer
)

from .utils import (
    setup_distributed_environment,
    setup_model_paths,
    check_data_consistency,
    validate_trial_arguments,
    handle_inference_mode,
    setup_experiment_logging,
    finalize_experiment_logging,
    cleanup_temporary_files,
    print_final_summary,
    log_system_info,
    create_experiment_summary,
    save_experiment_summary,
)

import h5py

from datasets import ATOM_TYPES, DEGREES, HYBRIDIZATIONS


def main_runner(args) -> Dict[str, Any]:
    """
    Main execution runner that orchestrates the entire training/inference pipeline.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary containing execution results
    """
    # Set random seed for reproducibility
    set_seed(42)
    
    # Log system information
    if is_main_process():
        log_system_info()
    
    # Validate arguments
    validate_args(args)
    
    # Setup paths
    setup_model_paths(args)
    
    # Check data consistency
    check_data_consistency(args)
    
    # Setup distributed environment
    device, is_ddp, local_rank, world_size = setup_distributed_environment(args)
    
    # Handle inference mode
    if handle_inference_mode(args):
        results = _run_inference_mode(args, device, is_ddp, local_rank, world_size)
        
        if is_ddp and dist.is_initialized():
            print(f"[Runner] Rank {local_rank}: Cleaning up distributed resources...")
            try:
                # Final barrier to ensure all ranks finish
                dist.barrier()  # FIXED: Removed timeout
                print(f"[Runner] Rank {local_rank}: Final barrier passed")
            except Exception as e:
                print(f"[Runner] Rank {local_rank}: Barrier error (may be expected): {e}")
            
            # Destroy process group
            try:
                dist.destroy_process_group()
                print(f"[Runner] Rank {local_rank}: Process group destroyed")
            except Exception as e:
                print(f"[Runner] Rank {local_rank}: Error destroying process group: {e}")
        
        return results
    
    # Setup experiment logging
    wandb_run = setup_experiment_logging(args)
    
    try:
        # Run training/evaluation
        results = _run_training_mode(args, device, is_ddp, local_rank, world_size)
        
        # Finalize logging
        finalize_experiment_logging(wandb_run, results)
        
        # Print final summary
        if is_main_process():
            print_final_summary(results, args)
        
        return results
        
    except Exception as e:
        print(f"Execution failed with error: {e}")
        if wandb_run:
            finalize_experiment_logging(wandb_run, {"error": str(e)})
        raise
    
    finally:
        # Cleanup
        cleanup_temporary_files(args)
        
        # CRITICAL FIX: Always cleanup distributed properly
        if is_ddp and dist.is_initialized():
            print(f"[Runner] Rank {local_rank}: Final cleanup...")
            try:
                dist.barrier(timeout=10)
            except:
                pass  # Timeout is OK here
            
            try:
                dist.destroy_process_group()
                print(f"[Runner] Rank {local_rank}: Cleanup complete")
            except:
                pass  # Already destroyed is OK

def _run_inference_mode(args, device, is_ddp, local_rank, world_size) -> Dict[str, Any]:
    """Run inference mode execution with progress tracking."""
    if is_main_process():
        print("="*80)
        print("RUNNING INFERENCE MODE")
        print("="*80)
    
    # Estimate total molecules for progress bar
    total_molecules = None
    if args.inference_hdf5:
        try:
            import h5py
            with h5py.File(args.inference_hdf5, 'r') as f:
                if 'metadata' in f and 'num_samples' in f['metadata'].attrs:
                    total_molecules = f['metadata'].attrs['num_samples']
                elif 'data' in f:
                    total_molecules = f['data'].shape[0]
            
            if is_main_process() and total_molecules:
                print(f"📊 Processing {total_molecules:,} molecules from HDF5")
        except:
            pass  # Will just not show progress bar
    
    # Run inference with progress tracking
    inference_main(args, device, is_ddp, local_rank, world_size, 
                   total_molecules=total_molecules)
    
    # CRITICAL FIX: Wait for all processes to complete before returning
    if is_ddp and dist.is_initialized():
        print(f"[Inference] Rank {local_rank}: Waiting for all ranks to finish...")
        try:
            dist.barrier()  # FIXED: Removed timeout parameter
            print(f"[Inference] Rank {local_rank}: All ranks completed")
        except Exception as e:
            print(f"[Inference] Rank {local_rank}: Barrier error: {e}")
    
    results = {
        "mode": "inference",
        "output_path": args.inference_output,
        "input_path": args.inference_csv or args.inference_hdf5,
    }
    
    if is_main_process():
        print("✅ Inference completed successfully")
    
    return results



def _run_training_mode(args, device, is_ddp, local_rank, world_size) -> Dict[str, Any]:
    """Run training mode execution."""
    if is_main_process():
        print("="*80)
        print("RUNNING TRAINING MODE")
        print("="*80)
    
    # Load and preprocess data
    data_info = _load_and_preprocess_data(args)
    
    # Create datasets
    datasets_info = _create_datasets(args, data_info)
    
    # Create data loaders
    data_loaders = _create_data_loaders(args, datasets_info, is_ddp, local_rank, world_size)
    
    # Create and setup model
    model = _create_and_setup_model(args, data_info, device, is_ddp)
    
    # Run training
    training_results = _run_training(args, model, data_loaders, device, is_ddp, data_info)
    
    # Run final evaluation
    final_results = _run_final_evaluation(args, model, data_loaders, device, is_ddp, data_info)
    
    # CRITICAL FIX: Save the best model with proper synchronization
    if is_main_process():
        print("="*80)
        print("SAVING TRAINED MODEL")
        print("="*80)
        print(f"[Rank 0] Starting model save to: {args.model_save_path}")
        
        try:
            _save_best_model(
                args, 
                training_results["model"], 
                data_info["preprocessing_pipeline"],
                final_results,
                is_ddp
            )
            
            # Verify the file actually exists and has content
            if not os.path.exists(args.model_save_path):
                raise RuntimeError(f"CRITICAL: Model file NOT FOUND after save: {args.model_save_path}")
            
            file_size = os.path.getsize(args.model_save_path)
            if file_size == 0:
                raise RuntimeError(f"CRITICAL: Model file is EMPTY: {args.model_save_path}")
            
            print(f"✅ [Rank 0] Model saved successfully:")
            print(f"   Path: {args.model_save_path}")
            print(f"   Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
            
        except Exception as e:
            print(f"❌ [Rank 0] FAILED to save model: {e}")
            raise
    
    # CRITICAL FIX: All ranks must wait for rank 0 to finish saving
    if is_ddp and dist.is_initialized():
        if not is_main_process():
            print(f"[Rank {local_rank}] Waiting for rank 0 to finish saving model...")
        
        dist.barrier()
        
        if is_main_process():
            print(f"[Rank 0] All ranks synchronized after model save")
        else:
            print(f"[Rank {local_rank}] Rank 0 finished saving, continuing...")
    
    # Extract embeddings if requested
    embedding_results = _extract_embeddings_if_requested(args, model, data_loaders, device)
    
    # Combine results
    results = {
        "mode": "training",
        "data_info": data_info,
        "training_results": training_results,
        "test_metrics": final_results,
        "model_path": args.model_save_path,
        **embedding_results
    }
    
    return results

def _load_and_preprocess_data(args) -> Dict[str, Any]:
    """Load and preprocess molecular data with optimized HDF5 handling."""
    if is_main_process():
        print("Loading and preprocessing data...")
    
    # Load raw data
    if args.data_path:
        if args.task_type == 'multitask':
            smiles_list, target_values = load_dataset_multitask(
                args.data_path, args.smiles_column, args.multi_target_columns_list
            )
        else:
            smiles_list, target_values = load_dataset_simple(
                args.data_path, args.smiles_column, args.target_column
            )
        
        # Split data
        smiles_train, target_train, smiles_val, target_val, smiles_test, target_test = split_dataset(
            smiles_list, target_values, args.train_split, args.val_split, args.test_split, 
            task_type=args.task_type
        )
    else:
        # Load separate files
        if args.task_type == 'multitask':
            smiles_train, target_train = load_dataset_multitask(
                args.train_data, args.smiles_column, args.multi_target_columns_list
            )
            smiles_val, target_val = load_dataset_multitask(
                args.val_data, args.smiles_column, args.multi_target_columns_list
            )
            smiles_test, target_test = load_dataset_multitask(
                args.test_data, args.smiles_column, args.multi_target_columns_list
            )
        else:
            smiles_train, target_train = load_dataset_simple(
                args.train_data, args.smiles_column, args.target_column
            )
            smiles_val, target_val = load_dataset_simple(
                args.val_data, args.smiles_column, args.target_column
            )
            smiles_test, target_test = load_dataset_simple(
                args.test_data, args.smiles_column, args.target_column
            )

    if args.iterable_dataset:
        # 🚀 OPTIMIZATION: Check if HDF5 files exist before expensive preprocessing
        if _check_hdf5_files_exist(args):
            # HDF5 files exist - skip expensive preprocessing and load info from files
            data_info = _load_hdf5_preprocessing_info(
                args, smiles_train, target_train, smiles_val, target_val, smiles_test, target_test
            )
        else:
            # HDF5 files don't exist - do full preprocessing and create them
            if is_main_process():
                print("HDF5 files not found - performing full preprocessing...")
            data_info = _preprocess_for_hdf5_storage(
                args, smiles_train, target_train, smiles_val, target_val, smiles_test, target_test
            )
    else:
        # For in-memory: Apply preprocessing and use processed targets for dataset creation
        data_info = _preprocess_for_inmemory_storage(
            args, smiles_train, target_train, smiles_val, target_val, smiles_test, target_test
        )
    
    if is_main_process():
        print(f"Data loaded: {len(smiles_train)} train, {len(smiles_val)} val, {len(smiles_test)} test")
        print(f"Number of tasks: {data_info['num_tasks']}")
    
    return data_info


def _preprocess_for_hdf5_storage(args, smiles_train, target_train, smiles_val, target_val, smiles_test, target_test) -> Dict[str, Any]:
    """
    For HDF5 datasets: Apply FULL preprocessing (SAE + Standard Scaling) and store processed data in HDF5.
    This way, HDF5 files contain ready-to-use processed data.
    """
    if is_main_process():
        print("HDF5 mode: Applying FULL preprocessing (SAE → Standard Scaling) and storing in HDF5...")
    
    # Create preprocessing configuration
    preprocessing_config = PreprocessingConfig(
        apply_sae=args.calculate_sae,
        sae_subtasks=args.sae_subtasks_list,
        apply_standard_scaling=(args.task_type in ['regression', 'multitask']),
        task_type=args.task_type,
        sae_percentile_cutoff=2.0
    )
    
    # Apply FULL preprocessing pipeline: SAE first, then Standard Scaling
    (processed_train_targets, _), \
    (processed_val_targets, _), \
    (processed_test_targets, _), \
    preprocessing_pipeline = preprocess_molecular_data(
        train_smiles=smiles_train,
        train_targets=target_train,
        val_smiles=smiles_val,
        val_targets=target_val,
        test_smiles=smiles_test,
        test_targets=target_test,
        config=preprocessing_config
    )
    
    # Get number of tasks
    num_tasks = preprocessing_pipeline.get_num_tasks(processed_train_targets)
    
    data_info = {
        "smiles_train": smiles_train,
        "smiles_val": smiles_val,
        "smiles_test": smiles_test,
        "target_train": processed_train_targets,  # PROCESSED targets for HDF5
        "target_val": processed_val_targets,      # PROCESSED targets for HDF5
        "target_test": processed_test_targets,    # PROCESSED targets for HDF5
        "num_tasks": num_tasks,
        "preprocessing_pipeline": preprocessing_pipeline,
        "max_hops": args.num_shells,
        "preprocessing_done": True,  # Flag to indicate preprocessing is complete
    }
    
    if is_main_process():
        print("✅ Full preprocessing applied for HDF5 storage:")
        if args.calculate_sae:
            print("   1. SAE normalization: applied")
        if preprocessing_config.apply_standard_scaling:
            print("   2. Standard scaling: applied")
        print("   → Processed data will be stored in HDF5")
    
    return data_info


def _preprocess_for_inmemory_storage(args, smiles_train, target_train, smiles_val, target_val, smiles_test, target_test) -> Dict[str, Any]:
    """
    For in-memory datasets: Apply preprocessing and use processed targets for dataset creation.
    """
    if is_main_process():
        print("In-memory mode: Applying preprocessing for immediate use...")
    
    # Create preprocessing configuration
    preprocessing_config = PreprocessingConfig(
        apply_sae=args.calculate_sae,
        sae_subtasks=args.sae_subtasks_list,
        apply_standard_scaling=(args.task_type in ['regression', 'multitask']),
        task_type=args.task_type,
        sae_percentile_cutoff=2.0
    )
    
    # Apply preprocessing with proper train/test isolation
    (processed_train_targets, _), \
    (processed_val_targets, _), \
    (processed_test_targets, _), \
    preprocessing_pipeline = preprocess_molecular_data(
        train_smiles=smiles_train,
        train_targets=target_train,
        val_smiles=smiles_val,
        val_targets=target_val,
        test_smiles=smiles_test,
        test_targets=target_test,
        config=preprocessing_config
    )
    
    # Get number of tasks
    num_tasks = preprocessing_pipeline.get_num_tasks(processed_train_targets)
    
    data_info = {
        "smiles_train": smiles_train,
        "smiles_val": smiles_val,
        "smiles_test": smiles_test,
        "target_train": processed_train_targets,
        "target_val": processed_val_targets,
        "target_test": processed_test_targets,
        "num_tasks": num_tasks,
        "preprocessing_pipeline": preprocessing_pipeline,
        "max_hops": args.num_shells,
        "preprocessing_done": True,
    }
    
    if is_main_process():
        print("✅ Preprocessing applied for in-memory datasets")
    
    return data_info


def _create_datasets(args, data_info: Dict[str, Any]) -> Dict[str, Any]:
    """Create PyTorch datasets from preprocessed data."""
    if args.iterable_dataset:
        return _create_hdf5_datasets(args, data_info)
    else:
        return _create_in_memory_datasets(args, data_info)


def _create_in_memory_datasets(args, data_info: Dict[str, Any]) -> Dict[str, Any]:
    """Create in-memory PyG datasets."""
    if is_main_process():
        print("Creating in-memory datasets...")
    
    # Precompute features for each split using processed targets
    smiles_train_valid, train_targets_valid, train_precomputed = precompute_all_and_filter(
        data_info["smiles_train"], data_info["target_train"], 
        data_info["max_hops"], num_workers=args.num_workers
    )
    
    smiles_val_valid, val_targets_valid, val_precomputed = precompute_all_and_filter(
        data_info["smiles_val"], data_info["target_val"], 
        data_info["max_hops"], num_workers=args.num_workers
    )
    
    smiles_test_valid, test_targets_valid, test_precomputed = precompute_all_and_filter(
        data_info["smiles_test"], data_info["target_test"], 
        data_info["max_hops"], num_workers=args.num_workers
    )
    
    # Create datasets
    train_dataset = PyGSMILESDataset(smiles_train_valid, train_targets_valid, train_precomputed)
    val_dataset = PyGSMILESDataset(smiles_val_valid, val_targets_valid, val_precomputed)
    test_dataset = PyGSMILESDataset(smiles_test_valid, test_targets_valid, test_precomputed)
    
    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "dataset_type": "in_memory"
    }

def _create_hdf5_datasets(args, data_info: Dict[str, Any]) -> Dict[str, Any]:
    """Create HDF5-based iterable datasets (optimized version)."""
    if is_main_process():
        print("Setting up HDF5 datasets...")
    
    # Setup HDF5 paths
    train_hdf5_path = args.train_hdf5 or "train.h5"
    val_hdf5_path = args.val_hdf5 or "val.h5"
    test_hdf5_path = args.test_hdf5 or "test.h5"
    
    # Check if HDF5 files need to be created
    need_precompute = not _check_hdf5_files_exist(args)
    
    # Only create HDF5 files if they don't exist
    if need_precompute:
        if is_main_process():
            print("Creating HDF5 files with PREPROCESSED data...")
            print("   → SAE normalization: applying during HDF5 creation")
            print("   → Standard scaling: applying during HDF5 creation") 
            print("   → HDF5 will contain ready-to-use processed targets")
            print("   → Storing preprocessing statistics in HDF5 metadata")
        
        precompute_workers = args.precompute_num_workers
        preprocessing_pipeline = data_info.get("preprocessing_pipeline")
        
        # Write PREPROCESSED data to HDF5 files with preprocessing metadata
        if is_main_process():
            # Use PROCESSED targets from the preprocessing pipeline
            _write_hdf5_with_preprocessing_metadata(
                data_info["smiles_train"], data_info["target_train"],
                data_info["max_hops"], train_hdf5_path,
                preprocessing_pipeline, args, precompute_workers,
                "train"
            )
            
            _write_hdf5_with_preprocessing_metadata(
                data_info["smiles_val"], data_info["target_val"],
                data_info["max_hops"], val_hdf5_path,
                preprocessing_pipeline, args, precompute_workers,
                "val"
            )
            
            _write_hdf5_with_preprocessing_metadata(
                data_info["smiles_test"], data_info["target_test"],
                data_info["max_hops"], test_hdf5_path,
                preprocessing_pipeline, args, precompute_workers,
                "test"
            )
            
            print("✅ HDF5 files created with FULLY PREPROCESSED data and metadata")
    else:
        if is_main_process():
            print("✅ Using existing HDF5 files with PREPROCESSED data")
    
    # Wait for HDF5 creation if using DDP
    if hasattr(dist, 'is_initialized') and dist.is_initialized():
        dist.barrier()
    
    return {
        "train_hdf5_path": train_hdf5_path,
        "val_hdf5_path": val_hdf5_path,
        "test_hdf5_path": test_hdf5_path,
        "dataset_type": "hdf5"
    }


def _write_hdf5_with_preprocessing_metadata(smiles_list, target_values, max_hops, 
                                          hdf5_path, preprocessing_pipeline, args, 
                                          num_workers, split_name):
    """
    Write HDF5 file with preprocessing metadata included.
    
    This wrapper around precompute_and_write_hdf5_parallel_chunked adds
    the preprocessing statistics to the HDF5 metadata.
    """
    from datasets import precompute_and_write_hdf5_parallel_chunked
    import h5py
    
    # First, create the HDF5 file with the standard function
    precompute_and_write_hdf5_parallel_chunked(
        smiles_list, target_values,
        max_hops, hdf5_path,
        chunk_size=1000, num_workers=num_workers,
        task_type=args.task_type,
        multi_target_columns=args.multi_target_columns_list,
        sae_subtasks=None,  # No additional SAE (already applied)
        preprocessing_applied=True
    )
    
    # Then, add the preprocessing statistics to the metadata
    if preprocessing_pipeline:
        with h5py.File(hdf5_path, 'a') as f:  # Open in append mode
            if 'metadata' in f:
                metadata = f['metadata']
                
                # Store standard scaler statistics
                if preprocessing_pipeline.standard_scaler and preprocessing_pipeline.standard_scaler.is_fitted:
                    metadata.attrs['scaler_means'] = preprocessing_pipeline.standard_scaler.means
                    metadata.attrs['scaler_stds'] = preprocessing_pipeline.standard_scaler.stds
                    print(f"   → Stored scaler statistics in {split_name} HDF5: means={preprocessing_pipeline.standard_scaler.means}, stds={preprocessing_pipeline.standard_scaler.stds}")
                
                # Store SAE statistics if available
                if preprocessing_pipeline.sae_normalizer and preprocessing_pipeline.sae_normalizer.is_fitted:
                    if preprocessing_pipeline.sae_normalizer.sae_statistics:
                        # Convert SAE statistics to a format that can be stored in HDF5
                        sae_data = {}
                        for key, value in preprocessing_pipeline.sae_normalizer.sae_statistics.items():
                            if isinstance(value, dict):
                                # Convert atomic number keys to strings for HDF5 storage
                                sae_data[str(key)] = {str(k): float(v) for k, v in value.items()}
                            else:
                                sae_data[str(key)] = float(value)
                        
                        # Store as JSON string in metadata
                        import json
                        metadata.attrs['sae_statistics'] = json.dumps(sae_data)
                        print(f"   → Stored SAE statistics in {split_name} HDF5")
                
                # Store preprocessing config
                if preprocessing_pipeline.config:
                    metadata.attrs['sae_applied'] = preprocessing_pipeline.config.apply_sae
                    metadata.attrs['standard_scaling_applied'] = preprocessing_pipeline.config.apply_standard_scaling
                    metadata.attrs['task_type'] = preprocessing_pipeline.config.task_type
                    if preprocessing_pipeline.config.sae_subtasks:
                        metadata.attrs['sae_subtasks'] = preprocessing_pipeline.config.sae_subtasks


def _create_data_loaders(args, datasets_info: Dict[str, Any], is_ddp: bool, 
                        local_rank: int, world_size: int) -> Dict[str, Any]:
    """Create data loaders for training."""
    if datasets_info["dataset_type"] == "in_memory":
        return _create_in_memory_data_loaders(args, datasets_info, is_ddp, local_rank, world_size)
    else:
        return _create_hdf5_data_loaders(args, datasets_info, is_ddp, local_rank, world_size)


def _create_in_memory_data_loaders(args, datasets_info: Dict[str, Any], is_ddp: bool,
                                  local_rank: int, world_size: int) -> Dict[str, Any]:
    """Create data loaders for in-memory datasets."""
    train_dataset = datasets_info["train_dataset"]
    val_dataset = datasets_info["val_dataset"]
    test_dataset = datasets_info["test_dataset"]
    
    # Setup samplers for DDP
    if is_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=local_rank, 
            shuffle=True, drop_last=False
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=world_size, rank=local_rank, 
            shuffle=False, drop_last=False
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=world_size, rank=local_rank, 
            shuffle=False, drop_last=False
        )
    else:
        train_sampler = val_sampler = test_sampler = None
    
    # Create data loaders
    train_loader = create_pyg_dataloader(
        dataset=train_dataset, batch_size=args.batch_size, 
        shuffle=(train_sampler is None), num_workers=args.num_workers, 
        sampler=train_sampler
    )
    
    val_loader = create_pyg_dataloader(
        dataset=val_dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=args.num_workers, 
        sampler=val_sampler
    )
    
    test_loader = create_pyg_dataloader(
        dataset=test_dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=args.num_workers, 
        sampler=test_sampler
    )
    
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }


def _create_hdf5_data_loaders(args, datasets_info: Dict[str, Any], is_ddp: bool,
                             local_rank: int, world_size: int) -> Dict[str, Any]:
    """Create data loaders for HDF5 datasets."""
    
    # Make sure we pass the correct DDP parameters
    train_loader = create_iterable_pyg_dataloader(
        hdf5_path=datasets_info["train_hdf5_path"], 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        shuffle_buffer_size=args.shuffle_buffer_size,
        ddp_enabled=is_ddp,  # This should be True for DDP
        rank=local_rank,     # This should be the local rank
        world_size=world_size  # This should be the world size
    )
    
    val_loader = create_iterable_pyg_dataloader(
        hdf5_path=datasets_info["val_hdf5_path"], 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        shuffle_buffer_size=args.shuffle_buffer_size,
        ddp_enabled=is_ddp,
        rank=local_rank,
        world_size=world_size
    )
    
    test_loader = create_iterable_pyg_dataloader(
        hdf5_path=datasets_info["test_hdf5_path"], 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        shuffle_buffer_size=args.shuffle_buffer_size,
        ddp_enabled=is_ddp,
        rank=local_rank,
        world_size=world_size
    )
    
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }


def _create_and_setup_model(args, data_info: Dict[str, Any], device: torch.device, 
                           is_ddp: bool) -> torch.nn.Module:
    """Create and setup the GNN model."""
    if is_main_process():
        print("Creating model...")
    
    # Define feature sizes
    feature_sizes = {
        'atom_type': len(ATOM_TYPES) + 1,
        'degree': len(DEGREES) + 1,
        'hybridization': len(HYBRIDIZATIONS) + 1,
        'hydrogen_count': 9
    }
    
    # Create model
    model = GNN(
        feature_sizes=feature_sizes,
        hidden_dim=args.hidden_dim,
        output_dim=data_info["num_tasks"],
        num_shells=args.num_shells,
        num_message_passing_layers=args.num_message_passing_layers,
        ffn_hidden_dim=args.ffn_hidden_dim,
        ffn_num_layers=args.ffn_num_layers,
        pooling_type=args.pooling_type,
        task_type=args.task_type,
        embedding_dim=args.embedding_dim,
        use_partial_charges=args.use_partial_charges,
        use_stereochemistry=args.use_stereochemistry,
        ffn_dropout=args.ffn_dropout,
        activation_type=args.activation_type,
        shell_conv_num_mlp_layers=args.shell_conv_num_mlp_layers,
        shell_conv_dropout=args.shell_conv_dropout,
        attention_num_heads=args.attention_num_heads,
        attention_temperature=args.attention_temperature,
        loss_function=args.loss_function
    )
    
    # Handle transfer learning
    if args.transfer_learning:
        _load_pretrained_weights(model, args)
    
    # Move to device
    model.to(device)
    
    # Setup DDP if needed
    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], output_device=device.index, 
            find_unused_parameters=True
        )
    
    if is_main_process():
        model_info = model.module.get_model_info() if is_ddp else model.get_model_info()
        print(f"Model created with {model_info['total_parameters']:,} parameters")
    
    return model

def _load_pretrained_weights(model: torch.nn.Module, args) -> None:
    """Load pretrained weights for transfer learning."""
    if not os.path.exists(args.transfer_learning):
        raise FileNotFoundError(f"Pretrained model not found: {args.transfer_learning}")
    
    if is_main_process():
        print(f"Loading pretrained weights from {args.transfer_learning}")
    
    # Load weights
    pretrained_weights = torch.load(args.transfer_learning, map_location="cpu")
    
    # Handle different save formats
    if "state_dict" in pretrained_weights:
        state_dict = pretrained_weights["state_dict"]
    else:
        state_dict = pretrained_weights
    
    # NEW: Handle different output dimensions for transfer learning
    model_state = model.state_dict()
    
    # Check if output layer dimensions match
    output_layer_key = "output_layer.weight"
    if output_layer_key in state_dict and output_layer_key in model_state:
        pretrained_shape = state_dict[output_layer_key].shape
        current_shape = model_state[output_layer_key].shape
        
        if pretrained_shape != current_shape:
            if is_main_process():
                print(f"Output dimension mismatch detected:")
                print(f"  Pretrained: {pretrained_shape}")
                print(f"  Current: {current_shape}")
                print(f"  Excluding output layer from transfer learning")
            
            # Remove output layer and bias from pretrained weights
            state_dict.pop("output_layer.weight", None)
            state_dict.pop("output_layer.bias", None)
    
    # Load with strict=False to allow missing keys (output layer)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if is_main_process():
        if missing_keys:
            print(f"Missing keys (will use random initialization): {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys (ignored): {unexpected_keys}")
    
    # Handle freezing - output layer will NOT be frozen since it wasn't loaded
    if args.freeze_pretrained:
        if is_main_process():
            print("Freezing pretrained layers (output layer remains trainable)")
        
        for name, param in model.named_parameters():
            # Don't freeze output layer - it needs to be trained from scratch
            if "output_layer" not in name:
                param.requires_grad = False
    
    # Handle specific layer freezing/unfreezing
    if args.freeze_layers:
        from utils.optimization import freeze_parameters
        freeze_patterns = [pattern.strip() for pattern in args.freeze_layers.split(',')]
        freeze_parameters(model, freeze_patterns)
    
    if args.unfreeze_layers:
        from utils.optimization import unfreeze_parameters
        unfreeze_patterns = [pattern.strip() for pattern in args.unfreeze_layers.split(',')]
        unfreeze_parameters(model, unfreeze_patterns)

def _run_training(args, model: torch.nn.Module, data_loaders: Dict[str, Any], 
                 device: torch.device, is_ddp: bool, data_info: Dict[str, Any]) -> Dict[str, Any]:
    """Run the training process."""
    if is_main_process():
        print("Starting training...")
    
    # Setup multitask weights
    multitask_weights = None
    if args.task_type == 'multitask':
        if args.multitask_weights_list:
            multitask_weights = args.multitask_weights_list
        else:
            multitask_weights = np.ones(data_info["num_tasks"], dtype=float)
    
    # Record training start time
    training_start_time = time.time()
    
    # Get the preprocessing pipeline
    preprocessing_pipeline = data_info["preprocessing_pipeline"]
    
    # Run training
    trained_model = train_gnn(
        model=model,
        train_loader=data_loaders["train_loader"],
        val_loader=data_loaders["val_loader"],
        test_loader=data_loaders["test_loader"],
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        early_stopping=args.early_stopping,
        task_type=args.task_type,
        mixed_precision=args.mixed_precision,
        num_tasks=data_info["num_tasks"],
        multitask_weights=multitask_weights,
        preprocessing_pipeline=preprocessing_pipeline,  # CHANGED: was std_scaler
        is_ddp=is_ddp,
        current_args=args
    )
    
    training_time = time.time() - training_start_time
    
    if is_main_process():
        print(f"Training completed in {training_time:.2f} seconds")
    
    return {
        "training_time": training_time,
        "model": trained_model,
    }

def _run_final_evaluation(args, model: torch.nn.Module, data_loaders: Dict[str, Any],
                         device: torch.device, is_ddp: bool, data_info: Dict[str, Any]) -> Dict[str, Any]:
    """Run final evaluation on test set."""
    if is_main_process():
        print("Running final evaluation...")
    
    # Setup loss function for evaluation
    if args.loss_function == 'l1':
        if args.task_type == 'multitask':
            multitask_weights = args.multitask_weights_list or np.ones(data_info["num_tasks"])
            w_tensor = torch.tensor(multitask_weights, dtype=torch.float)
            criterion = WeightedL1Loss(w_tensor)
        else:
            criterion = torch.nn.L1Loss()
    elif args.loss_function == 'mse':
        if args.task_type == 'multitask':
            multitask_weights = args.multitask_weights_list or np.ones(data_info["num_tasks"])
            w_tensor = torch.tensor(multitask_weights, dtype=torch.float)
            criterion = WeightedMSELoss(w_tensor)
        else:
            criterion = torch.nn.MSELoss()
    elif args.loss_function == 'evidential':
        lambda_reg = getattr(args, 'evidential_lambda', 1.0)
        if args.task_type == 'multitask':
            multitask_weights = args.multitask_weights_list or np.ones(data_info["num_tasks"])
            w_tensor = torch.tensor(multitask_weights, dtype=torch.float)
            criterion = WeightedEvidentialLoss(w_tensor, lambda_reg=lambda_reg)
        else:
            criterion = EvidentialLoss(lambda_reg=lambda_reg)
    else:
        raise ValueError(f"Invalid loss function: {args.loss_function}")
    
    # Get the preprocessing pipeline
    preprocessing_pipeline = data_info["preprocessing_pipeline"]
    
    # Evaluate on test set
    test_metrics = evaluate(
        model=model,
        data_loader=data_loaders["test_loader"],
        criterion=criterion,
        device=device,
        task_type=args.task_type,
        mixed_precision=args.mixed_precision,
        num_tasks=data_info["num_tasks"],
        preprocessing_pipeline=preprocessing_pipeline,  # CHANGED: was std_scaler
        is_ddp=is_ddp
    )
    
    if is_main_process():
        print("Final evaluation completed")
    
    return test_metrics

def _extract_embeddings_if_requested(args, model: torch.nn.Module, data_loaders: Dict[str, Any],
                                   device: torch.device) -> Dict[str, Any]:
    """Extract molecular embeddings if requested."""
    if not args.save_embeddings or not is_main_process():
        return {}
    
    print("Extracting molecular embeddings...")
    
    # Ensure output directory exists
    embeddings_dir = os.path.dirname(os.path.abspath(args.embeddings_output_path))
    if embeddings_dir:
        os.makedirs(embeddings_dir, exist_ok=True)
    
    # Extract embeddings
    extract_embeddings_main(
        args=args,
        model=model,
        train_loader=data_loaders["train_loader"],
        val_loader=data_loaders["val_loader"],
        test_loader=data_loaders["test_loader"],
        device=device
    )
    
    print(f"Embeddings saved to: {args.embeddings_output_path}")
    
    return {"embeddings_path": args.embeddings_output_path}

def _save_best_model(args, model: torch.nn.Module, preprocessing_pipeline, 
                    test_metrics: Dict[str, Any], is_ddp: bool) -> None:
    """Save the best model with COMPLETE preprocessing pipeline information."""
    if not is_main_process():
        return
    
    # Get model state dict
    if is_ddp:
        model_state_dict = {k: v.cpu() for k, v in model.module.state_dict().items()}
        model_for_config = model.module
    else:
        model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        model_for_config = model
    
    # CRITICAL FIX: Save COMPLETE model configuration
    model_artifact = {
        "hyperparams": {
            # Core model architecture - COMPLETE SET
            "task_type": args.task_type,
            "num_shells": args.num_shells,
            "hidden_dim": args.hidden_dim,
            "num_message_passing_layers": args.num_message_passing_layers,
            "ffn_hidden_dim": args.ffn_hidden_dim,
            "ffn_num_layers": args.ffn_num_layers,
            "pooling_type": args.pooling_type,
            "embedding_dim": args.embedding_dim,
            "use_partial_charges": args.use_partial_charges,
            "use_stereochemistry": args.use_stereochemistry,
            "ffn_dropout": args.ffn_dropout,
            "learning_rate": args.learning_rate,
            "activation_type": args.activation_type,
            "shell_conv_num_mlp_layers": args.shell_conv_num_mlp_layers,
            "shell_conv_dropout": args.shell_conv_dropout,
            "attention_num_heads": args.attention_num_heads,
            "attention_temperature": args.attention_temperature,
            "loss_function": args.loss_function,
            "evidential_lambda": getattr(args, 'evidential_lambda', 1.0),
            "best_val_loss": test_metrics.get("loss", float('inf')),
            
            # CRITICAL FIX: Save feature sizes used during training
            "feature_sizes": {
                'atom_type': len(ATOM_TYPES) + 1,
                'hydrogen_count': 9,
                'degree': len(DEGREES) + 1,
                'hybridization': len(HYBRIDIZATIONS) + 1,
            },
            
            # CRITICAL FIX: Save data configuration used during training
            "data_config": {
                "smiles_column": args.smiles_column,
                "target_column": getattr(args, 'target_column', None),
                "multi_target_columns": getattr(args, 'multi_target_columns', None),
                "multitask_weights": getattr(args, 'multitask_weights', None),
            },
            
            # NEW: Save target column information for inference output naming
            "target_column_name": getattr(args, 'target_column', 'target'),
            "multi_target_column_names": getattr(args, 'multi_target_columns_list', None),
            
            # CRITICAL FIX: Complete preprocessing pipeline information
            "preprocessing_config": {
                "apply_sae": preprocessing_pipeline.config.apply_sae if preprocessing_pipeline else False,
                "sae_subtasks": preprocessing_pipeline.config.sae_subtasks if preprocessing_pipeline else None,
                "apply_standard_scaling": preprocessing_pipeline.config.apply_standard_scaling if preprocessing_pipeline else True,
                "task_type": preprocessing_pipeline.config.task_type if preprocessing_pipeline else args.task_type,
                "sae_percentile_cutoff": preprocessing_pipeline.config.sae_percentile_cutoff if preprocessing_pipeline else 2.0
            },
            
            # Scaler parameters - REQUIRED for inference
            "scaler_means": preprocessing_pipeline.standard_scaler.means.tolist() if (
                preprocessing_pipeline and preprocessing_pipeline.standard_scaler and preprocessing_pipeline.standard_scaler.is_fitted
            ) else None,
            "scaler_stds": preprocessing_pipeline.standard_scaler.stds.tolist() if (
                preprocessing_pipeline and preprocessing_pipeline.standard_scaler and preprocessing_pipeline.standard_scaler.is_fitted
            ) else None,
            
            # SAE statistics - REQUIRED for inference if SAE was used
            "sae_statistics": preprocessing_pipeline.sae_normalizer.sae_statistics if (
                preprocessing_pipeline and preprocessing_pipeline.sae_normalizer and preprocessing_pipeline.sae_normalizer.is_fitted
            ) else None,
        },
        "state_dict": model_state_dict
    }
    
    # Ensure output directory exists
    model_dir = os.path.dirname(os.path.abspath(args.model_save_path))
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        print(f"✓ Output directory ready: {model_dir}")
    
    # CRITICAL FIX: Save with explicit error handling
    try:
        print(f"[Rank 0] Writing {len(model_state_dict)} parameter tensors to disk...")
        torch.save(model_artifact, args.model_save_path)
        
        # CRITICAL FIX: Force OS to flush buffers to disk
        print(f"[Rank 0] Syncing file buffers to disk...")
        with open(args.model_save_path, 'rb') as f:
            os.fsync(f.fileno())
        
        # Verify the save
        actual_size = os.path.getsize(args.model_save_path)
        print(f"✅ Model artifact written and synced to disk")
        print(f"   File: {args.model_save_path}")
        print(f"   Size: {actual_size:,} bytes ({actual_size/1024/1024:.2f} MB)")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR during model save: {e}")
        print(f"   Attempted path: {args.model_save_path}")
        raise


def _check_hdf5_files_exist(args) -> bool:
    """
    Check if all required HDF5 files exist.
    
    Args:
        args: Command line arguments
        
    Returns:
        True if all HDF5 files exist, False otherwise
    """
    train_hdf5_path = args.train_hdf5 or "train.h5"
    val_hdf5_path = args.val_hdf5 or "val.h5"
    test_hdf5_path = args.test_hdf5 or "test.h5"
    
    files_exist = (
        os.path.exists(train_hdf5_path) and
        os.path.exists(val_hdf5_path) and
        os.path.exists(test_hdf5_path)
    )
    
    if files_exist and is_main_process():
        print("✅ All HDF5 files found:")
        print(f"   - Train: {train_hdf5_path}")
        print(f"   - Val: {val_hdf5_path}")
        print(f"   - Test: {test_hdf5_path}")
        
    return files_exist


def _load_hdf5_preprocessing_info(args, smiles_train, target_train, smiles_val, 
                                   target_val, smiles_test, target_test) -> Dict[str, Any]:
    """
    Load preprocessing information when HDF5 files already exist.
    
    CRITICAL: HDF5 files MUST contain PREPROCESSED data for training.
    This function validates that and reconstructs the preprocessing pipeline.
    """
    if is_main_process():
        print("="*80)
        print("LOADING EXISTING HDF5 FILES")
        print("="*80)
    
    train_hdf5_path = args.train_hdf5 or "train.h5"
    
    # CRITICAL: Validate HDF5 file is readable and has correct format
    try:
        with h5py.File(train_hdf5_path, 'r') as f:
            if 'metadata' not in f:
                raise ValueError(
                    f"HDF5 file {train_hdf5_path} is missing metadata section.\n\n"
                    "This file is CORRUPTED or created with an incompatible version.\n\n"
                    "SOLUTION:\n"
                    "1. Delete the HDF5 files:\n"
                    f"   rm {args.train_hdf5} {args.val_hdf5} {args.test_hdf5}\n\n"
                    "2. Re-run training - files will be recreated automatically"
                )
            
            metadata = f['metadata']
            
            # CRITICAL: Check preprocessing status
            preprocessing_applied = metadata.attrs.get('preprocessing_applied', None)
            
            if preprocessing_applied is None:
                raise ValueError(
                    f"Cannot determine if {train_hdf5_path} contains preprocessed data.\n\n"
                    "The 'preprocessing_applied' flag is missing from metadata.\n\n"
                    "SOLUTION: Delete HDF5 files and recreate:\n"
                    f"  rm {args.train_hdf5} {args.val_hdf5} {args.test_hdf5}\n"
                    "  Then re-run training."
                )
            
            if not preprocessing_applied:
                raise ValueError(
                    f"CRITICAL: {train_hdf5_path} contains RAW (unpreprocessed) data!\n\n"
                    "For TRAINING, HDF5 files MUST contain PREPROCESSED data.\n"
                    "For INFERENCE, HDF5 files MUST contain RAW data.\n\n"
                    "You are trying to TRAIN, so you need preprocessed data.\n\n"
                    "SOLUTION:\n"
                    "1. Delete the current HDF5 files:\n"
                    f"   rm {args.train_hdf5} {args.val_hdf5} {args.test_hdf5}\n\n"
                    "2. Re-run training - files will be recreated with preprocessing"
                )
            
            # Load scaler statistics
            if 'scaler_means' not in metadata.attrs:
                raise ValueError(
                    f"{train_hdf5_path} is missing scaler statistics in metadata.\n\n"
                    "The preprocessing pipeline cannot be reconstructed.\n\n"
                    "SOLUTION: Delete HDF5 files and recreate:\n"
                    f"  rm {args.train_hdf5} {args.val_hdf5} {args.test_hdf5}"
                )
            
            actual_scaler_means = np.array(metadata.attrs['scaler_means'])
            actual_scaler_stds = np.array(metadata.attrs['scaler_stds'])
            
            # Validate scaler statistics are reasonable
            if np.any(np.isnan(actual_scaler_means)) or np.any(np.isnan(actual_scaler_stds)):
                raise ValueError(
                    f"Scaler statistics in {train_hdf5_path} contain NaN values!\n\n"
                    "The HDF5 file is CORRUPTED.\n\n"
                    "SOLUTION: Delete and recreate HDF5 files."
                )
            
            if np.any(actual_scaler_stds <= 0):
                raise ValueError(
                    f"Scaler standard deviations in {train_hdf5_path} are <= 0!\n\n"
                    "This means your data has zero variance (all same values).\n\n"
                    "Check your input CSV files for data problems."
                )
            
            # Load SAE information if present
            sae_applied = metadata.attrs.get('sae_applied', False)
            sae_statistics = None
            
            if sae_applied and 'sae_statistics' in metadata.attrs:
                import json
                try:
                    sae_data = json.loads(metadata.attrs['sae_statistics'])
                    sae_statistics = {}
                    for key, value in sae_data.items():
                        if isinstance(value, dict):
                            sae_statistics[key if key == "regression" else int(key)] = {
                                int(k): float(v) for k, v in value.items()
                            }
                        else:
                            sae_statistics[key if key == "regression" else int(key)] = float(value)
                except Exception as e:
                    raise ValueError(
                        f"Failed to load SAE statistics from {train_hdf5_path}.\n"
                        f"Error: {e}\n\n"
                        "The HDF5 file may be corrupted.\n\n"
                        "SOLUTION: Delete and recreate HDF5 files."
                    )
                    
    except OSError as e:
        raise ValueError(
            f"Cannot open HDF5 file: {train_hdf5_path}\n"
            f"Error: {e}\n\n"
            "The file may be corrupted, locked by another process, or permissions issue.\n\n"
            "SOLUTION:\n"
            "1. Make sure no other training process is running\n"
            "2. Check file permissions\n"
            "3. If problem persists, delete and recreate:\n"
            f"   rm {train_hdf5_path}"
        )
    
    # Reconstruct the preprocessing pipeline
    preprocessing_config = PreprocessingConfig(
        apply_sae=sae_applied,
        sae_subtasks=args.sae_subtasks_list,
        apply_standard_scaling=True,
        task_type=args.task_type,
        sae_percentile_cutoff=2.0
    )
    
    preprocessing_pipeline = PreprocessingPipeline(preprocessing_config)
    
    # Restore SAE normalizer if used
    if sae_applied and sae_statistics:
        from data.preprocessing import SAENormalizer
        preprocessing_pipeline.sae_normalizer = SAENormalizer(
            task_type=preprocessing_config.task_type,
            percentile_cutoff=preprocessing_config.sae_percentile_cutoff
        )
        preprocessing_pipeline.sae_normalizer.sae_statistics = sae_statistics
        preprocessing_pipeline.sae_normalizer.is_fitted = True
    
    # Restore standard scaler
    from data.preprocessing import StandardScaler
    preprocessing_pipeline.standard_scaler = StandardScaler()
    preprocessing_pipeline.standard_scaler.means = actual_scaler_means
    preprocessing_pipeline.standard_scaler.stds = actual_scaler_stds
    preprocessing_pipeline.standard_scaler.is_fitted = True
    
    preprocessing_pipeline.is_fitted = True
    
    # Get number of tasks
    if args.task_type == 'multitask':
        num_tasks = len(args.multi_target_columns_list)
    else:
        num_tasks = 1
    
    if is_main_process():
        print(f"✓ Successfully loaded preprocessing pipeline from HDF5")
        print(f"  • SAE applied: {sae_applied}")
        print(f"  • Scaler means: {actual_scaler_means}")
        print(f"  • Scaler stds: {actual_scaler_stds}")
        print(f"  • Number of tasks: {num_tasks}")
        print("="*80 + "\n")
    
    return {
        "smiles_train": smiles_train,
        "smiles_val": smiles_val,
        "smiles_test": smiles_test,
        "target_train": target_train,  # Original for reference
        "target_val": target_val,
        "target_test": target_test,
        "num_tasks": num_tasks,
        "preprocessing_pipeline": preprocessing_pipeline,
        "max_hops": args.num_shells,
        "preprocessing_done": True,
        "hdf5_contains_preprocessed_data": True,
    }

def run_single_trial(args) -> Dict[str, Any]:
    """
    Run a single trial/experiment.
    
    This is the main function called by hyperparameter optimization.
    
    Args:
        args: Trial-specific arguments
        
    Returns:
        Dictionary containing trial results
    """
    # Validate arguments for this trial
    if not validate_trial_arguments(args):
        raise ValueError("Trial arguments validation failed")
    
    # Setup distributed environment
    device, is_ddp, local_rank, world_size = setup_distributed_environment(args)
    
    # Run training mode (inference mode handled separately)
    results = _run_training_mode(args, device, is_ddp, local_rank, world_size)
    
    # Save model ONLY if this is NOT a hyperparameter optimization trial
    # (hyperopt will handle saving the best model separately)
    if not hasattr(args, '_trial_temp_dir') and not hasattr(args, '_is_hyperopt_trial'):
        _save_best_model(
            args, 
            results["training_results"]["model"], 
            results["data_info"]["preprocessing_pipeline"],
            results["test_metrics"],
            is_ddp
        )
    else:
        # For hyperopt trials, just indicate the model is ready but don't save yet
        if is_main_process():
            print(f"[Trial] Model ready for evaluation (will save best after all trials)")
    
    # Extract key metrics for hyperparameter optimization
    trial_results = {
        "val_loss": results["test_metrics"].get("loss", float('inf')),
        "val_mae": results["test_metrics"].get("mae", float('inf')),
        "val_rmse": results["test_metrics"].get("rmse", float('inf')),
        "val_r2": results["test_metrics"].get("r2", -float('inf')),
        "training_time": results["training_results"].get("training_time", 0),
        "epoch": results.get("best_epoch", args.epochs),
        # Store the model state for potential saving later
        "_model_state": results["training_results"]["model"],
        "_preprocessing_pipeline": results["data_info"]["preprocessing_pipeline"],
        "_test_metrics": results["test_metrics"],
    }
    
    # Add per-task metrics for multitask
    if args.task_type == "multitask" and "mae_per_target" in results["test_metrics"]:
        trial_results["val_mae_per_target"] = results["test_metrics"]["mae_per_target"]
        trial_results["val_rmse_per_target"] = results["test_metrics"]["rmse_per_target"]
        trial_results["val_r2_per_target"] = results["test_metrics"]["r2_per_target"]
    
    # Cleanup if this is a trial
    if hasattr(args, '_trial_temp_dir'):
        cleanup_temporary_files(args)
    
    return trial_results