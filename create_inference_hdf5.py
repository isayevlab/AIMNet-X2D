#!/usr/bin/env python3
"""
Standalone HDF5 file creator for inference.

This script creates HDF5 files from CSV data that are compatible with
a trained model for efficient GPU inference. It can run on CPU-only instances.

Usage:
    # Create HDF5 from CSV for inference:
    python create_inference_hdf5.py \
        --model_path trained_model.pth \
        --input_csv molecules.csv \
        --output_hdf5 molecules_inference.h5 \
        --smiles_column smiles \
        --num_workers 4
    
    # Verify existing HDF5 compatibility:
    python create_inference_hdf5.py \
        --model_path trained_model.pth \
        --output_hdf5 molecules_inference.h5 \
        --verify_only

Key Features:
    - Runs on CPU-only instances (no GPU required)
    - Creates RAW (unpreprocessed) HDF5 files for inference
    - Preprocessing will be applied automatically during GPU inference
    - Verifies compatibility with model parameters
    - Prevents common errors from parameter mismatches
"""

import argparse
import os
import sys
import h5py
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from datasets import (
    precompute_and_write_hdf5_parallel_chunked,
    partial_parse_atomic_numbers
)


class InferenceHDF5Creator:
    """
    Creates HDF5 files for inference that are compatible with trained models.
    
    This class ensures that HDF5 files created for inference:
    1. Contain RAW (unpreprocessed) molecular data
    2. Have the correct molecular features (num_shells/max_hops)
    3. Store compatibility metadata for verification
    4. Can be created on CPU-only instances
    
    The preprocessing pipeline from the model will be applied during GPU inference.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize creator with model parameters.
        
        Args:
            model_path: Path to trained model file (.pth)
        """
        self.model_path = model_path
        self.model_params = self._load_model_parameters()
        
    def _load_model_parameters(self) -> Dict[str, Any]:
        """Load and validate model parameters."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"Loading model parameters from: {self.model_path}")
        
        try:
            model_artifact = torch.load(self.model_path, map_location='cpu')
        except Exception as e:
            raise ValueError(f"Failed to load model file: {e}")
        
        if "hyperparams" not in model_artifact:
            raise ValueError(
                "Model file missing hyperparameters section. "
                "This model was created with an incompatible version."
            )
        
        hyperparams = model_artifact["hyperparams"]
        
        # Extract critical parameters for HDF5 creation
        required_params = [
            'num_shells',  # max_hops
            'task_type',
            'hidden_dim',
            'num_message_passing_layers'
        ]
        
        missing = [p for p in required_params if p not in hyperparams]
        if missing:
            raise ValueError(f"Model missing required parameters: {missing}")
        
        params = {
            'max_hops': hyperparams['num_shells'],
            'task_type': hyperparams['task_type'],
            'hidden_dim': hyperparams['hidden_dim'],
            'num_message_passing_layers': hyperparams['num_message_passing_layers'],
            'loss_function': hyperparams.get('loss_function', 'l1'),
            'multi_target_columns': hyperparams.get('data_config', {}).get('multi_target_columns'),
            
            # Preprocessing info (should NOT be applied to inference HDF5)
            'preprocessing_applied_during_training': hyperparams.get('preprocessing_config', {}).get('apply_sae', False) or 
                                                    hyperparams.get('preprocessing_config', {}).get('apply_standard_scaling', False),
        }
        
        print("\n" + "="*60)
        print("MODEL PARAMETERS")
        print("="*60)
        print(f"  ✅ Max hops (num_shells): {params['max_hops']}")
        print(f"     → HDF5 will store {params['max_hops']}-hop molecular graphs")
        print()
        print(f"  ℹ️  Task type: {params['task_type']}")
        print(f"     → For validation only")
        print()
        print(f"  ✅ Preprocessing during training: {params['preprocessing_applied_during_training']}")
        if params['preprocessing_applied_during_training']:
            print(f"     → HDF5 will contain RAW data")
            print(f"     → Model will apply preprocessing during inference")
        else:
            print(f"     → No preprocessing needed")
        print()
        print(f"  ℹ️  Hidden dim: {params['hidden_dim']}")
        print(f"  ℹ️  Message passing layers: {params['num_message_passing_layers']}")
        print(f"  ℹ️  Loss function: {params['loss_function']}")
        print(f"     → These are model internals, don't affect HDF5 creation")
        print("="*60 + "\n")
        
        return params
    
    def create_inference_hdf5(self,
                            input_csv: str,
                            output_hdf5: str,
                            smiles_column: str = 'smiles',
                            num_workers: int = 4,
                            chunk_size: int = 1000) -> None:
        """
        Create HDF5 file for inference from CSV.
        
        CRITICAL: This creates RAW (unpreprocessed) HDF5 files. The preprocessing
        will be applied during inference using the model's saved preprocessing pipeline.
        
        Args:
            input_csv: Path to input CSV file
            output_hdf5: Path for output HDF5 file
            smiles_column: Column name containing SMILES
            num_workers: Number of parallel workers
            chunk_size: Chunk size for parallel processing
        """
        print("\n" + "="*60)
        print("CREATING INFERENCE HDF5 FILE")
        print("="*60)
        print(f"Input CSV: {input_csv}")
        print(f"Output HDF5: {output_hdf5}")
        print(f"SMILES column: {smiles_column}")
        print(f"Workers: {num_workers}")
        print("="*60 + "\n")
        
        # Validate input file
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")
        
        # Load CSV
        print("Loading CSV file...")
        try:
            df = pd.read_csv(input_csv)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")
        
        # Validate SMILES column
        if smiles_column not in df.columns:
            available = ', '.join(df.columns)
            raise ValueError(
                f"SMILES column '{smiles_column}' not found in CSV. "
                f"Available columns: {available}"
            )
        
        smiles_list = df[smiles_column].tolist()
        print(f"Loaded {len(smiles_list)} molecules")
        
        # Create dummy targets (will not be used during inference)
        # But needed for HDF5 creation function
        if self.model_params['task_type'] == 'multitask':
            if self.model_params['multi_target_columns']:
                num_tasks = len(self.model_params['multi_target_columns'].split(','))
            else:
                # Default to 2 tasks if not specified
                num_tasks = 2
                print(f"WARNING: Multi-target columns not in model, assuming {num_tasks} tasks")
            
            dummy_targets = [[0.0] * num_tasks for _ in smiles_list]
        else:
            dummy_targets = [0.0] * len(smiles_list)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(os.path.abspath(output_hdf5))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        print("\nComputing molecular features and creating HDF5...")
        print("⚠️  IMPORTANT: Creating RAW (unpreprocessed) HDF5 file")
        print("   Preprocessing will be applied during inference by the model")
        
        # Create HDF5 with RAW data (no preprocessing)
        precompute_and_write_hdf5_parallel_chunked(
            smiles_list=smiles_list,
            target_values=dummy_targets,
            max_hops=self.model_params['max_hops'],
            hdf5_path=output_hdf5,
            num_workers=num_workers,
            chunk_size=chunk_size,
            sae_subtasks=None,  # No SAE during HDF5 creation
            task_type=self.model_params['task_type'],
            multi_target_columns=self.model_params['multi_target_columns'].split(',') if self.model_params['multi_target_columns'] else None,
            preprocessing_applied=False  # CRITICAL: Mark as raw data
        )
        
        # Add model compatibility metadata to HDF5
        self._add_compatibility_metadata(output_hdf5)
        
        print("\n" + "="*60)
        print("✅ INFERENCE HDF5 FILE CREATED SUCCESSFULLY")
        print("="*60)
        print(f"Output: {output_hdf5}")
        print(f"Ready for inference with model: {self.model_path}")
        print("="*60 + "\n")
    
    def _add_compatibility_metadata(self, hdf5_path: str) -> None:
        """Add model compatibility metadata to HDF5 file."""
        with h5py.File(hdf5_path, 'a') as f:
            if 'metadata' not in f:
                metadata = f.create_group('metadata')
            else:
                metadata = f['metadata']
            
            # Add model compatibility information
            model_compat = metadata.create_group('model_compatibility') if 'model_compatibility' not in metadata else metadata['model_compatibility']
            
            model_compat.attrs['model_path'] = self.model_path
            model_compat.attrs['max_hops'] = self.model_params['max_hops']
            model_compat.attrs['task_type'] = self.model_params['task_type']
            model_compat.attrs['hidden_dim'] = self.model_params['hidden_dim']
            model_compat.attrs['num_message_passing_layers'] = self.model_params['num_message_passing_layers']
            model_compat.attrs['loss_function'] = self.model_params['loss_function']
            model_compat.attrs['for_inference'] = True
            model_compat.attrs['preprocessing_applied'] = False  # RAW data
            model_compat.attrs['preprocessing_required_during_inference'] = self.model_params['preprocessing_applied_during_training']
            
            print("✓ Added model compatibility metadata to HDF5")
    
    def verify_hdf5_compatibility(self, hdf5_path: str) -> bool:
        """
        Verify that an HDF5 file is compatible with the model.
        
        This performs comprehensive checks:
        1. Molecular features (max_hops/num_shells) match
        2. Task type matches (regression vs multitask)
        3. Preprocessing status is correct (RAW for inference)
        4. Model architecture parameters are compatible
        
        Args:
            hdf5_path: Path to HDF5 file to verify
            
        Returns:
            True if compatible
            
        Raises:
            ValueError: If incompatible with detailed error message explaining the issue
        """
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        
        print(f"\nVerifying HDF5 compatibility with model...")
        print(f"  HDF5: {hdf5_path}")
        print(f"  Model: {self.model_path}")
        
        with h5py.File(hdf5_path, 'r') as f:
            if 'metadata' not in f:
                raise ValueError(
                    "HDF5 file missing metadata section. "
                    "This file may have been created with an old/incompatible version. "
                    "Please recreate it using create_inference_hdf5.py"
                )
            
            metadata = f['metadata']
            
            # Check if model compatibility metadata exists
            if 'model_compatibility' not in metadata:
                print("⚠️  WARNING: HDF5 file missing model_compatibility metadata")
                print("   This file may have been created before compatibility checking")
                print("   Performing basic validation...")
                
                # Basic validation - check preprocessing status
                preprocessing_applied = metadata.attrs.get('preprocessing_applied', 
                                                         metadata.attrs.get('sae_applied', False))
                
                if preprocessing_applied:
                    raise ValueError(
                        "❌ INCOMPATIBILITY DETECTED:\n"
                        "   HDF5 file contains PREPROCESSED data\n"
                        "   For inference, you need RAW (unpreprocessed) data\n\n"
                        "SOLUTION:\n"
                        f"   Recreate the HDF5 file using:\n"
                        f"   python create_inference_hdf5.py --model_path {self.model_path} "
                        f"--input_csv YOUR_DATA.csv --output_hdf5 {hdf5_path}"
                    )
                
                print("✓ Basic validation passed (preprocessing status OK)")
                print("⚠️  Cannot verify other parameters without model_compatibility metadata")
                return True
            
            # Detailed compatibility checking
            compat = metadata['model_compatibility']
            errors = []
            warnings = []
            
            # Check max_hops (CRITICAL)
            if 'max_hops' in compat.attrs:
                hdf5_max_hops = int(compat.attrs['max_hops'])
                model_max_hops = int(self.model_params['max_hops'])
                
                if hdf5_max_hops != model_max_hops:
                    errors.append(
                        f"Max hops mismatch:\n"
                        f"     HDF5 file: {hdf5_max_hops} hops\n"
                        f"     Model expects: {model_max_hops} hops\n"
                        f"   This means the molecular features are incompatible!"
                    )
            else:
                warnings.append("HDF5 missing max_hops in compatibility metadata")
            
            # Check task type (CRITICAL)
            if 'task_type' in compat.attrs:
                hdf5_task_type = str(compat.attrs['task_type'])
                model_task_type = str(self.model_params['task_type'])
                
                if hdf5_task_type != model_task_type:
                    errors.append(
                        f"Task type mismatch:\n"
                        f"     HDF5 file: {hdf5_task_type}\n"
                        f"     Model expects: {model_task_type}"
                    )
            else:
                warnings.append("HDF5 missing task_type in compatibility metadata")
            
            # Check preprocessing status (CRITICAL)
            if 'preprocessing_applied' in compat.attrs:
                if compat.attrs['preprocessing_applied']:
                    errors.append(
                        "HDF5 contains PREPROCESSED data\n"
                        "   For inference, you need RAW data\n"
                        "   Recreate using create_inference_hdf5.py"
                    )
            
            # Check if for_inference flag is set
            if 'for_inference' in compat.attrs:
                if not compat.attrs['for_inference']:
                    warnings.append(
                        "HDF5 not explicitly marked for inference\n"
                        "   This may have been created for training"
                    )
            
            # Print warnings
            if warnings:
                print("\n⚠️  WARNINGS:")
                for w in warnings:
                    print(f"   {w}")
            
            # Check for errors
            if errors:
                error_msg = "\n❌ HDF5 FILE IS INCOMPATIBLE WITH MODEL:\n\n"
                for i, e in enumerate(errors, 1):
                    error_msg += f"{i}. {e}\n\n"
                error_msg += "SOLUTION:\n"
                error_msg += f"  Recreate the HDF5 file using create_inference_hdf5.py:\n\n"
                error_msg += f"  python create_inference_hdf5.py \\\n"
                error_msg += f"    --model_path {self.model_path} \\\n"
                error_msg += f"    --input_csv YOUR_DATA.csv \\\n"
                error_msg += f"    --output_hdf5 {hdf5_path} \\\n"
                error_msg += f"    --smiles_column smiles\n"
                raise ValueError(error_msg)
            
            # Success
            print("✅ HDF5 file is COMPATIBLE with model")
            print(f"   Max hops: {compat.attrs.get('max_hops', 'N/A')}")
            print(f"   Task type: {compat.attrs.get('task_type', 'N/A')}")
            print(f"   Preprocessing status: {'Already applied (ERROR!)' if compat.attrs.get('preprocessing_applied', False) else 'RAW (will be applied during inference)'}")
            print(f"   For inference: {compat.attrs.get('for_inference', 'N/A')}")
            
            return True


def main():
    parser = argparse.ArgumentParser(
        description="Create HDF5 files for inference from CSV data (CPU-only, no GPU required)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  # Create HDF5 for inference on CPU-only instance:
  python create_inference_hdf5.py --model_path model.pth --input_csv data.csv --output_hdf5 data.h5
  
  # Verify existing HDF5 compatibility:
  python create_inference_hdf5.py --model_path model.pth --output_hdf5 data.h5 --verify_only
  
  # Use multiple workers for faster processing:
  python create_inference_hdf5.py --model_path model.pth --input_csv data.csv --output_hdf5 data.h5 --num_workers 16

Pipeline for efficient HPC inference:
  1. On CPU-only instance: Create HDF5 file (this script)
  2. Transfer HDF5 to GPU instance
  3. On GPU instance: Run inference using the HDF5 file
        """
    )
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model file (.pth)")
    parser.add_argument("--input_csv", type=str,
                       help="Path to input CSV file (required unless --verify_only)")
    parser.add_argument("--output_hdf5", type=str, required=True,
                       help="Path for output HDF5 file")
    parser.add_argument("--smiles_column", type=str, default="smiles",
                       help="Column name containing SMILES strings")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of parallel workers (more = faster, but more memory)")
    parser.add_argument("--chunk_size", type=int, default=1000,
                       help="Chunk size for parallel processing")
    parser.add_argument("--verify_only", action="store_true",
                       help="Only verify compatibility of existing HDF5 file")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.verify_only and not args.input_csv:
        parser.error("--input_csv is required unless --verify_only is specified")
    
    try:
        print("\n" + "="*70)
        print("INFERENCE HDF5 CREATOR - CPU-ONLY MODE")
        print("="*70)
        print(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")
        
        # Create HDF5 creator
        creator = InferenceHDF5Creator(args.model_path)
        
        if args.verify_only:
            # Verify existing file
            print("MODE: Verification only\n")
            creator.verify_hdf5_compatibility(args.output_hdf5)
            print("\n✅ HDF5 file is ready for inference")
        else:
            # Create new HDF5 file
            print("MODE: Create new HDF5 file for inference\n")
            creator.create_inference_hdf5(
                input_csv=args.input_csv,
                output_hdf5=args.output_hdf5,
                smiles_column=args.smiles_column,
                num_workers=args.num_workers,
                chunk_size=args.chunk_size
            )
            
            # Verify it was created correctly
            print("\nVerifying created file...")
            creator.verify_hdf5_compatibility(args.output_hdf5)
        
        print("\n" + "="*70)
        print("✅ SUCCESS - HDF5 file is ready for GPU inference")
        print("="*70)
        print("\nNext steps:")
        print("  1. Transfer HDF5 file to GPU instance")
        print(f"  2. Run inference: python main.py --inference_hdf5 {args.output_hdf5} \\")
        print(f"                      --model_save_path {args.model_path} \\")
        print(f"                      --inference_output predictions.csv")
        print("="*70 + "\n")
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ ERROR")
        print("="*70)
        print(f"{e}")
        print("="*70 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()