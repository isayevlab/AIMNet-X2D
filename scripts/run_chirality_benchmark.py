#!/usr/bin/env python
"""
Run chirality benchmark: train models with and without stereochemistry.

This script trains two models on the chirality benchmark dataset:
1. Model with stereochemistry enabled
2. Model without stereochemistry (baseline)

Then compares their ability to distinguish R/S enantiomers.

Usage:
    python scripts/run_chirality_benchmark.py --epochs 50 --output_dir results/chirality_benchmark
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


def run_training(
    train_path: str,
    val_path: str,
    test_path: str,
    model_path: str,
    use_stereochemistry: bool,
    epochs: int = 50,
    batch_size: int = 64,
    hidden_dim: int = 128,
    num_shells: int = 3,
) -> dict:
    """
    Run model training with specified configuration.

    Returns:
        Dictionary with training results including test MAE
    """
    cmd = [
        sys.executable, 'main.py',
        '--train_data', train_path,
        '--val_data', val_path,
        '--test_data', test_path,
        '--target_column', 'property',
        '--task_type', 'regression',
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--hidden_dim', str(hidden_dim),
        '--num_shells', str(num_shells),
        '--model_save_path', model_path,
        '--loss_function', 'l1',
        '--lr', '0.001',
        '--early_stopping',
        '--patience', '10',
    ]

    if use_stereochemistry:
        cmd.append('--use_stereochemistry')

    print(f"\n{'='*60}")
    print(f"Training {'WITH' if use_stereochemistry else 'WITHOUT'} stereochemistry")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        # Parse output for metrics
        output = result.stdout + result.stderr
        print(output)

        # Extract test MAE from output
        test_mae = None
        for line in output.split('\n'):
            if 'Test MAE' in line or 'test_mae' in line.lower():
                try:
                    # Try to extract number
                    parts = line.split(':')
                    if len(parts) >= 2:
                        test_mae = float(parts[-1].strip().split()[0])
                except (ValueError, IndexError):
                    pass

        return {
            'success': result.returncode == 0,
            'test_mae': test_mae,
            'output': output,
            'use_stereochemistry': use_stereochemistry,
        }

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'test_mae': None,
            'output': 'Training timed out',
            'use_stereochemistry': use_stereochemistry,
        }
    except Exception as e:
        return {
            'success': False,
            'test_mae': None,
            'output': str(e),
            'use_stereochemistry': use_stereochemistry,
        }


def evaluate_enantiomer_discrimination(
    test_path: str,
    predictions_path: str,
) -> dict:
    """
    Evaluate how well the model distinguishes R/S enantiomers.

    Returns:
        Dictionary with discrimination metrics
    """
    # Load test data and predictions
    test_df = pd.read_csv(test_path)
    pred_df = pd.read_csv(predictions_path)

    # Merge on SMILES
    if 'smiles' in pred_df.columns:
        merged = test_df.merge(pred_df, on='smiles', suffixes=('_true', '_pred'))
    else:
        # Assume same order
        merged = test_df.copy()
        merged['prediction'] = pred_df['prediction'].values if 'prediction' in pred_df.columns else pred_df.iloc[:, 1].values

    # Separate R, S, and achiral
    r_data = merged[merged['chirality'] == 'R']
    s_data = merged[merged['chirality'] == 'S']
    achiral_data = merged[merged['chirality'] == 'achiral']

    # Calculate mean prediction difference between R and S
    r_mean_pred = r_data['prediction'].mean() if 'prediction' in r_data.columns else r_data['property_pred'].mean()
    s_mean_pred = s_data['prediction'].mean() if 'prediction' in s_data.columns else s_data['property_pred'].mean()

    pred_diff = r_mean_pred - s_mean_pred

    # Expected difference is ~2.0 (CHIRALITY_OFFSET * 2)
    expected_diff = 2.0

    # Discrimination score: how much of the expected difference is captured
    discrimination_score = pred_diff / expected_diff if expected_diff != 0 else 0

    return {
        'r_mean_pred': r_mean_pred,
        's_mean_pred': s_mean_pred,
        'pred_diff': pred_diff,
        'expected_diff': expected_diff,
        'discrimination_score': discrimination_score,
        'n_r': len(r_data),
        'n_s': len(s_data),
        'n_achiral': len(achiral_data),
    }


def main():
    parser = argparse.ArgumentParser(description='Run chirality benchmark')
    parser.add_argument('--train_data', type=str, default='data/chirality_benchmark_train.csv',
                        help='Training data path')
    parser.add_argument('--val_data', type=str, default='data/chirality_benchmark_val.csv',
                        help='Validation data path')
    parser.add_argument('--test_data', type=str, default='data/chirality_benchmark_test.csv',
                        help='Test data path')
    parser.add_argument('--output_dir', type=str, default='results/chirality_benchmark',
                        help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--num_shells', type=int, default=3,
                        help='Number of message passing shells')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    results = {
        'timestamp': timestamp,
        'config': vars(args),
        'models': {},
    }

    # Train model WITH stereochemistry
    stereo_model_path = str(output_dir / f'model_stereo_{timestamp}.pth')
    stereo_result = run_training(
        args.train_data,
        args.val_data,
        args.test_data,
        stereo_model_path,
        use_stereochemistry=True,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_shells=args.num_shells,
    )
    results['models']['with_stereo'] = stereo_result

    # Train model WITHOUT stereochemistry
    no_stereo_model_path = str(output_dir / f'model_no_stereo_{timestamp}.pth')
    no_stereo_result = run_training(
        args.train_data,
        args.val_data,
        args.test_data,
        no_stereo_model_path,
        use_stereochemistry=False,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_shells=args.num_shells,
    )
    results['models']['without_stereo'] = no_stereo_result

    # Print comparison
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")

    print(f"\nModel WITH stereochemistry:")
    print(f"  Test MAE: {stereo_result.get('test_mae', 'N/A')}")

    print(f"\nModel WITHOUT stereochemistry:")
    print(f"  Test MAE: {no_stereo_result.get('test_mae', 'N/A')}")

    if stereo_result.get('test_mae') and no_stereo_result.get('test_mae'):
        improvement = (no_stereo_result['test_mae'] - stereo_result['test_mae']) / no_stereo_result['test_mae'] * 100
        print(f"\nImprovement: {improvement:.1f}%")
        results['improvement_percent'] = improvement

    # Save results
    results_path = output_dir / f'results_{timestamp}.json'
    with open(results_path, 'w') as f:
        # Remove non-serializable output
        save_results = {k: v for k, v in results.items()}
        for model_key in save_results.get('models', {}):
            if 'output' in save_results['models'][model_key]:
                save_results['models'][model_key]['output'] = save_results['models'][model_key]['output'][:1000]  # Truncate
        json.dump(save_results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Test MAE':<15} {'Success':<10}")
    print(f"{'-'*50}")
    print(f"{'With stereochemistry':<25} {stereo_result.get('test_mae', 'N/A'):<15} {stereo_result.get('success', False)}")
    print(f"{'Without stereochemistry':<25} {no_stereo_result.get('test_mae', 'N/A'):<15} {no_stereo_result.get('success', False)}")

    # Expected results explanation
    print(f"\n{'='*60}")
    print("EXPECTED RESULTS")
    print(f"{'='*60}")
    print("""
For a properly working stereochemistry implementation:
- Model WITH stereo should have LOW MAE (< 0.5) - can distinguish R/S
- Model WITHOUT stereo should have HIGH MAE (> 0.8) - cannot distinguish R/S
- The R-S property difference is 2.0 in this dataset
- A model that cannot distinguish R/S will predict the mean for both,
  resulting in ~1.0 MAE on chiral molecules
""")


if __name__ == '__main__':
    main()
