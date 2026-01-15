# datasets/io.py
"""
Data input/output functions for molecular datasets.
"""

import pandas as pd
from typing import Any
from sklearn.model_selection import train_test_split


def load_dataset_simple(
    file_path: str,
    smiles_column: str,
    target_column: str
) -> tuple[list[str], list[float]]:
    """
    Load a simple dataset from CSV with one target.
    
    Args:
        file_path: Path to CSV file
        smiles_column: Column name for SMILES strings
        target_column: Column name for target values
        
    Returns:
        Tuple of (smiles_list, target_values)
    """
    df = pd.read_csv(file_path)
    smiles_list = df[smiles_column].tolist()
    target_values = df[target_column].tolist()
    return smiles_list, target_values


def load_dataset_multitask(
    file_path: str,
    smiles_column: str,
    multi_target_columns: list[str]
) -> tuple[list[str], list[list[float]]]:
    """
    Load a multi-task dataset from CSV.
    
    Args:
        file_path: Path to CSV file
        smiles_column: Column name for SMILES strings
        multi_target_columns: List of column names for multiple targets
        
    Returns:
        Tuple of (smiles_list, target_values) where target_values is a list of lists
    """
    df = pd.read_csv(file_path)
    smiles_list = df[smiles_column].tolist()
    target_values = df[multi_target_columns].values.tolist()
    return smiles_list, target_values


def split_dataset(
    smiles_list: list[str],
    target_values: list[Any],
    train_split: float,
    val_split: float,
    test_split: float,
    task_type: str = 'regression'
) -> tuple[list[str], list[Any], list[str], list[Any], list[str], list[Any]]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        smiles_list: List of SMILES strings
        target_values: List of target values
        train_split: Fraction for training set
        val_split: Fraction for validation set
        test_split: Fraction for test set
        task_type: Type of task ('regression' or 'multitask')
        
    Returns:
        Tuple of (smiles_train, target_train, smiles_val, target_val, smiles_test, target_test)
    """
    train_val_split = train_split + val_split
    smiles_train_val, smiles_test, target_train_val, target_test = train_test_split(
        smiles_list, target_values, test_size=test_split, random_state=42
    )
    smiles_train, smiles_val, target_train, target_val = train_test_split(
        smiles_train_val, target_train_val,
        test_size=val_split / train_val_split, random_state=42
    )
    return smiles_train, target_train, smiles_val, target_val, smiles_test, target_test