#!/usr/bin/env python
"""
Generate synthetic benchmark dataset for stereochemistry validation.

This script creates molecules where properties depend DIRECTLY on chirality,
allowing us to test whether the model can distinguish enantiomers.

Property formula:
- R enantiomer: property = base_value + CHIRALITY_OFFSET
- S enantiomer: property = base_value - CHIRALITY_OFFSET
- Achiral: property = base_value

Usage:
    python scripts/generate_chirality_benchmark.py --output data/chirality_benchmark.csv --n_molecules 1000
"""

import argparse
import random
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem


# Chirality offset - the property difference between R and S enantiomers
CHIRALITY_OFFSET = 1.0

# Base scaffolds for generating chiral molecules
CHIRAL_SCAFFOLDS = [
    # Amino acid-like (alpha carbon)
    ('N[C@@H]({R})C(=O)O', 'N[C@H]({R})C(=O)O', 'amino_acid'),
    # Secondary alcohol
    ('C[C@@H](O){R}', 'C[C@H](O){R}', 'secondary_alcohol'),
    # Ether with chiral carbon
    ('CO[C@@H](C){R}', 'CO[C@H](C){R}', 'chiral_ether'),
    # Amine with chiral carbon
    ('CN[C@@H](C){R}', 'CN[C@H](C){R}', 'chiral_amine'),
    # Sulfoxide (pyramidal heteroatom)
    ('C[S@@](=O){R}', 'C[S@](=O){R}', 'sulfoxide'),
]

# R-groups to substitute into scaffolds
R_GROUPS = [
    'C',           # methyl
    'CC',          # ethyl
    'CCC',         # propyl
    'C(C)C',       # isopropyl
    'CCCC',        # butyl
    'c1ccccc1',    # phenyl
    'Cc1ccccc1',   # benzyl
    'CC(C)C',      # isobutyl
    'C(=O)C',      # acetyl
    'C(=O)OC',     # methyl ester
    'CF',          # fluoromethyl
    'CCO',         # 2-hydroxyethyl
    'CCN',         # 2-aminoethyl
    'C1CCCCC1',    # cyclohexyl
    'C1CCCC1',     # cyclopentyl
]

# Achiral scaffolds for control molecules
ACHIRAL_SCAFFOLDS = [
    'CC(C)C',           # isobutane
    'c1ccccc1',         # benzene
    'CC(=O)O',          # acetic acid
    'CCCC',             # butane
    'C1CCCCC1',         # cyclohexane
    'CCO',              # ethanol
    'CCCO',             # propanol
    'CC(C)(C)C',        # neopentane
    'c1ccc(C)cc1',      # toluene
    'CC(=O)CC',         # butanone
]


def smiles_to_hash(smiles: str) -> float:
    """Convert SMILES to a deterministic hash value in [0, 10]."""
    # Remove stereochemistry for base property calculation
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 5.0  # Default

    # Get canonical SMILES without stereochemistry
    Chem.RemoveStereochemistry(mol)
    canonical = Chem.MolToSmiles(mol)

    # Hash to get deterministic base value
    h = hashlib.md5(canonical.encode()).hexdigest()
    return (int(h[:8], 16) % 10000) / 1000.0  # Value in [0, 10]


def compute_base_property(smiles: str) -> float:
    """
    Compute base property value from molecular features.

    Uses molecular weight and logP to create a property that
    varies with molecular structure but is independent of chirality.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 5.0

    # Base on molecular descriptors (chirality-independent)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)

    # Normalize to reasonable range
    base = (mw / 50.0) + logp

    # Add hash-based component for variety
    hash_component = smiles_to_hash(smiles)

    return base + hash_component


def get_chirality_sign(smiles: str) -> int:
    """
    Get chirality sign from SMILES.

    Returns:
        +1 for R configuration (@@)
        -1 for S configuration (@)
        0 for achiral
    """
    if '@@' in smiles:
        return 1  # R
    elif '@' in smiles:
        return -1  # S
    return 0  # achiral


def generate_chiral_pair(scaffold_r: str, scaffold_s: str, r_group: str) -> tuple[str, str] | None:
    """
    Generate R/S enantiomer pair from scaffold and R-group.

    Returns:
        Tuple of (r_smiles, s_smiles) or None if invalid
    """
    r_smiles = scaffold_r.replace('{R}', r_group)
    s_smiles = scaffold_s.replace('{R}', r_group)

    # Validate SMILES
    r_mol = Chem.MolFromSmiles(r_smiles)
    s_mol = Chem.MolFromSmiles(s_smiles)

    if r_mol is None or s_mol is None:
        return None

    # Canonicalize
    r_smiles = Chem.MolToSmiles(r_mol)
    s_smiles = Chem.MolToSmiles(s_mol)

    return r_smiles, s_smiles


def generate_dataset(n_molecules: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic chirality benchmark dataset.

    Args:
        n_molecules: Target number of molecules (actual may differ slightly)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: smiles, property, chirality, scaffold_type
    """
    random.seed(seed)
    np.random.seed(seed)

    data = []

    # Generate chiral pairs
    n_chiral_pairs = n_molecules // 3  # ~1/3 R, ~1/3 S, ~1/3 achiral

    for _ in range(n_chiral_pairs):
        # Select random scaffold and R-group
        scaffold_r, scaffold_s, scaffold_type = random.choice(CHIRAL_SCAFFOLDS)
        r_group = random.choice(R_GROUPS)

        result = generate_chiral_pair(scaffold_r, scaffold_s, r_group)
        if result is None:
            continue

        r_smiles, s_smiles = result
        base_value = compute_base_property(r_smiles)

        # Add noise to make it more realistic
        noise_r = np.random.normal(0, 0.1)
        noise_s = np.random.normal(0, 0.1)

        # R enantiomer: base + offset
        data.append({
            'smiles': r_smiles,
            'property': base_value + CHIRALITY_OFFSET + noise_r,
            'chirality': 'R',
            'scaffold_type': scaffold_type,
        })

        # S enantiomer: base - offset
        data.append({
            'smiles': s_smiles,
            'property': base_value - CHIRALITY_OFFSET + noise_s,
            'chirality': 'S',
            'scaffold_type': scaffold_type,
        })

    # Generate achiral molecules
    n_achiral = n_molecules - len(data)
    for _ in range(n_achiral):
        scaffold = random.choice(ACHIRAL_SCAFFOLDS)
        mol = Chem.MolFromSmiles(scaffold)
        if mol is None:
            continue

        smiles = Chem.MolToSmiles(mol)
        base_value = compute_base_property(smiles)
        noise = np.random.normal(0, 0.1)

        data.append({
            'smiles': smiles,
            'property': base_value + noise,
            'chirality': 'achiral',
            'scaffold_type': 'achiral',
        })

    df = pd.DataFrame(data)

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


def split_dataset(df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/val/test ensuring R/S pairs stay together.

    Args:
        df: Full dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation (test = 1 - train - val)
        seed: Random seed

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    np.random.seed(seed)

    # Separate chiral and achiral
    chiral_df = df[df['chirality'] != 'achiral'].copy()
    achiral_df = df[df['chirality'] == 'achiral'].copy()

    # For chiral molecules, group by canonical SMILES without stereochemistry
    def get_base_smiles(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol)

    chiral_df['base_smiles'] = chiral_df['smiles'].apply(get_base_smiles)

    # Get unique base SMILES (each represents an R/S pair)
    unique_bases = chiral_df['base_smiles'].unique()
    np.random.shuffle(unique_bases)

    n_bases = len(unique_bases)
    n_train = int(n_bases * train_ratio)
    n_val = int(n_bases * val_ratio)

    train_bases = set(unique_bases[:n_train])
    val_bases = set(unique_bases[n_train:n_train + n_val])
    test_bases = set(unique_bases[n_train + n_val:])

    # Split chiral molecules
    train_chiral = chiral_df[chiral_df['base_smiles'].isin(train_bases)].drop(columns=['base_smiles'])
    val_chiral = chiral_df[chiral_df['base_smiles'].isin(val_bases)].drop(columns=['base_smiles'])
    test_chiral = chiral_df[chiral_df['base_smiles'].isin(test_bases)].drop(columns=['base_smiles'])

    # Split achiral molecules randomly
    achiral_indices = achiral_df.index.values
    np.random.shuffle(achiral_indices)

    n_achiral = len(achiral_df)
    n_train_achiral = int(n_achiral * train_ratio)
    n_val_achiral = int(n_achiral * val_ratio)

    train_achiral = achiral_df.loc[achiral_indices[:n_train_achiral]]
    val_achiral = achiral_df.loc[achiral_indices[n_train_achiral:n_train_achiral + n_val_achiral]]
    test_achiral = achiral_df.loc[achiral_indices[n_train_achiral + n_val_achiral:]]

    # Combine and shuffle
    train_df = pd.concat([train_chiral, train_achiral]).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = pd.concat([val_chiral, val_achiral]).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat([test_chiral, test_achiral]).sample(frac=1, random_state=seed).reset_index(drop=True)

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description='Generate chirality benchmark dataset')
    parser.add_argument('--output', type=str, default='data/chirality_benchmark',
                        help='Output path prefix (without extension)')
    parser.add_argument('--n_molecules', type=int, default=3000,
                        help='Number of molecules to generate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n_molecules} molecules...")
    df = generate_dataset(args.n_molecules, seed=args.seed)

    print(f"Generated {len(df)} molecules:")
    print(f"  R enantiomers: {(df['chirality'] == 'R').sum()}")
    print(f"  S enantiomers: {(df['chirality'] == 'S').sum()}")
    print(f"  Achiral: {(df['chirality'] == 'achiral').sum()}")

    print(f"\nScaffold distribution:")
    print(df['scaffold_type'].value_counts())

    print(f"\nProperty statistics:")
    print(f"  R mean: {df[df['chirality'] == 'R']['property'].mean():.3f}")
    print(f"  S mean: {df[df['chirality'] == 'S']['property'].mean():.3f}")
    print(f"  Achiral mean: {df[df['chirality'] == 'achiral']['property'].mean():.3f}")
    print(f"  R-S difference: {df[df['chirality'] == 'R']['property'].mean() - df[df['chirality'] == 'S']['property'].mean():.3f}")

    # Split dataset
    print(f"\nSplitting dataset (train={args.train_ratio}, val={args.val_ratio})...")
    train_df, val_df, test_df = split_dataset(df, args.train_ratio, args.val_ratio, args.seed)

    print(f"  Train: {len(train_df)} molecules")
    print(f"  Val: {len(val_df)} molecules")
    print(f"  Test: {len(test_df)} molecules")

    # Save datasets
    full_path = f"{args.output}_full.csv"
    train_path = f"{args.output}_train.csv"
    val_path = f"{args.output}_val.csv"
    test_path = f"{args.output}_test.csv"

    df.to_csv(full_path, index=False)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSaved datasets:")
    print(f"  Full: {full_path}")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")
    print(f"  Test: {test_path}")

    # Verify R-S pairs are in same split
    print("\nVerifying R/S pair integrity...")
    for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        chiral = split_df[split_df['chirality'] != 'achiral']
        r_count = (chiral['chirality'] == 'R').sum()
        s_count = (chiral['chirality'] == 'S').sum()
        if r_count != s_count:
            print(f"  WARNING: {name} has unbalanced R/S: {r_count} R, {s_count} S")
        else:
            print(f"  {name}: {r_count} R/S pairs (balanced)")


if __name__ == '__main__':
    main()
