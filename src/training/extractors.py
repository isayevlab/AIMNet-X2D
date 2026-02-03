"""
Feature extraction functionality for trained GNN models.

This module contains functions for extracting embeddings and partial charges.
"""

import torch
import numpy as np
import h5py
import tqdm
from datetime import datetime

from utils.logging import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def extract_partial_charges(model, data_loader, device) -> list[tuple[str, list[float]]]:
    """
    Iterates over the data_loader, does a forward pass,
    and collects partial charges if available.

    Args:
        model: Model to extract partial charges from
        data_loader: DataLoader with input data
        device: Device to run inference on

    Returns:
        List of tuples (smiles, [q1, q2, ...]) for each molecule
    """
    model.eval()
    results = []

    for batch_idx, batch in enumerate(data_loader):
        if batch is None:
            continue
            
        # Prepare batch data
        batch_multi_hop_edges = batch.multi_hop_edge_indices.to(device)
        batch_indices = batch.batch_indices.to(device)
        batch_atom_features = {k: v.to(device) for k, v in batch.atom_features_map.items()}
        total_charges = batch.total_charges.to(device)
        tetrahedral_indices = batch.final_tetrahedral_chiral_tensor.to(device)
        cis_indices = batch.final_cis_tensor.to(device)
        trans_indices = batch.final_trans_tensor.to(device)
        chiral_signs = batch.chiral_signs.to(device) if batch.chiral_signs.numel() > 0 else None
        chiral_is_virtual_lp = batch.chiral_is_virtual_lp.to(device) if batch.chiral_is_virtual_lp.numel() > 0 else None
        allene_centers = batch.allene_centers.to(device) if batch.allene_centers.numel() > 0 else None
        allene_subs = batch.allene_subs.to(device) if batch.allene_subs.numel() > 0 else None

        # Forward pass with partial charges
        _, _, partial_charges = model(
            batch_atom_features,
            batch_multi_hop_edges,
            batch_indices,
            total_charges,
            tetrahedral_indices,
            cis_indices,
            trans_indices,
            chiral_signs,
            chiral_is_virtual_lp,
            allene_centers,
            allene_subs
        )

        # Skip if partial_charges is None
        if partial_charges is None:
            for smi in batch.smiles_list:
                results.append((smi, []))
            continue

        # Group partial charges by molecule
        unique_mols = batch_indices.unique().tolist()
        for mol_id in unique_mols:
            smi = batch.smiles_list[mol_id]
            mask = (batch_indices == mol_id)
            q_vals = partial_charges[mask].detach().cpu().numpy().tolist()
            results.append((smi, q_vals))

    return results


def extract_all_embeddings(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    output_path: str,
    embedding_type: str = 'pooled',
    include_atom_embeddings: bool = False
) -> dict:
    """
    Extract molecular embeddings from all data loaders (train, validation, test)
    and save them to a structured HDF5 file.

    Args:
        model: The trained GNN model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        device: Device to run inference on
        output_path: Path to save the embeddings
        embedding_type: Type of embedding to extract ('pooled' by default)
        include_atom_embeddings: Whether to also extract atom-level embeddings
    """
    model.eval()
    
    # Dictionary to store all embeddings and metadata
    dataset_embeddings = {
        'train': {'mol_embeddings': [], 'smiles': [], 'atom_embeddings': {}, 'atom_counts': {}},
        'val': {'mol_embeddings': [], 'smiles': [], 'atom_embeddings': {}, 'atom_counts': {}},
        'test': {'mol_embeddings': [], 'smiles': [], 'atom_embeddings': {}, 'atom_counts': {}}
    }
    
    # Create hooks for embeddings
    mol_embeddings_hook = []
    atom_embeddings_hook = []
    
    def mol_hook_fn(module, input, output):
        mol_embeddings_hook.append(output[0].detach().cpu().numpy())
    
    def atom_hook_fn(module, input, output):
        atom_embeddings_hook.append(output.detach().cpu().numpy())
    
    # Register the hooks at the appropriate layers
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        mol_hook = model.module.pooling.register_forward_hook(mol_hook_fn)
        if include_atom_embeddings:
            atom_hook = model.module.concat_self_other.register_forward_hook(atom_hook_fn)
    else:
        mol_hook = model.pooling.register_forward_hook(mol_hook_fn)
        if include_atom_embeddings:
            atom_hook = model.concat_self_other.register_forward_hook(atom_hook_fn)
    
    # Process each dataset
    for dataset_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        logger.info(f"Extracting embeddings from {dataset_name} set...")
        
        # Process each batch
        for batch_idx, batch in enumerate(tqdm.tqdm(loader, desc=f"Processing {dataset_name}")):
            if batch is None:
                continue
            
            # Clear the hook lists before each batch
            mol_embeddings_hook.clear()
            atom_embeddings_hook.clear()
            
            # Prepare batch data for forward pass
            batch_multi_hop_edges = batch.multi_hop_edge_indices.to(device)
            batch_indices = batch.batch_indices.to(device)
            batch_atom_features = {k: v.to(device) for k, v in batch.atom_features_map.items()}
            total_charges = batch.total_charges.to(device)
            tetrahedral_indices = batch.final_tetrahedral_chiral_tensor.to(device)
            cis_indices = batch.final_cis_tensor.to(device)
            trans_indices = batch.final_trans_tensor.to(device)
            chiral_signs = batch.chiral_signs.to(device) if batch.chiral_signs.numel() > 0 else None
            chiral_is_virtual_lp = batch.chiral_is_virtual_lp.to(device) if batch.chiral_is_virtual_lp.numel() > 0 else None
            allene_centers = batch.allene_centers.to(device) if batch.allene_centers.numel() > 0 else None
            allene_subs = batch.allene_subs.to(device) if batch.allene_subs.numel() > 0 else None

            # Forward pass to trigger the hooks
            _ = model(
                batch_atom_features,
                batch_multi_hop_edges,
                batch_indices,
                total_charges,
                tetrahedral_indices,
                cis_indices,
                trans_indices,
                chiral_signs,
                chiral_is_virtual_lp,
                allene_centers,
                allene_subs
            )
            
            # Save molecule embeddings
            batch_mol_embeddings = mol_embeddings_hook[0]
            dataset_embeddings[dataset_name]['mol_embeddings'].append(batch_mol_embeddings)
            
            # Save SMILES
            dataset_embeddings[dataset_name]['smiles'].extend(batch.smiles_list)
            
            # Process atom embeddings if requested
            if include_atom_embeddings:
                batch_atom_embeddings = atom_embeddings_hook[0]
                batch_indices_np = batch_indices.cpu().numpy()
                
                # Group atoms by molecule
                unique_mol_indices = np.unique(batch_indices_np)
                
                # For each molecule in this batch
                for mol_idx in unique_mol_indices:
                    # Get SMILES for this molecule
                    mol_smiles = batch.smiles_list[mol_idx]
                    
                    # Find atoms belonging to this molecule
                    mask = (batch_indices_np == mol_idx)
                    mol_atoms = batch_atom_embeddings[mask]
                    
                    # Store atom embeddings by SMILES
                    current_idx = len(dataset_embeddings[dataset_name]['mol_embeddings']) * loader.batch_size + mol_idx
                    dataset_embeddings[dataset_name]['atom_embeddings'][current_idx] = mol_atoms
                    dataset_embeddings[dataset_name]['atom_counts'][mol_smiles] = len(mol_atoms)
    
    # Remove the hooks
    mol_hook.remove()
    if include_atom_embeddings:
        atom_hook.remove()
    
    # Combine and finalize embeddings
    results = {}
    for dataset_name, data in dataset_embeddings.items():
        if data['mol_embeddings']:
            # Stack molecule embeddings
            results[f"{dataset_name}_mol_embeddings"] = np.vstack(data['mol_embeddings'])
            results[f"{dataset_name}_smiles"] = data['smiles']
            
            if include_atom_embeddings:
                results[f"{dataset_name}_atom_embeddings"] = data['atom_embeddings']
                results[f"{dataset_name}_atom_counts"] = data['atom_counts']
    
    # Total molecule counts
    train_count = len(results.get('train_smiles', []))
    val_count = len(results.get('val_smiles', []))
    test_count = len(results.get('test_smiles', []))
    total_count = train_count + val_count + test_count
    
    logger.info(f"Extracted embeddings for {total_count} molecules: {train_count} train, {val_count} validation, {test_count} test")
    
    # Save the embeddings to HDF5
    save_embeddings_to_hdf5(results, output_path, include_atom_embeddings)
    
    return results


def save_embeddings_to_hdf5(results: dict, output_path: str, include_atom_embeddings: bool = False) -> None:
    """
    Save extracted embeddings to an HDF5 file with a structured format.

    Args:
        results: Dictionary containing embeddings and metadata
        output_path: Path to save the HDF5 file
        include_atom_embeddings: Whether atom embeddings are included
    """
    with h5py.File(output_path, 'w') as f:
        # Create dataset groups
        train_group = f.create_group('train')
        val_group = f.create_group('validation')
        test_group = f.create_group('test')
        
        # Create metadata group
        metadata = f.create_group('metadata')
        
        # Save dataset sizes
        train_size = len(results.get('train_smiles', []))
        val_size = len(results.get('val_smiles', []))
        test_size = len(results.get('test_smiles', []))
        
        metadata.attrs['train_size'] = train_size
        metadata.attrs['validation_size'] = val_size
        metadata.attrs['test_size'] = test_size
        metadata.attrs['total_size'] = train_size + val_size + test_size
        metadata.attrs['include_atom_embeddings'] = include_atom_embeddings
        
        # Save molecule embeddings and SMILES
        for dataset_name, group in [('train', train_group), ('val', val_group), ('test', test_group)]:
            # Check if this dataset has data
            mol_embeddings_key = f"{dataset_name}_mol_embeddings"
            smiles_key = f"{dataset_name}_smiles"
            
            if mol_embeddings_key in results and smiles_key in results:
                mol_embeddings = results[mol_embeddings_key]
                smiles = results[smiles_key]
                
                # Save molecular embeddings
                group.create_dataset('mol_embeddings', data=mol_embeddings)
                
                # Save SMILES strings
                dt = h5py.special_dtype(vlen=str)
                smiles_dataset = group.create_dataset('smiles', (len(smiles),), dtype=dt)
                for i, smi in enumerate(smiles):
                    smiles_dataset[i] = smi
                
                # Save atom embeddings if available
                if include_atom_embeddings:
                    atom_key = f"{dataset_name}_atom_embeddings"
                    if atom_key in results:
                        atom_group = group.create_group('atom_embeddings')
                        atom_counts = group.create_dataset('atom_counts', (len(smiles),), dtype=np.int32)
                        
                        atom_embeddings = results[atom_key]
                        for idx, atoms in atom_embeddings.items():
                            atom_dataset = atom_group.create_dataset(f'mol_{idx}', data=atoms)
                        
                        # Save atom counts
                        counts_dict = results.get(f"{dataset_name}_atom_counts", {})
                        for i, smi in enumerate(smiles):
                            atom_counts[i] = counts_dict.get(smi, 0)
    
    logger.info(f"Successfully saved embeddings to: {output_path}")


def extract_embeddings_main(args, model, train_loader, val_loader, test_loader, device) -> None:
    """
    Extract molecular embeddings from the trained model for all datasets.

    This function:
    1. Runs molecules through the trained GNN model
    2. Extracts embeddings from the pooling layer (molecule-level representations)
    3. Optionally extracts atom-level embeddings
    4. Saves everything to an HDF5 file for downstream analysis

    Args:
        args: Command line arguments
        model: The trained GNN model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        device: Device to run extraction on
    """
    logger.info("=" * 80)
    logger.info("EXTRACTING MOLECULAR EMBEDDINGS")
    logger.info("=" * 80)
    logger.info(f"Output file: {args.embeddings_output_path}")
    
    # Ensure model is in eval mode
    model.eval()
    
    # Create hooks for embeddings (pooling output = molecular embeddings)
    mol_embeddings = []
    atom_embeddings = []
    batch_smiles = []
    
    # Define hook functions
    def mol_hook_fn(module, input, output):
        mol_embeddings.append(output[0].detach().cpu().numpy())
    
    def atom_hook_fn(module, input, output):
        atom_embeddings.append(output.detach().cpu().numpy())
    
    # Register hooks at appropriate layers
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        mol_hook = model.module.pooling.register_forward_hook(mol_hook_fn)
        if args.include_atom_embeddings:
            atom_hook = model.module.concat_self_other.register_forward_hook(atom_hook_fn)
    else:
        mol_hook = model.pooling.register_forward_hook(mol_hook_fn)
        if args.include_atom_embeddings:
            atom_hook = model.concat_self_other.register_forward_hook(atom_hook_fn)
    
    # Process datasets
    dataset_embeddings = {}
    
    # Function to process a dataset
    def process_dataset(name, loader):
        logger.info(f"Processing {name} dataset...")
        
        # Clear collections for this dataset
        mol_embeddings.clear()
        atom_embeddings.clear()
        batch_smiles.clear()
        
        # Process batches
        total_molecules = 0
        with torch.no_grad():
            for batch in tqdm.tqdm(loader, desc=f"Extracting {name} embeddings"):
                if batch is None:
                    continue
                
                # Add SMILES to collection
                batch_smiles.extend(batch.smiles_list)
                total_molecules += len(batch.smiles_list)
                
                # Prepare batch for model
                batch_multi_hop_edges = batch.multi_hop_edge_indices.to(device)
                batch_indices = batch.batch_indices.to(device)
                batch_atom_features = {k: v.to(device) for k, v in batch.atom_features_map.items()}
                total_charges = batch.total_charges.to(device)
                tetrahedral_indices = batch.final_tetrahedral_chiral_tensor.to(device)
                cis_indices = batch.final_cis_tensor.to(device)
                trans_indices = batch.final_trans_tensor.to(device)
                chiral_signs = batch.chiral_signs.to(device) if batch.chiral_signs.numel() > 0 else None
                chiral_is_virtual_lp = batch.chiral_is_virtual_lp.to(device) if batch.chiral_is_virtual_lp.numel() > 0 else None
                allene_centers = batch.allene_centers.to(device) if batch.allene_centers.numel() > 0 else None
                allene_subs = batch.allene_subs.to(device) if batch.allene_subs.numel() > 0 else None

                # Forward pass triggers hooks
                _ = model(
                    batch_atom_features,
                    batch_multi_hop_edges,
                    batch_indices,
                    total_charges,
                    tetrahedral_indices,
                    cis_indices,
                    trans_indices,
                    chiral_signs,
                    chiral_is_virtual_lp,
                    allene_centers,
                    allene_subs
                )
        
        # Process molecule embeddings
        mols_emb = np.vstack(mol_embeddings) if mol_embeddings else np.array([])
        
        # Return results
        result = {
            'embeddings': mols_emb,
            'smiles': batch_smiles,
            'count': total_molecules
        }
        
        # Process atom embeddings if requested
        if args.include_atom_embeddings and atom_embeddings:
            # Create a mapping from SMILES to atom embeddings
            atom_emb_map = {}
            atom_count_map = {}
            
            # Group atom embeddings by molecule
            for batch_idx, atom_emb_batch in enumerate(atom_embeddings):
                batch_data = loader.dataset.data_list[batch_idx] if hasattr(loader.dataset, 'data_list') else None
                
                if batch_data is not None:
                    # For in-memory dataset
                    mol_smiles = batch_data.smiles
                    atom_emb_map[mol_smiles] = atom_emb_batch
                    atom_count_map[mol_smiles] = len(atom_emb_batch)
            
            result['atom_embeddings'] = atom_emb_map
            result['atom_counts'] = atom_count_map
        
        return result
    
    # Process each dataset
    if train_loader:
        dataset_embeddings['train'] = process_dataset('train', train_loader)
        logger.info(f"Extracted embeddings for {dataset_embeddings['train']['count']:,} training molecules")

    if val_loader:
        dataset_embeddings['validation'] = process_dataset('validation', val_loader)
        logger.info(f"Extracted embeddings for {dataset_embeddings['validation']['count']:,} validation molecules")

    if test_loader:
        dataset_embeddings['test'] = process_dataset('test', test_loader)
        logger.info(f"Extracted embeddings for {dataset_embeddings['test']['count']:,} test molecules")
    
    # Remove hooks
    mol_hook.remove()
    if args.include_atom_embeddings:
        atom_hook.remove()
    
    # Save embeddings to HDF5
    with h5py.File(args.embeddings_output_path, 'w') as f:
        # Create metadata
        metadata = f.create_group('metadata')
        metadata.attrs['creation_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata.attrs['include_atom_embeddings'] = args.include_atom_embeddings
        
        # Add model information
        model_info = metadata.create_group('model')
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_info.attrs['hidden_dim'] = model.module.hidden_dim if hasattr(model.module, 'hidden_dim') else 0
            model_info.attrs['num_shells'] = model.module.num_shells if hasattr(model.module, 'num_shells') else 0
            emb_dim = model_info.attrs['embedding_dim'] = model.module.embedding_dim if hasattr(model.module, 'embedding_dim') else 0
        else:
            model_info.attrs['hidden_dim'] = model.hidden_dim if hasattr(model, 'hidden_dim') else 0
            model_info.attrs['num_shells'] = model.num_shells if hasattr(model, 'num_shells') else 0
            model_info.attrs['embedding_dim'] = model.embedding_dim if hasattr(model, 'embedding_dim') else 0
        
        # Store embeddings for each dataset
        for dataset_name, data in dataset_embeddings.items():
            # Create dataset group
            dataset_group = f.create_group(dataset_name)
            
            # Store molecular embeddings
            embedding_shape = data['embeddings'].shape
            dataset_group.create_dataset('embeddings', data=data['embeddings'])
            
            # Store SMILES
            dt = h5py.special_dtype(vlen=str)
            smiles_dataset = dataset_group.create_dataset('smiles', (len(data['smiles']),), dtype=dt)
            for i, smi in enumerate(data['smiles']):
                smiles_dataset[i] = smi
            
            # Store atom embeddings if available
            if args.include_atom_embeddings and 'atom_embeddings' in data and 'atom_counts' in data:
                atoms_group = dataset_group.create_group('atom_embeddings')
                atom_counts = dataset_group.create_dataset('atom_counts', (len(data['smiles']),), dtype=np.int32)
                
                # Store atom embeddings for each molecule
                for i, (smi, count) in enumerate(data['atom_counts'].items()):
                    if smi in data['atom_embeddings']:
                        mol_atom_emb = data['atom_embeddings'][smi]
                        atoms_group.create_dataset(f'mol_{i}', data=mol_atom_emb)
                        atom_counts[i] = count
    
    total_count = sum(data['count'] for data in dataset_embeddings.values())
    logger.info(f"Successfully saved embeddings for {total_count:,} molecules to: {args.embeddings_output_path}")
    logger.info("=" * 80)