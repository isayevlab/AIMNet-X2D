"""
Embedding extraction functionality for inference.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import h5py
import pickle
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path

from utils.distributed import safe_get_rank, is_main_process
from datasets.constants import DEFAULT_PROGRESS_LOG_INTERVAL, DEFAULT_BATCH_FLUSH_THRESHOLD


class EmbeddingExtractor:
    """Extracts embeddings from trained models during inference."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.mol_embeddings = []
        self.atom_embeddings = []
        self.batch_indices = []
        self.smiles_list = []
        self.hooks = []
    
    def setup_hooks(self, include_atom_embeddings: bool = False):
        """Setup forward hooks for embedding extraction."""
        self._clear_hooks()
        
        # Molecular embedding hook (pooling layer output)
        def mol_hook_fn(module, input, output):
            # output[0] is the pooled molecular embedding
            self.mol_embeddings.append(output[0].detach().cpu().numpy())
        
        # Register molecular embedding hook
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            mol_hook = self.model.module.pooling.register_forward_hook(mol_hook_fn)
        else:
            mol_hook = self.model.pooling.register_forward_hook(mol_hook_fn)
        
        self.hooks.append(mol_hook)
        
        # Atom embedding hook (atom features before pooling)
        if include_atom_embeddings:
            def atom_hook_fn(module, input, output):
                # Get the atom features and corresponding batch indices
                atom_features = output.detach().cpu().numpy()
                self.atom_embeddings.append(atom_features)
            
            # Hook into the layer right before pooling
            # This should be the output of the last message passing layer
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                # Try to find the right layer - the combined features before pooling
                if hasattr(self.model.module, 'concat_self_other'):
                    atom_hook = self.model.module.concat_self_other.register_forward_hook(atom_hook_fn)
                else:
                    # Fallback: hook into the last message passing layer
                    if self.model.module.message_passing_layers:
                        last_mp_layer = self.model.module.message_passing_layers[-1]
                        atom_hook = last_mp_layer.register_forward_hook(atom_hook_fn)
                    else:
                        print("[Embeddings] Warning: Could not find suitable layer for atom embeddings")
                        return
            else:
                if hasattr(self.model, 'concat_self_other'):
                    atom_hook = self.model.concat_self_other.register_forward_hook(atom_hook_fn)
                else:
                    # Fallback: hook into the last message passing layer
                    if self.model.message_passing_layers:
                        last_mp_layer = self.model.message_passing_layers[-1]
                        atom_hook = last_mp_layer.register_forward_hook(atom_hook_fn)
                    else:
                        print("[Embeddings] Warning: Could not find suitable layer for atom embeddings")
                        return
            
            self.hooks.append(atom_hook)
    
    def extract_batch_embeddings(self, batch) -> Dict[str, np.ndarray]:
        """
        Extract embeddings for a single batch.
        
        Args:
            batch: Input batch
            
        Returns:
            Dictionary containing embeddings
        """
        # Clear previous batch data
        self.mol_embeddings.clear()
        self.atom_embeddings.clear()
        
        # Store SMILES and batch indices for this batch
        batch_smiles = batch.smiles_list
        batch_indices_tensor = batch.batch_indices
        
        # Prepare batch for model
        batch_multi_hop_edges = batch.multi_hop_edge_indices.to(self.device)
        batch_indices = batch_indices_tensor.to(self.device)
        batch_atom_features = {k: v.to(self.device) for k, v in batch.atom_features_map.items()}
        total_charges = batch.total_charges.to(self.device)
        tetrahedral_indices = batch.final_tetrahedral_chiral_tensor.to(self.device)
        cis_indices = batch.final_cis_tensor.to(self.device)
        trans_indices = batch.final_trans_tensor.to(self.device)
        
        # Forward pass (triggers hooks)
        with torch.no_grad():
            _ = self.model(
                batch_atom_features,
                batch_multi_hop_edges,
                batch_indices,
                total_charges,
                tetrahedral_indices,
                cis_indices,
                trans_indices
            )
        
        # Process embeddings
        result = {
            'smiles': batch_smiles,
            'mol_embeddings': self.mol_embeddings[0] if self.mol_embeddings else None
        }
        
        # Process atom embeddings if available
        if self.atom_embeddings:
            atom_emb = self.atom_embeddings[0]
            batch_indices_np = batch_indices_tensor.cpu().numpy()
            
            # Group atom embeddings by molecule
            atom_emb_by_mol = {}
            atom_counts = {}
            
            for mol_idx in range(len(batch_smiles)):
                mask = (batch_indices_np == mol_idx)
                mol_atoms = atom_emb[mask]
                if len(mol_atoms) > 0:  # Only store if atoms found
                    atom_emb_by_mol[mol_idx] = mol_atoms
                    atom_counts[mol_idx] = len(mol_atoms)
            
            result['atom_embeddings'] = atom_emb_by_mol
            result['atom_counts'] = atom_counts
        
        return result
    
    def _clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def __del__(self):
        """Cleanup hooks when object is destroyed."""
        self._clear_hooks()


class StreamingEmbeddingWriter:
    """Writes embeddings to HDF5 file in streaming fashion with DDP support."""
    
    def __init__(self, 
                 output_path: str, 
                 expected_total_molecules: int,
                 embedding_dim: int,
                 include_atom_embeddings: bool = False,
                 rank: int = 0,
                 world_size: int = 1):
        self.output_path = Path(output_path)
        self.expected_total_molecules = expected_total_molecules
        self.embedding_dim = embedding_dim
        self.include_atom_embeddings = include_atom_embeddings
        self.rank = rank
        self.world_size = world_size
        
        # For DDP, create rank-specific files
        if world_size > 1:
            base_path = self.output_path.with_suffix('')
            self.rank_output_path = Path(f"{base_path}_rank{rank}.h5")
        else:
            self.rank_output_path = self.output_path
        
        # Create output directory
        self.rank_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize file and datasets
        self.file = None
        self.mol_embeddings_dataset = None
        self.smiles_dataset = None
        self.atom_embeddings_group = None
        self.atom_counts_dataset = None
        self.current_index = 0
        
        # Store all data for DDP combination
        self.all_mol_embeddings = []
        self.all_smiles = []
        self.all_atom_embeddings = {}
        self.all_atom_counts = []
        
        self._initialize_file()
    
    def _initialize_file(self):
        """Initialize HDF5 file and datasets."""
        self.file = h5py.File(self.rank_output_path, 'w')
        
        # Create inference group
        inference_group = self.file.create_group('inference')
        
        # Create metadata
        metadata = self.file.create_group('metadata')
        metadata.attrs['creation_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata.attrs['expected_molecules'] = self.expected_total_molecules
        metadata.attrs['embedding_dim'] = self.embedding_dim
        metadata.attrs['include_atom_embeddings'] = self.include_atom_embeddings
        metadata.attrs['rank'] = self.rank
        metadata.attrs['world_size'] = self.world_size
    
    def write_batch(self, embeddings_data: Dict[str, np.ndarray]):
        """
        Write a batch of embeddings to the file.
        
        Args:
            embeddings_data: Dictionary containing batch embeddings
        """
        batch_size = len(embeddings_data['smiles'])
        
        # Store molecular embeddings
        if embeddings_data['mol_embeddings'] is not None:
            self.all_mol_embeddings.append(embeddings_data['mol_embeddings'])
        
        # Store SMILES
        self.all_smiles.extend(embeddings_data['smiles'])
        
        # Store atom embeddings if available
        if self.include_atom_embeddings and 'atom_embeddings' in embeddings_data:
            atom_emb_dict = embeddings_data['atom_embeddings']
            atom_counts_dict = embeddings_data.get('atom_counts', {})
            
            # Track atom counts for each molecule
            batch_atom_counts = []
            
            for mol_idx in range(batch_size):
                global_mol_idx = self.current_index + mol_idx
                
                if mol_idx in atom_emb_dict:
                    atom_emb = atom_emb_dict[mol_idx]
                    self.all_atom_embeddings[global_mol_idx] = atom_emb
                    batch_atom_counts.append(len(atom_emb))
                else:
                    # No atoms found for this molecule (shouldn't happen normally)
                    batch_atom_counts.append(0)
            
            self.all_atom_counts.extend(batch_atom_counts)
        elif self.include_atom_embeddings:
            # Fill with zeros if atom embeddings expected but not found
            self.all_atom_counts.extend([0] * batch_size)
        
        self.current_index += batch_size
        
        # Show progress for embeddings if this is the main rank
        if self.rank == 0 and self.current_index % DEFAULT_PROGRESS_LOG_INTERVAL == 0:
            if self.include_atom_embeddings:
                print(f"[Embeddings] Processed {self.current_index} molecules (mol + atom embeddings)")
            else:
                print(f"[Embeddings] Processed {self.current_index} molecules (mol embeddings)")
        
        # Periodic flush to avoid memory issues
        if len(self.all_mol_embeddings) > DEFAULT_BATCH_FLUSH_THRESHOLD:
            self._partial_write()
    
    def _partial_write(self):
        """Periodically write accumulated data to avoid memory issues."""
        if not self.all_mol_embeddings:
            return
        
        # Concatenate accumulated embeddings
        combined_embeddings = np.vstack(self.all_mol_embeddings)
        
        # Write to temporary datasets
        inference_group = self.file['inference']
        
        if 'temp_embeddings' not in inference_group:
            # Create temporary datasets
            temp_emb_dataset = inference_group.create_dataset(
                'temp_embeddings',
                data=combined_embeddings,
                maxshape=(None, self.embedding_dim),
                chunks=True,
                compression='gzip'
            )
        else:
            # Append to existing dataset
            temp_emb_dataset = inference_group['temp_embeddings']
            old_size = temp_emb_dataset.shape[0]
            new_size = old_size + combined_embeddings.shape[0]
            temp_emb_dataset.resize((new_size, self.embedding_dim))
            temp_emb_dataset[old_size:new_size] = combined_embeddings
        
        # Clear memory
        self.all_mol_embeddings = []
        self.file.flush()
    
    def finalize(self):
        """Finalize the file and update metadata."""
        if self.file is not None:
            # Write any remaining data
            self._partial_write()
            
            # Create final datasets
            inference_group = self.file['inference']
            
            # Handle molecular embeddings
            if 'temp_embeddings' in inference_group:
                temp_embeddings = inference_group['temp_embeddings'][:]
                del inference_group['temp_embeddings']
                
                # Create final embeddings dataset
                inference_group.create_dataset(
                    'embeddings',
                    data=temp_embeddings,
                    compression='gzip'
                )
            elif self.all_mol_embeddings:
                # Direct write if no temp data
                combined_embeddings = np.vstack(self.all_mol_embeddings)
                inference_group.create_dataset(
                    'embeddings',
                    data=combined_embeddings,
                    compression='gzip'
                )
            
            # Handle SMILES
            if self.all_smiles:
                dt = h5py.special_dtype(vlen=str)
                smiles_dataset = inference_group.create_dataset(
                    'smiles',
                    shape=(len(self.all_smiles),),
                    dtype=dt,
                    compression='gzip'
                )
                for i, smi in enumerate(self.all_smiles):
                    smiles_dataset[i] = smi
            
            # Handle atom embeddings
            if self.include_atom_embeddings:
                if self.all_atom_embeddings:
                    atom_group = inference_group.create_group('atom_embeddings')
                    for mol_idx, atom_emb in self.all_atom_embeddings.items():
                        dataset_name = f'mol_{mol_idx}'
                        atom_group.create_dataset(
                            dataset_name,
                            data=atom_emb,
                            compression='gzip'
                        )
                
                # Store atom counts
                if self.all_atom_counts:
                    inference_group.create_dataset(
                        'atom_counts',
                        data=np.array(self.all_atom_counts, dtype=np.int32),
                        compression='gzip'
                    )
            
            # Update final metadata
            self.file['metadata'].attrs['actual_molecules'] = len(self.all_smiles)
            self.file['metadata'].attrs['final_embedding_count'] = len(self.all_smiles)
            
            self.file.close()
            self.file = None
            
            if is_main_process():
                print(f"[Embeddings] Rank {self.rank} wrote {len(self.all_smiles)} embeddings to {self.rank_output_path}")
                if self.include_atom_embeddings:
                    print(f"[Embeddings] Rank {self.rank} wrote {len(self.all_atom_embeddings)} atom embedding groups")
    
    def combine_ddp_files(self):
        """Combine embedding files from all ranks (called by rank 0)."""
        if self.rank != 0 or self.world_size <= 1:
            return
        
        if dist.is_available() and dist.is_initialized():
            dist.barrier()  # Wait for all ranks to finish
        
        print(f"[Embeddings] Combining embeddings from {self.world_size} ranks...")
        
        all_embeddings = []
        all_smiles = []
        all_atom_embeddings = {}
        all_atom_counts = []
        
        # Collect data from all rank files
        for rank in range(self.world_size):
            base_path = self.output_path.with_suffix('')
            rank_file = Path(f"{base_path}_rank{rank}.h5")
            
            if rank_file.exists():
                try:
                    with h5py.File(rank_file, 'r') as f:
                        if 'inference' in f:
                            inference_group = f['inference']
                            
                            # Read embeddings
                            if 'embeddings' in inference_group:
                                embeddings = inference_group['embeddings'][:]
                                all_embeddings.append(embeddings)
                            
                            # Read SMILES
                            if 'smiles' in inference_group:
                                smiles_data = inference_group['smiles']
                                rank_smiles = [
                                    smi.decode() if isinstance(smi, bytes) else smi
                                    for smi in smiles_data[:]
                                ]
                                all_smiles.extend(rank_smiles)
                            
                            # Read atom embeddings
                            if self.include_atom_embeddings and 'atom_embeddings' in inference_group:
                                atom_group = inference_group['atom_embeddings']
                                base_idx = len(all_smiles) - len(rank_smiles) if rank_smiles else 0
                                
                                for mol_key in atom_group.keys():
                                    # Extract molecule index and adjust for global indexing
                                    mol_idx = int(mol_key.split('_')[1])
                                    global_idx = base_idx + mol_idx
                                    all_atom_embeddings[global_idx] = atom_group[mol_key][:]
                            
                            # Read atom counts
                            if self.include_atom_embeddings and 'atom_counts' in inference_group:
                                rank_atom_counts = inference_group['atom_counts'][:]
                                all_atom_counts.extend(rank_atom_counts)
                
                except Exception as e:
                    print(f"[Embeddings] Error reading rank {rank} file: {e}")
        
        # Write combined file
        if all_embeddings and all_smiles:
            combined_embeddings = np.vstack(all_embeddings)
            
            with h5py.File(self.output_path, 'w') as f:
                # Create inference group
                inference_group = f.create_group('inference')
                
                # Write combined embeddings
                inference_group.create_dataset(
                    'embeddings',
                    data=combined_embeddings,
                    compression='gzip'
                )
                
                # Write combined SMILES
                dt = h5py.special_dtype(vlen=str)
                smiles_dataset = inference_group.create_dataset(
                    'smiles',
                    shape=(len(all_smiles),),
                    dtype=dt,
                    compression='gzip'
                )
                for i, smi in enumerate(all_smiles):
                    smiles_dataset[i] = smi
                
                # Write combined atom embeddings
                if self.include_atom_embeddings and all_atom_embeddings:
                    atom_group = inference_group.create_group('atom_embeddings')
                    for mol_idx, atom_emb in all_atom_embeddings.items():
                        dataset_name = f'mol_{mol_idx}'
                        atom_group.create_dataset(
                            dataset_name,
                            data=atom_emb,
                            compression='gzip'
                        )
                
                # Write combined atom counts
                if self.include_atom_embeddings and all_atom_counts:
                    inference_group.create_dataset(
                        'atom_counts',
                        data=np.array(all_atom_counts, dtype=np.int32),
                        compression='gzip'
                    )
                
                # Write metadata
                metadata = f.create_group('metadata')
                metadata.attrs['creation_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                metadata.attrs['total_molecules'] = len(all_smiles)
                metadata.attrs['embedding_dim'] = self.embedding_dim
                metadata.attrs['include_atom_embeddings'] = self.include_atom_embeddings
                metadata.attrs['combined_from_ranks'] = self.world_size
            
            print(f"[Embeddings] Combined {len(all_smiles)} embeddings from {self.world_size} ranks")
            if self.include_atom_embeddings:
                print(f"[Embeddings] Combined {len(all_atom_embeddings)} atom embedding groups")
            print(f"[Embeddings] Final embeddings saved to: {self.output_path}")
        
        # Clean up rank files
        for rank in range(self.world_size):
            base_path = self.output_path.with_suffix('')
            rank_file = Path(f"{base_path}_rank{rank}.h5")
            if rank_file.exists():
                rank_file.unlink()
    
    def __del__(self):
        """Ensure file is closed when object is destroyed."""
        if self.file is not None:
            self.finalize()


class EmbeddingManager:
    """Manages embedding extraction and streaming writing with proper DDP support."""
    
    def __init__(self, 
                 model: nn.Module, 
                 device: torch.device,
                 output_path: str,
                 expected_total_molecules: int,
                 include_atom_embeddings: bool = False,
                 rank: int = 0,
                 world_size: int = 1):
        
        self.extractor = EmbeddingExtractor(model, device)
        self.include_atom_embeddings = include_atom_embeddings
        self.rank = rank
        self.world_size = world_size
        
        # Determine embedding dimension
        embedding_dim = self._get_embedding_dimension(model)
        
        self.writer = StreamingEmbeddingWriter(
            output_path=output_path,
            expected_total_molecules=expected_total_molecules,
            embedding_dim=embedding_dim,
            include_atom_embeddings=include_atom_embeddings,
            rank=rank,
            world_size=world_size
        )
        
        # Setup hooks
        self.extractor.setup_hooks(include_atom_embeddings)
    
    def _get_embedding_dimension(self, model: nn.Module) -> int:
        """Determine embedding dimension from model."""
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            return model.module.hidden_dim
        else:
            return model.hidden_dim
    
    def process_batch(self, batch):
        """Process a batch and write embeddings."""
        embeddings_data = self.extractor.extract_batch_embeddings(batch)
        self.writer.write_batch(embeddings_data)
    
    def finalize(self):
        """Finalize embedding writing and combine DDP files."""
        self.writer.finalize()
        
        # Combine files from all ranks if using DDP
        if self.world_size > 1:
            self.writer.combine_ddp_files()