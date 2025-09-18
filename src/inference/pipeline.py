"""
Main inference pipeline orchestration.
"""

import os
import time
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
from multiprocessing import Pool
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm

from .config import InferenceConfig
from .preprocessing import PreprocessingReconstructor
from .uncertainty import MCDropoutPredictor, DeterministicPredictor, UncertaintyEstimator
from .embeddings import EmbeddingManager
from datasets import _worker_process_smiles, MyBatch
from torch_geometric.data import Data
from models import GNN
from utils.distributed import safe_get_rank, is_main_process



class InferencePipeline:
    """Main inference pipeline that orchestrates the entire process."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.preprocessing_pipeline = None
        self.device = None
        self.embedding_manager = None
        self.uncertainty_estimator = None
        self.deterministic_predictor = None
        
        # DDP state
        self.is_ddp = config.ddp_enabled
        self.rank = config.rank
        self.world_size = config.world_size
        
        # Statistics
        self.total_processed = 0
        self.valid_count = 0
        self.invalid_count = 0
        
        # Track processed SMILES for correspondence
        self.processed_smiles = []
        self.processed_indices = []
    
    def setup(self, device: torch.device):
        """Setup the inference pipeline components."""
        self.device = device
        
        # Load model and preprocessing
        self._load_model_and_preprocessing()
        
        # Setup prediction methods
        if self.config.mc_samples > 0:
            from .uncertainty import MCDropoutPredictor
            self.uncertainty_estimator = MCDropoutPredictor(
                model=self.model,
                device=device,
                num_samples=self.config.mc_samples
            )
        else:
            from .uncertainty import DeterministicPredictor
            self.deterministic_predictor = DeterministicPredictor(
                model=self.model,
                device=device
            )
        
        # Setup embedding extraction if requested
        if self.config.save_embeddings:
            total_molecules = self._estimate_total_molecules()
            # Adjust for DDP work distribution
            if self.is_ddp:
                _, _, rank_molecules = self._calculate_ddp_work_distribution(total_molecules)
                total_molecules = rank_molecules
            
            from .embeddings import EmbeddingManager
            self.embedding_manager = EmbeddingManager(
                model=self.model,
                device=device,
                output_path=self.config.embeddings_path,
                expected_total_molecules=total_molecules,
                include_atom_embeddings=self.config.include_atom_embeddings,
                rank=self.rank,
                world_size=self.world_size
            )

    def _load_model_and_preprocessing(self):
        """Load model and reconstruct preprocessing pipeline."""
        if not self.config.model_path or not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model file not found: {self.config.model_path}")
        
        # Load model artifact
        model_artifact = torch.load(self.config.model_path, map_location=self.device)
        
        if "hyperparams" not in model_artifact:
            raise ValueError("Model file missing hyperparams - incompatible model format")
        
        hyperparams = model_artifact["hyperparams"]
        state_dict = model_artifact["state_dict"]
        
        # CRITICAL FIX: Verify critical hyperparameters match expectations
        self._verify_model_compatibility(hyperparams)
        
        # Reconstruct preprocessing pipeline
        self.preprocessing_pipeline = PreprocessingReconstructor.load_preprocessing_pipeline(model_artifact)
        
        # Build model with EXACT hyperparameters from saved model
        self.model = self._build_model_from_hyperparams(hyperparams, state_dict)
        
        if is_main_process():
            print(f"[Pipeline] Model loaded from {self.config.model_path}")
            print(f"[Pipeline] Model hyperparameters verified and restored")
            loss_function = hyperparams.get('loss_function', 'l1')
            print(f"[Pipeline] Loss function: {loss_function}")
            print(f"[Pipeline] Hidden dim: {hyperparams.get('hidden_dim')}")
            print(f"[Pipeline] Num shells: {hyperparams.get('num_shells')}")
            print(f"[Pipeline] Task type: {hyperparams.get('task_type')}")
            print(f"[Pipeline] FFN dropout: {hyperparams.get('ffn_dropout')}")
            print(f"[Pipeline] Shell conv dropout: {hyperparams.get('shell_conv_dropout')}")
            print(f"[Pipeline] Attention heads: {hyperparams.get('attention_num_heads')}")
            if self.preprocessing_pipeline:
                print(f"[Pipeline] Preprocessing pipeline loaded successfully")

    def _verify_model_compatibility(self, hyperparams: Dict[str, Any]) -> None:
        """Verify that loaded model parameters are compatible with inference requirements."""
        required_params = [
            'hidden_dim', 'num_shells', 'num_message_passing_layers', 
            'task_type', 'loss_function'
        ]
        
        missing_params = []
        for param in required_params:
            if param not in hyperparams:
                missing_params.append(param)
        
        if missing_params:
            raise ValueError(f"Model missing critical hyperparameters: {missing_params}")
        
        # Verify inference configuration compatibility
        if hasattr(self.config, 'max_hops') and self.config.max_hops and self.config.max_hops != hyperparams.get('num_shells'):
            print(f"WARNING: Config max_hops ({self.config.max_hops}) != model num_shells ({hyperparams.get('num_shells')})")
            print(f"Using model's num_shells value: {hyperparams.get('num_shells')}")
            self.config.max_hops = hyperparams.get('num_shells')


    def _build_model_from_hyperparams(self, hyperparams: Dict[str, Any], state_dict: Dict[str, Any]) -> GNN:
        """Build model using EXACT hyperparameters from saved model."""
        
        # CRITICAL FIX: Use feature sizes from saved model, not hardcoded values
        if "feature_sizes" in hyperparams:
            feature_sizes = hyperparams["feature_sizes"]
            print(f"[Model] Using saved feature sizes: {feature_sizes}")
        else:
            # Fallback with warning
            print(f"[Model] WARNING: No feature_sizes in saved model, using defaults")
            feature_sizes = {
                'atom_type': 119,
                'hydrogen_count': 9,
                'degree': 7,
                'hybridization': 7,
            }
        
        # Get exact output dimension from state dict
        output_dim = self._get_output_dim_from_state_dict(state_dict, hyperparams)
        print(f"[Model] Building model with output_dim={output_dim}")
        
        # CRITICAL FIX: Use ALL hyperparameters from saved model with proper defaults
        model = GNN(
            feature_sizes=feature_sizes,
            hidden_dim=hyperparams["hidden_dim"],
            output_dim=output_dim,
            num_shells=hyperparams["num_shells"],
            num_message_passing_layers=hyperparams["num_message_passing_layers"],
            ffn_hidden_dim=hyperparams.get("ffn_hidden_dim", hyperparams["hidden_dim"]),
            ffn_num_layers=hyperparams.get("ffn_num_layers", 3),
            pooling_type=hyperparams.get("pooling_type", "attention"),
            task_type=hyperparams["task_type"],
            embedding_dim=hyperparams.get("embedding_dim", 64),
            use_partial_charges=hyperparams.get("use_partial_charges", False),
            use_stereochemistry=hyperparams.get("use_stereochemistry", False),
            ffn_dropout=hyperparams.get("ffn_dropout", 0.05),
            activation_type=hyperparams.get("activation_type", "silu"),
            shell_conv_num_mlp_layers=hyperparams.get("shell_conv_num_mlp_layers", 2),
            shell_conv_dropout=hyperparams.get("shell_conv_dropout", 0.05),
            attention_num_heads=hyperparams.get("attention_num_heads", 4),
            attention_temperature=hyperparams.get("attention_temperature", 1.0),
            loss_function=hyperparams.get("loss_function", "l1")
        ).to(self.device)
        
        # CRITICAL FIX: Strict loading with proper error handling
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
            if missing_keys:
                raise ValueError(f"Missing keys in state dict: {missing_keys}")
            if unexpected_keys:
                raise ValueError(f"Unexpected keys in state dict: {unexpected_keys}")
        except Exception as e:
            raise ValueError(f"Failed to load model state dict: {e}")
        
        model.eval()
        
        print(f"[Model] Model loaded successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # CRITICAL FIX: Validate that loaded model matches expected configuration
        self._validate_loaded_model(model, hyperparams)
        
        return model

    def _validate_loaded_model(self, model: GNN, hyperparams: Dict[str, Any]) -> None:
        """Validate that loaded model matches saved hyperparameters."""
        try:
            # Check critical architecture parameters
            assert model.hidden_dim == hyperparams["hidden_dim"], f"Hidden dim mismatch: {model.hidden_dim} != {hyperparams['hidden_dim']}"
            assert model.num_shells == hyperparams["num_shells"], f"Num shells mismatch: {model.num_shells} != {hyperparams['num_shells']}"
            assert model.task_type == hyperparams["task_type"], f"Task type mismatch: {model.task_type} != {hyperparams['task_type']}"
            assert model.loss_function == hyperparams.get("loss_function", "l1"), f"Loss function mismatch"
            
            print(f"[Model] ✅ Model validation passed")
            
        except AssertionError as e:
            raise ValueError(f"Model validation failed: {e}")

    def _get_output_dim_from_state_dict(self, state_dict: Dict[str, Any], hyperparams: Dict[str, Any]) -> int:
        """Determine output dimension from state dict."""
        output_keys = [
            "output_layer.weight", "module.output_layer.weight", 
            "classifier.weight", "module.classifier.weight"
        ]
        
        for key in output_keys:
            if key in state_dict:
                output_layer_size = state_dict[key].shape[0]
                
                # For evidential loss, the actual number of tasks is output_size / 4
                loss_function = hyperparams.get('loss_function', 'l1')
                if loss_function == 'evidential' and output_layer_size % 4 == 0:
                    return output_layer_size // 4
                else:
                    return output_layer_size
        
        # Fallback to hyperparams
        return hyperparams.get('output_dim', 1)
    
    def _estimate_total_molecules(self) -> int:
        """Estimate total number of molecules for embedding pre-allocation."""
        try:
            if self.config.input_path.endswith('.csv'):
                # Quick line count for CSV
                with open(self.config.input_path, 'r') as f:
                    return sum(1 for _ in f) - 1  # Subtract header
            else:
                # Default estimate for other formats
                return 10000
        except:
            return 10000
    
    def run_streaming_inference(self) -> None:
        """Run streaming inference on CSV input."""
        if not self.config.input_path.endswith('.csv'):
            raise ValueError("Streaming inference requires CSV input")
        
        try:
            # Count total molecules
            total_molecules = self._estimate_total_molecules()
            
            # Calculate work distribution for DDP
            if self.is_ddp:
                start_line, end_line, rank_molecules = self._calculate_ddp_work_distribution(total_molecules)
            else:
                start_line, end_line = 1, None  # Skip header
                rank_molecules = total_molecules
            
            if is_main_process():
                print(f"[Pipeline] Starting streaming inference for {rank_molecules} molecules")
            
            # Setup output file
            output_file = self._setup_output_file()
            
            # Process in chunks
            self._process_csv_chunks(start_line, end_line, output_file)
            
            # Combine DDP results if needed
            if self.is_ddp:
                self._combine_ddp_results(output_file)
            
            # Finalize embeddings if needed
            if self.embedding_manager:
                self.embedding_manager.finalize()
            
            if is_main_process():
                print(f"[Pipeline] Streaming inference completed")
                print(f"[Pipeline] Processed: {self.total_processed}, Valid: {self.valid_count}, Invalid: {self.invalid_count}")
        
        except Exception as e:
            print(f"[Pipeline] ERROR combining files: {e}")
    

    def _calculate_ddp_work_distribution(self, total_molecules: int) -> Tuple[int, int, int]:
        """Calculate work distribution for DDP."""
        if not self.is_ddp or self.world_size <= 1:
            return 1, None, total_molecules  # Skip header, process all
        
        # Distribute molecules across ranks
        molecules_per_rank = total_molecules // self.world_size
        remainder = total_molecules % self.world_size
        
        # Calculate start and end for this rank
        if self.rank < remainder:
            # First 'remainder' ranks get one extra molecule
            rank_molecules = molecules_per_rank + 1
            start_molecule = self.rank * rank_molecules
        else:
            # Remaining ranks get standard amount
            rank_molecules = molecules_per_rank
            start_molecule = remainder * (molecules_per_rank + 1) + (self.rank - remainder) * molecules_per_rank
        
        end_molecule = start_molecule + rank_molecules
        
        # Convert to line numbers (add 1 for header)
        start_line = start_molecule + 1  # +1 for header
        end_line = end_molecule + 1      # +1 for header
        
        print(f"[DDP] Rank {self.rank}: molecules {start_molecule}-{end_molecule-1} "
              f"(lines {start_line}-{end_line-1}): {rank_molecules} molecules")
        
        return start_line, end_line, rank_molecules
    
    def _setup_output_file(self) -> str:
        """Setup output file path for this rank."""
        if self.is_ddp and self.world_size > 1:
            base, ext = os.path.splitext(self.config.output_path)
            output_file = f"{base}_rank{self.rank}{ext}"
        else:
            output_file = self.config.output_path
        
        # Create output directory
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Write header
        header = self._generate_output_header()
        with open(output_file, 'w') as f:
            f.write(','.join(header) + '\n')
        
        return output_file
    
    def _generate_output_header(self) -> list:
        """Generate output CSV header."""
        header = [self.config.smiles_column]
        
        # Determine number of output dimensions
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            loss_function = getattr(self.model.module, 'loss_function', 'l1')
            output_layer_size = self.model.module.output_layer.weight.shape[0]
        else:
            loss_function = getattr(self.model, 'loss_function', 'l1')
            output_layer_size = self.model.output_layer.weight.shape[0]
        
        # For evidential loss, calculate actual number of tasks
        if loss_function == 'evidential' and output_layer_size % 4 == 0:
            output_dim = output_layer_size // 4
        else:
            output_dim = output_layer_size
        
        # Add prediction columns
        if output_dim > 1:  # Multi-task
            for i in range(output_dim):
                header.append(f"prediction_{i}")
                if self.config.mc_samples > 0:
                    header.append(f"uncertainty_{i}")
                elif loss_function == 'evidential':
                    header.append(f"uncertainty_{i}")
        else:  # Single task
            header.append("prediction")
            if self.config.mc_samples > 0:
                header.append("uncertainty")
            elif loss_function == 'evidential':
                header.append("uncertainty")
        
        return header
    
    def _process_csv_chunks(self, start_line: int, end_line: Optional[int], output_file: str):
        """Process CSV file in chunks."""
        # Setup chunk reading
        skiprows = list(range(1, start_line)) if start_line > 1 else None
        nrows = None if end_line is None else (end_line - start_line)
        
        chunk_iterator = pd.read_csv(
            self.config.input_path, 
            skiprows=skiprows, 
            nrows=nrows,
            chunksize=self.config.chunk_size
        )
        
        for chunk_idx, chunk_df in enumerate(chunk_iterator):
            self._process_single_chunk(chunk_df, chunk_idx, output_file)

    def _process_single_chunk(self, chunk_df: pd.DataFrame, chunk_idx: int, output_file: str):
        """Process a single chunk of data."""
        smiles_list = chunk_df[self.config.smiles_column].tolist()
        
        if not smiles_list:
            return
        
        start_time = time.time()
        
        # Parallel feature computation
        valid_data = self._compute_features_parallel(smiles_list, chunk_idx)
        
        if not valid_data['smiles']:
            if is_main_process():
                print(f"[Pipeline] No valid SMILES in chunk {chunk_idx+1}")
            return
        
        # Create data loader
        data_loader = self._create_chunk_dataloader(valid_data)
        
        # Define embedding callback for proper correspondence
        def embedding_callback(batch):
            if self.embedding_manager:
                self.embedding_manager.process_batch(batch)
        
        # Make predictions with embedding extraction
        if self.config.mc_samples > 0:
            predictions, uncertainties = self.uncertainty_estimator.predict_with_uncertainty(
                data_loader, 
                preprocessing_pipeline=self.preprocessing_pipeline,
                show_progress=False,
                embedding_callback=embedding_callback
            )
        else:
            # Check if this is an evidential model
            loss_function = getattr(self.model, 'loss_function', 'l1')
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                loss_function = getattr(self.model.module, 'loss_function', 'l1')
            
            if loss_function == 'evidential':
                # Use evidential uncertainty estimation
                predictions, uncertainties = self._predict_evidential_with_uncertainty(
                    data_loader, embedding_callback
                )
            else:
                predictions = self.deterministic_predictor.predict(
                    data_loader,
                    preprocessing_pipeline=self.preprocessing_pipeline,
                    embedding_callback=embedding_callback
                )
                uncertainties = np.zeros_like(predictions)
        
        # Write results
        self._write_chunk_results(chunk_df, valid_data, predictions, uncertainties, output_file)
        
        # Update statistics
        self.total_processed += len(valid_data['smiles'])
        processing_time = time.time() - start_time
        
        if self.rank == 0:  # Only main rank prints progress
            print(f"[Pipeline] Chunk {chunk_idx+1}: {len(valid_data['smiles'])} molecules in {processing_time:.2f}s")

    def _predict_evidential_with_uncertainty(self, data_loader, embedding_callback):
        """Make predictions with evidential uncertainty estimation."""
        self.model.eval()
        all_preds = []
        all_uncertainties = []
        all_smiles = []  # FIXED: Collect SMILES for SAE inverse transform
        
        with torch.no_grad():
            for batch in data_loader:
                if batch is None:
                    continue
                
                # FIXED: Collect SMILES from each batch
                all_smiles.extend(batch.smiles_list)
                
                # Extract embeddings if callback provided
                if embedding_callback is not None:
                    embedding_callback(batch)
                
                # Prepare batch
                batch_multi_hop_edges = batch.multi_hop_edge_indices.to(self.device)
                batch_indices = batch.batch_indices.to(self.device)
                batch_atom_features = {k: v.to(self.device) for k, v in batch.atom_features_map.items()}
                total_charges = batch.total_charges.to(self.device)
                tetrahedral_indices = batch.final_tetrahedral_chiral_tensor.to(self.device)
                cis_indices = batch.final_cis_tensor.to(self.device)
                trans_indices = batch.final_trans_tensor.to(self.device)
                
                # Forward pass
                outputs, _, _ = self.model(
                    batch_atom_features,
                    batch_multi_hop_edges,
                    batch_indices,
                    total_charges,
                    tetrahedral_indices,
                    cis_indices,
                    trans_indices
                )
                
                # Process evidential outputs
                predictions, uncertainties = self._process_evidential_outputs_with_uncertainty(outputs)
                
                all_preds.append(predictions.cpu().numpy())
                all_uncertainties.append(uncertainties.cpu().numpy())
        
        if all_preds:
            combined_preds = np.concatenate(all_preds, axis=0)
            combined_uncertainties = np.concatenate(all_uncertainties, axis=0)
            
            # FIXED: Apply inverse preprocessing with SMILES
            if self.preprocessing_pipeline:
                combined_preds = self.preprocessing_pipeline.inverse_transform(
                    smiles_list=all_smiles,  # FIXED: Pass the collected SMILES
                    transformed_targets=combined_preds
                )
            
            return combined_preds, combined_uncertainties
        
        return np.array([]), np.array([])

    def _process_evidential_outputs_with_uncertainty(self, outputs):
        """Process evidential outputs to get predictions and uncertainties."""
        batch_size = outputs.shape[0]
        if outputs.shape[1] % 4 == 0:
            num_tasks = outputs.shape[1] // 4
            evidential_params = outputs.view(batch_size, num_tasks, 4)
            
            # Extract evidential parameters
            gamma = evidential_params[:, :, 0]  # predicted mean
            nu = F.softplus(evidential_params[:, :, 1]) + 1.0  # degrees of freedom
            alpha = F.softplus(evidential_params[:, :, 2]) + 1.0  # concentration
            beta = F.softplus(evidential_params[:, :, 3])  # rate parameter
            
            # Calculate uncertainty (epistemic + aleatoric)
            aleatoric = beta / torch.clamp(alpha - 1, min=1e-6)
            epistemic = beta / (nu * torch.clamp(alpha - 1, min=1e-6))
            total_uncertainty = aleatoric + epistemic
            
            return gamma, total_uncertainty
        else:
            # Fallback for non-evidential outputs
            return outputs, torch.zeros_like(outputs)
    
    def _compute_features_parallel(self, smiles_list: list, chunk_idx: int) -> dict:
        """Compute molecular features in parallel."""
        max_hops = self.config.max_hops or 3  # Default to 3 if not set
        process_inputs = [(idx, smi, max_hops) for idx, smi in enumerate(smiles_list)]
        
        with Pool(processes=self.config.num_workers) as pool:
            results = list(pool.imap(
                _worker_process_smiles, 
                process_inputs,
                chunksize=max(1, len(process_inputs) // (self.config.num_workers * 4))
            ))
        
        # Collect valid results
        valid_data = {
            'smiles': [],
            'indices': [],
            'precomputed': []
        }
        
        for idx, precomp in results:
            if precomp is not None:
                valid_data['smiles'].append(smiles_list[idx])
                valid_data['indices'].append(idx)
                valid_data['precomputed'].append(precomp)
                self.valid_count += 1
            else:
                self.invalid_count += 1
        
        return valid_data
    
    def _create_chunk_dataloader(self, valid_data: dict) -> torch.utils.data.DataLoader:
        """Create DataLoader for a chunk of valid data."""
        data_objects = []
        
        for smi, precomp in zip(valid_data['smiles'], valid_data['precomputed']):
            data_obj = self._create_data_object(smi, precomp)
            if data_obj is not None:
                data_objects.append(data_obj)
        
        return torch.utils.data.DataLoader(
            data_objects,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=MyBatch.from_data_list,
            num_workers=0  # No multiprocessing in DataLoader for inference
        )
    
    def _create_data_object(self, smiles: str, precomp: dict) -> Optional[Data]:
        """Create a PyG Data object from precomputed features."""
        try:
            num_atoms = precomp['atom_features']['atom_type'].shape[0]
            x_dummy = torch.ones((num_atoms, 1), dtype=torch.float)
            
            data_obj = Data()
            data_obj.x = x_dummy
            data_obj.smiles = smiles
            data_obj.target = torch.tensor([0.0], dtype=torch.float)  # Dummy target
            
            # Multi-hop edges
            data_obj.multi_hop_edges = [torch.from_numpy(e).long() for e in precomp["multi_hop_edges"]]
            
            # Atom features
            atom_feats_map = {}
            for k, arr in precomp["atom_features"].items():
                atom_feats_map[k] = torch.from_numpy(arr).long()
            data_obj.atom_features_map = atom_feats_map
            
            # Stereochemistry
            data_obj.chiral_tensors = [torch.from_numpy(x).long() for x in precomp["chiral_tensors"]]
            data_obj.cis_bonds_tensors = [torch.from_numpy(x).long() for x in precomp["cis_bonds_tensors"]]
            data_obj.trans_bonds_tensors = [torch.from_numpy(x).long() for x in precomp["trans_bonds_tensors"]]
            
            # Additional features
            data_obj.total_charge = torch.tensor([precomp["total_charge"]], dtype=torch.float)
            data_obj.atomic_numbers = torch.from_numpy(precomp["atomic_numbers"]).long()
            
            return data_obj
        except Exception as e:
            print(f"[Pipeline] Error creating data object for {smiles[:30]}...: {str(e)}")
            return None
    
    def _write_chunk_results(self, chunk_df: pd.DataFrame, valid_data: dict, 
                            predictions: np.ndarray, uncertainties: np.ndarray, output_file: str):
        """Write chunk results to output file."""
        if len(predictions) == 0:
            return
        
        # Check if evidential
        loss_function = getattr(self.model, 'loss_function', 'l1')
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            loss_function = getattr(self.model.module, 'loss_function', 'l1')
        
        has_uncertainties = (self.config.mc_samples > 0) or (loss_function == 'evidential')
        
        with open(output_file, 'a') as f:
            for i, (orig_idx, pred_idx) in enumerate(zip(valid_data['indices'], range(len(predictions)))):
                smiles = chunk_df.iloc[orig_idx][self.config.smiles_column]
                line = [smiles]
                
                # Add predictions
                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    # Multi-task
                    for j in range(predictions.shape[1]):
                        line.append(str(predictions[pred_idx, j]))
                        if has_uncertainties:
                            line.append(str(uncertainties[pred_idx, j]))
                else:
                    # Single task
                    pred_val = predictions[pred_idx].item() if len(predictions.shape) > 1 else predictions[pred_idx]
                    line.append(str(pred_val))
                    if has_uncertainties:
                        unc_val = uncertainties[pred_idx].item() if len(uncertainties.shape) > 1 else uncertainties[pred_idx]
                        line.append(str(unc_val))

                f.write(','.join(line) + '\n')
            
            f.flush()  # Ensure data is written immediately
    
    def _combine_ddp_results(self, rank_output_file: str):
        """Combine results from all DDP ranks."""
        if not self.is_ddp:
            return
        
        # Non-main ranks just exit cleanly
        if self.rank != 0:
            print(f"[Pipeline] Rank {self.rank}: Finished processing, exiting...")
            return
        
        # Only rank 0 does file combination
        print(f"[Pipeline] Rank 0: Waiting briefly for other ranks to finish writing...")
        import time
        time.sleep(3)  # Give other ranks time to finish writing
        
        print(f"[Pipeline] Combining results from {self.world_size} ranks...")
        
        # Find existing rank files
        existing_files = []
        for rank in range(self.world_size):
            base, ext = os.path.splitext(self.config.output_path)
            rank_file = f"{base}_rank{rank}{ext}"
            if os.path.exists(rank_file):
                existing_files.append(rank_file)
                print(f"[Pipeline] Found: {rank_file}")
        
        if not existing_files:
            print("[Pipeline] No rank files found!")
            return
        
        # Simple file combination
        total_lines = 0
        try:
            with open(self.config.output_path, 'w') as outfile:
                # Get header from first file
                with open(existing_files[0], 'r') as f:
                    header = f.readline()
                    outfile.write(header)
                
                # Combine all rank files
                for i, rank_file in enumerate(existing_files):
                    with open(rank_file, 'r') as f:
                        f.readline()  # Skip header
                        lines_written = 0
                        for line in f:
                            line = line.strip()
                            if line:
                                outfile.write(line + '\n')
                                lines_written += 1
                                total_lines += 1
                        print(f"[Pipeline] Added {lines_written:,} lines from rank file {i}")
            
            print(f"[Pipeline] SUCCESS: Combined {total_lines:,} total predictions")
            print(f"[Pipeline] Output: {self.config.output_path}")
            
            # Clean up rank files
            for rank_file in existing_files:
                try:
                    os.remove(rank_file)
                    print(f"[Pipeline] Cleaned up: {rank_file}")
                except:
                    pass
                    
        except Exception as e:
            print(f"[Pipeline] ERROR combining files: {e}")

    def cleanup_and_exit(self):
        """Properly clean up DDP resources without hanging."""
        try:
            rank = self.rank
            print(f"[Pipeline] Rank {rank}: Starting cleanup...")
            
            # 1. Finalize any ongoing operations
            if self.embedding_manager:
                self.embedding_manager.finalize()
            
            # 2. Close file handles
            self._close_file_handles()
            
            # 3. Synchronize all ranks before GPU cleanup
            if dist.is_available() and dist.is_initialized():
                print(f"[Pipeline] Rank {rank}: Synchronizing before cleanup...")
                dist.barrier()
            
            # 4. GPU cleanup
            if torch.cuda.is_available():
                # Move model to CPU first to free GPU memory
                if hasattr(self, 'model') and self.model is not None:
                    self.model.cpu()
                
                # Clear cache and synchronize
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Reset CUDA context if possible
                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats()
            
            # 5. CPU memory cleanup
            import gc
            gc.collect()
            
            # 6. Final barrier before process group cleanup
            if dist.is_available() and dist.is_initialized():
                print(f"[Pipeline] Rank {rank}: Final synchronization...")
                dist.barrier()
                
                # Let rank 0 signal completion
                if rank == 0:
                    print("[Pipeline] All ranks synchronized, cleanup complete")
            
            print(f"[Pipeline] Rank {rank}: Cleanup successful")
            
        except Exception as e:
            print(f"[Pipeline] Rank {rank}: Cleanup error: {e}")
            # Don't re-raise - we're already in cleanup mode
        
        finally:
            # Ensure process exits cleanly
            if dist.is_available() and dist.is_initialized():
                try:
                    # Small delay to ensure all ranks reach this point
                    import time
                    time.sleep(0.1)
                except:
                    pass

    def _close_file_handles(self):
        """Close any open file handles."""
        try:
            # Close any HDF5 files
            import h5py
            # This will close any unclosed HDF5 files
            h5py.get_config().default_file_mode = 'r'
            
            # Close any CSV writers or other file handles
            # Add specific cleanup for your file handles here
            
        except Exception as e:
            print(f"[Pipeline] Warning: Error closing file handles: {e}")