"""
Distributed training utilities for PyTorch DDP.
"""

import pickle
from typing import List, Any, Union

import torch
import torch.distributed as dist
import numpy as np


def is_main_process() -> bool:
    """
    Returns True if the current process is the main process (rank 0) or if 
    distributed processing is not being used.
    
    Returns:
        True if main process or no distributed training
    """
    return (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)


def safe_get_rank() -> int:
    """
    Returns the current process rank. If not running in a distributed environment, returns 0.
    
    Returns:
        Process rank (0 if not distributed)
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


def get_world_size() -> int:
    """
    Returns the total number of processes in distributed training.
    
    Returns:
        World size (1 if not distributed)
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


def gather_ndarray_to_rank0(arr: np.ndarray, device: Union[str, torch.device] = "cpu") -> np.ndarray:
    """
    Gather a NumPy array from all ranks onto rank 0.
    
    Args:
        arr: NumPy array to gather
        device: Device to perform gathering on
        
    Returns:
        On rank 0: Concatenated array from all ranks
        On other ranks: Empty array with same dtype
    """
    if not (dist.is_available() and dist.is_initialized()):
        return arr
    
    # Convert local NumPy to torch tensor
    local_tensor = torch.from_numpy(arr).float().to(device)
    local_size = torch.tensor([local_tensor.size(0)], dtype=torch.long, device=device)

    # Gather all local sizes
    world_size = dist.get_world_size()
    sizes_list = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes_list, local_size)

    # Pad local tensor to max size
    max_size = max(s.item() for s in sizes_list)
    if local_size < max_size:
        pad_amount = max_size - local_size.item()
        pad_shape = (pad_amount,) + tuple(local_tensor.shape[1:])
        local_tensor = torch.cat([local_tensor, torch.zeros(pad_shape, device=device)], dim=0)

    # All-gather
    gathered_list = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_list, local_tensor)

    # Unpad each chunk and convert to CPU
    rank_idx = dist.get_rank()
    cat_list = []
    for i in range(world_size):
        valid_n = sizes_list[i].item()
        chunk = gathered_list[i][:valid_n]
        cat_list.append(chunk.cpu().numpy())

    if rank_idx == 0:
        return np.concatenate(cat_list, axis=0)
    else:
        return np.array([], dtype=arr.dtype)


def gather_strings_to_rank0(local_list: List[str], device: Union[str, torch.device] = "cpu") -> List[str]:
    """
    Gather a list of strings from all ranks onto rank 0.
    
    Args:
        local_list: List of strings on each rank
        device: Device to perform gathering on
        
    Returns:
        On rank 0: Combined list of strings from all ranks
        On other ranks: Empty list
    """
    if not (dist.is_available() and dist.is_initialized()):
        return local_list
    
    # Convert local_list into pickled bytes
    local_bytes = pickle.dumps(local_list)
    local_size = torch.tensor([len(local_bytes)], dtype=torch.long, device=device)

    # Gather all local sizes
    world_size = dist.get_world_size()
    sizes_list = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes_list, local_size)

    max_size = max(s.item() for s in sizes_list)
    # Pad local_bytes to max_size
    if local_size.item() < max_size:
        local_bytes += b'\x00' * (max_size - local_size.item())

    # Convert to tensor
    local_tensor = torch.ByteTensor(list(local_bytes)).to(device)

    # All-gather
    gathered_tensors = [torch.zeros(max_size, dtype=torch.uint8, device=device) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, local_tensor)

    rank_idx = dist.get_rank()
    if rank_idx == 0:
        all_strs = []
        for i in range(world_size):
            valid_n = sizes_list[i].item()
            chunk_bytes = gathered_tensors[i][:valid_n].cpu().numpy().tobytes()
            chunk_list = pickle.loads(chunk_bytes)
            all_strs.extend(chunk_list)
        return all_strs
    else:
        return []


def broadcast_object(obj: Any, src_rank: int = 0, device: str = "cpu") -> Any:
    """
    Broadcast a Python object from source rank to all other ranks.
    
    Args:
        obj: Object to broadcast (only meaningful on src_rank)
        src_rank: Rank to broadcast from
        device: Device for tensor operations
        
    Returns:
        Broadcasted object on all ranks
    """
    if not (dist.is_available() and dist.is_initialized()):
        return obj
    
    rank = dist.get_rank()
    
    # Serialize object on source rank
    if rank == src_rank:
        obj_bytes = pickle.dumps(obj)
        size_tensor = torch.tensor([len(obj_bytes)], dtype=torch.long, device=device)
    else:
        size_tensor = torch.tensor([0], dtype=torch.long, device=device)
    
    # Broadcast size
    dist.broadcast(size_tensor, src=src_rank)
    
    # Broadcast object
    if rank != src_rank:
        obj_bytes = bytearray(size_tensor.item())
    
    obj_tensor = torch.ByteTensor(list(obj_bytes)).to(device)
    dist.broadcast(obj_tensor, src=src_rank)
    
    # Deserialize on non-source ranks
    if rank != src_rank:
        obj = pickle.loads(obj_tensor.cpu().numpy().tobytes())
    
    return obj


def all_reduce_tensor(tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """
    All-reduce a tensor across all ranks.
    
    Args:
        tensor: Tensor to reduce
        op: Reduction operation ("sum", "mean", "max", "min")
        
    Returns:
        Reduced tensor
    """
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    
    # Map operation string to ReduceOp
    op_map = {
        "sum": dist.ReduceOp.SUM,
        "mean": dist.ReduceOp.SUM,  # We'll divide by world_size after
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
    }
    
    if op not in op_map:
        raise ValueError(f"Unsupported operation: {op}")
    
    # Perform reduction
    dist.all_reduce(tensor, op=op_map[op])
    
    # For mean, divide by world size
    if op == "mean":
        tensor /= dist.get_world_size()
    
    return tensor


def barrier() -> None:
    """
    Synchronize all processes.
    """
    if dist.is_available() and dist.is_initialized():
        dist.barrier()