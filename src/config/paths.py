# config/paths.py
"""
Path management and directory creation utilities with robust error handling.
"""

import os
import stat
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)


class PathError(Exception):
    """Raised when path operations fail."""
    pass


def setup_paths(args) -> None:
    """
    Setup and validate all paths in arguments with comprehensive error handling.

    Args:
        args: Arguments containing paths to setup

    Raises:
        PathError: If path setup fails
    """
    try:
        logger.info("Setting up paths...")

        # Create output directories
        _create_output_directories(args)

        # Setup HDF5 paths if needed
        if getattr(args, 'iterable_dataset', False):
            _create_hdf5_directories(args)

        # Validate input paths
        _validate_input_paths(args)

        logger.info("Path setup completed successfully")

    except Exception as e:
        raise PathError(f"Failed to setup paths: {e}") from e


def _create_output_directories(args) -> None:
    """Create directories for output files."""
    output_paths = []

    # Collect all output paths
    if hasattr(args, 'model_save_path') and args.model_save_path:
        output_paths.append(args.model_save_path)

    if hasattr(args, 'inference_output') and args.inference_output:
        output_paths.append(args.inference_output)

    if hasattr(args, 'embeddings_output_path') and args.embeddings_output_path:
        output_paths.append(args.embeddings_output_path)

    if hasattr(args, 'output_partial_charges') and args.output_partial_charges:
        output_paths.append(args.output_partial_charges)

    # Create parent directories
    for file_path in output_paths:
        if file_path:
            try:
                parent_dir = Path(file_path).parent
                create_directories([str(parent_dir)])
            except Exception as e:
                raise PathError(f"Failed to create directory for {file_path}: {e}") from e


def _create_hdf5_directories(args) -> None:
    """Create directories for HDF5 files if they don't exist."""
    hdf5_paths = []

    # Collect HDF5 paths
    if hasattr(args, 'train_hdf5') and args.train_hdf5:
        hdf5_paths.append(args.train_hdf5)
    if hasattr(args, 'val_hdf5') and args.val_hdf5:
        hdf5_paths.append(args.val_hdf5)
    if hasattr(args, 'test_hdf5') and args.test_hdf5:
        hdf5_paths.append(args.test_hdf5)

    for path in hdf5_paths:
        try:
            dirname = os.path.dirname(path)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
                logger.info(f"  Created HDF5 directory: {dirname}")
        except Exception as e:
            raise PathError(f"Failed to create HDF5 directory for {path}: {e}") from e


def _validate_input_paths(args) -> None:
    """Validate that input paths exist and are accessible."""
    input_paths = []

    # Collect input paths
    if hasattr(args, 'data_path') and args.data_path:
        input_paths.append(('data_path', args.data_path))

    if hasattr(args, 'train_data') and args.train_data:
        input_paths.append(('train_data', args.train_data))
    if hasattr(args, 'val_data') and args.val_data:
        input_paths.append(('val_data', args.val_data))
    if hasattr(args, 'test_data') and args.test_data:
        input_paths.append(('test_data', args.test_data))

    if hasattr(args, 'transfer_learning') and args.transfer_learning:
        input_paths.append(('transfer_learning', args.transfer_learning))

    if hasattr(args, 'hyperparameter_file') and args.hyperparameter_file:
        input_paths.append(('hyperparameter_file', args.hyperparameter_file))

    if hasattr(args, 'inference_csv') and args.inference_csv:
        input_paths.append(('inference_csv', args.inference_csv))
    if hasattr(args, 'inference_hdf5') and args.inference_hdf5:
        input_paths.append(('inference_hdf5', args.inference_hdf5))

    # Validate each path
    for path_name, path_value in input_paths:
        if not os.path.exists(path_value):
            raise PathError(f"Input file not found for {path_name}: {path_value}")

        if not os.access(path_value, os.R_OK):
            raise PathError(f"Cannot read input file for {path_name}: {path_value}")


def create_directories(dir_paths: list[str]) -> None:
    """
    Create parent directories for given paths with robust error handling.

    Args:
        dir_paths: List of directory paths to create

    Raises:
        PathError: If directory creation fails
    """
    for dir_path in dir_paths:
        if not dir_path:
            continue

        try:
            path_obj = Path(dir_path)

            # Skip if directory already exists
            if path_obj.exists() and path_obj.is_dir():
                continue

            # Create directory with parents
            path_obj.mkdir(parents=True, exist_ok=True)

            # Verify creation was successful
            if not path_obj.exists():
                raise PathError(f"Directory creation failed: {dir_path}")

            logger.info(f"  Created directory: {dir_path}")

        except PermissionError:
            raise PathError(f"Permission denied creating directory: {dir_path}")
        except OSError as e:
            raise PathError(f"OS error creating directory {dir_path}: {e}") from e
        except Exception as e:
            raise PathError(f"Unexpected error creating directory {dir_path}: {e}") from e


def ensure_path_exists(file_path: str, create_parents: bool = True) -> bool:
    """
    Ensure a file path exists or can be created.

    Args:
        file_path: Path to check/create
        create_parents: Whether to create parent directories

    Returns:
        bool: True if path exists or was created successfully

    Raises:
        PathError: If path cannot be created
    """
    try:
        path_obj = Path(file_path)

        # If file already exists, check if it's writable
        if path_obj.exists():
            if not os.access(file_path, os.W_OK):
                raise PathError(f"File exists but is not writable: {file_path}")
            return True

        # Create parent directories if requested
        if create_parents:
            parent_dir = path_obj.parent
            if not parent_dir.exists():
                create_directories([str(parent_dir)])

        # Try to create and remove a test file to check writability
        try:
            path_obj.touch()
            path_obj.unlink()  # Remove the test file
            return True
        except PermissionError:
            raise PathError(f"Cannot write to path: {file_path}")
        except OSError as e:
            raise PathError(f"OS error accessing path {file_path}: {e}") from e

    except PathError:
        raise
    except Exception as e:
        raise PathError(f"Unexpected error checking path {file_path}: {e}") from e


def get_default_paths(base_dir: str = ".") -> dict[str, str]:
    """
    Get default paths for common files with error handling.

    Args:
        base_dir: Base directory for relative paths

    Returns:
        Dictionary of default paths

    Raises:
        PathError: If base directory is invalid
    """
    try:
        base_path = Path(base_dir).resolve()

        if not base_path.exists():
            raise PathError(f"Base directory does not exist: {base_dir}")

        if not base_path.is_dir():
            raise PathError(f"Base path is not a directory: {base_dir}")

        return {
            'model_save_path': str(base_path / "models" / "gnn_model.pth"),
            'embeddings_output_path': str(base_path / "embeddings" / "molecular_embeddings.h5"),
            'train_hdf5': str(base_path / "data" / "processed" / "train.h5"),
            'val_hdf5': str(base_path / "data" / "processed" / "val.h5"),
            'test_hdf5': str(base_path / "data" / "processed" / "test.h5"),
            'results_dir': str(base_path / "results"),
            'logs_dir': str(base_path / "logs"),
            'config_dir': str(base_path / "configs"),
        }

    except Exception as e:
        raise PathError(f"Failed to generate default paths: {e}") from e


def check_disk_space(path: str, required_gb: float = 1.0) -> bool:
    """
    Check if there's enough disk space at the given path.

    Args:
        path: Path to check disk space for
        required_gb: Required space in GB

    Returns:
        bool: True if enough space is available

    Raises:
        PathError: If disk space cannot be checked
    """
    try:
        # Get the directory to check (use parent if path is a file)
        check_path = Path(path)
        if check_path.is_file() or not check_path.exists():
            check_path = check_path.parent

        # Get disk usage statistics
        statvfs = os.statvfs(str(check_path))

        # Calculate available space in bytes
        available_bytes = statvfs.f_frsize * statvfs.f_available
        available_gb = available_bytes / (1024**3)

        if available_gb < required_gb:
            logger.warning(f"Low disk space: {available_gb:.2f} GB available, {required_gb:.2f} GB required")
            return False

        return True

    except Exception as e:
        raise PathError(f"Failed to check disk space for {path}: {e}") from e


def backup_file(file_path: str, backup_suffix: str = ".backup") -> str | None:
    """
    Create a backup of an existing file.

    Args:
        file_path: Path to file to backup
        backup_suffix: Suffix to add to backup file

    Returns:
        Path to backup file if created, None if original doesn't exist

    Raises:
        PathError: If backup creation fails
    """
    try:
        source_path = Path(file_path)

        if not source_path.exists():
            return None

        # Create backup filename
        backup_path = source_path.with_suffix(source_path.suffix + backup_suffix)

        # If backup already exists, add timestamp
        if backup_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = source_path.with_suffix(f"{source_path.suffix}.{timestamp}{backup_suffix}")

        # Copy file to backup location
        shutil.copy2(str(source_path), str(backup_path))

        logger.info(f"  Created backup: {backup_path}")
        return str(backup_path)

    except Exception as e:
        raise PathError(f"Failed to create backup of {file_path}: {e}") from e


def clean_old_files(directory: str, pattern: str = "*", max_age_days: int = 30) -> int:
    """
    Clean old files from a directory.

    Args:
        directory: Directory to clean
        pattern: File pattern to match
        max_age_days: Maximum age in days before deletion

    Returns:
        Number of files deleted

    Raises:
        PathError: If cleaning fails
    """
    try:
        import time
        from datetime import datetime, timedelta

        dir_path = Path(directory)
        if not dir_path.exists():
            return 0

        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        deleted_count = 0

        for file_path in dir_path.glob(pattern):
            if file_path.is_file():
                file_age = file_path.stat().st_mtime
                if file_age < cutoff_time:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.info(f"  Deleted old file: {file_path}")
                    except Exception as e:
                        logger.warning(f"  Failed to delete {file_path}: {e}")

        if deleted_count > 0:
            logger.info(f"  Cleaned {deleted_count} old files from {directory}")

        return deleted_count

    except Exception as e:
        raise PathError(f"Failed to clean old files from {directory}: {e}")


def get_file_info(file_path: str) -> dict[str, Any]:
    """
    Get detailed information about a file.

    Args:
        file_path: Path to analyze

    Returns:
        Dictionary with file information

    Raises:
        PathError: If file info cannot be retrieved
    """
    try:
        path_obj = Path(file_path)

        if not path_obj.exists():
            return {'exists': False}

        stat_info = path_obj.stat()

        return {
            'exists': True,
            'is_file': path_obj.is_file(),
            'is_dir': path_obj.is_dir(),
            'size_bytes': stat_info.st_size,
            'size_mb': stat_info.st_size / (1024**2),
            'modified_time': datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            'permissions': oct(stat_info.st_mode)[-3:],
            'readable': os.access(file_path, os.R_OK),
            'writable': os.access(file_path, os.W_OK),
            'executable': os.access(file_path, os.X_OK),
        }

    except Exception as e:
        raise PathError(f"Failed to get file info for {file_path}: {e}") from e


# Legacy compatibility function
def check_and_create_hdf5_directories(args) -> None:
    """Legacy function for backward compatibility."""
    try:
        _create_hdf5_directories(args)
    except Exception as e:
        logger.warning(f"Failed to create HDF5 directories: {e}")
