import torch
import os
import time
import logging
from typing import Optional, Union
from urllib.request import urlopen
from urllib.error import URLError
import hashlib


def load_network_with_retry(
    ckpt_path: str, 
    model: torch.nn.Module, 
    max_retries: int = 3, 
    retry_delay: float = 1.0
) -> torch.nn.Module:
    """
    Load network weights with retry mechanism.
    
    Args:
        ckpt_path: Path to checkpoint file
        model: Model to load weights into
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Model with loaded weights
        
    Raises:
        RuntimeError: If loading fails after all retries
    """
    logger = logging.getLogger(__name__)
    
    for attempt in range(max_retries + 1):
        try:
            logger.debug(f"Loading network weights from {ckpt_path} (attempt {attempt + 1})")
            model = load_network(ckpt_path, model)
            logger.debug("Network weights loaded successfully")
            return model
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"Failed to load network weights (attempt {attempt + 1}): {e}")
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to load network weights after {max_retries + 1} attempts: {e}")
                raise RuntimeError(f"Failed to load network weights from {ckpt_path} after {max_retries + 1} attempts") from e
    
    # This should never be reached
    raise RuntimeError("Unexpected error in load_network_with_retry")


def load_network(ckpt_path: str, model: torch.nn.Module) -> torch.nn.Module:
    """
    Load network weights from checkpoint file.
    
    Args:
        ckpt_path: Path to checkpoint file
        model: Model to load weights into
        
    Returns:
        Model with loaded weights
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                # Assume the entire checkpoint is the state dict
                state_dict = checkpoint
        else:
            # Checkpoint is directly the state dict
            state_dict = checkpoint
        
        # Handle DataParallel models
        if list(state_dict.keys())[0].startswith('module.'):
            # Remove 'module.' prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        logger.debug(f"Loaded network weights from {ckpt_path}")
        
        return model
    except Exception as e:
        logger.error(f"Error loading network from {ckpt_path}: {e}")
        raise


def download_file_with_retry(
    url: str, 
    save_path: str, 
    max_retries: int = 3, 
    retry_delay: float = 1.0,
    chunk_size: int = 8192
) -> bool:
    """
    Download file with retry mechanism.
    
    Args:
        url: URL to download from
        save_path: Path to save downloaded file
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        chunk_size: Size of chunks to download
        
    Returns:
        True if download successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    for attempt in range(max_retries + 1):
        try:
            logger.debug(f"Downloading {url} to {save_path} (attempt {attempt + 1})")
            
            # Open URL and get file size
            with urlopen(url) as response:
                # Get total file size if available
                total_size = response.getheader('content-length')
                total_size = int(total_size) if total_size else None
                
                # Download file in chunks
                downloaded = 0
                with open(save_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress if total size is known
                        if total_size:
                            percent = (downloaded / total_size) * 100
                            logger.debug(f"Download progress: {percent:.1f}%")
            
            logger.debug(f"Download completed successfully: {save_path}")
            return True
            
        except URLError as e:
            if attempt < max_retries:
                logger.warning(f"Failed to download {url} (attempt {attempt + 1}): {e}")
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to download {url} after {max_retries + 1} attempts: {e}")
                # Clean up partial file if it exists
                if os.path.exists(save_path):
                    os.remove(save_path)
                return False
        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {e}")
            # Clean up partial file if it exists
            if os.path.exists(save_path):
                os.remove(save_path)
            return False
    
    return False


def verify_file_checksum(file_path: str, expected_checksum: str, algorithm: str = 'sha256') -> bool:
    """
    Verify file checksum.
    
    Args:
        file_path: Path to file to verify
        expected_checksum: Expected checksum
        algorithm: Hash algorithm to use (sha256, md5, etc.)
        
    Returns:
        True if checksum matches, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(file_path):
        logger.error(f"File not found for checksum verification: {file_path}")
        return False
    
    try:
        # Get hash function
        if algorithm.lower() == 'sha256':
            hash_func = hashlib.sha256()
        elif algorithm.lower() == 'md5':
            hash_func = hashlib.md5()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        # Calculate checksum
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
        
        actual_checksum = hash_func.hexdigest()
        
        # Compare checksums
        if actual_checksum.lower() == expected_checksum.lower():
            logger.debug(f"Checksum verification passed for {file_path}")
            return True
        else:
            logger.warning(f"Checksum mismatch for {file_path}")
            logger.warning(f"Expected: {expected_checksum}")
            logger.warning(f"Actual: {actual_checksum}")
            return False
            
    except Exception as e:
        logger.error(f"Error verifying checksum for {file_path}: {e}")
        return False


def get_file_size(file_path: str) -> Optional[int]:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes or None if file doesn't exist
    """
    try:
        return os.path.getsize(file_path)
    except OSError:
        return None