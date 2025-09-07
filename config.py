import cupy as cp
import numpy as np

class Config:
    """Global configuration class"""
    
    # Auto-detect device
    DEVICE = 'cuda' if cp.cuda.is_available() else 'cpu'
    
    # Performance optimization parameters
    DEFAULT_BATCH_SIZE = 10000
    DEFAULT_TOL = 1e-6
    DTYPE = cp.float32
    CPU_DTYPE = np.float32
    VERBOSE = True

    @classmethod
    def setup_device(cls, device=None):
        """Set compute device"""
        if device is None:
            device = cls.DEVICE
        
        if device == 'cuda' and not cp.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU mode")
            device = 'cpu'
        
        cls.DEVICE = device
        cls.xp = cp if device == 'cuda' else np
        
        if cls.VERBOSE:
            print(f"Using device: {device}")
            if device == 'cuda':
                print(f"CuPy version: {cp.__version__}")
        
        return device

# Global numerical module
xp = Config.xp if hasattr(Config, 'xp') else (cp if Config.DEVICE == 'cuda' else np)