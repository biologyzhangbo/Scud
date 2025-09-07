import numpy as np
import cupy as cp
from sklearn.preprocessing import StandardScaler
from typing import Literal
from config import Config

def compute_kinship_matrix(genotypes: np.ndarray, 
                          method: Literal['vanraden', 'centered'] = 'vanraden') -> np.ndarray:
    """Compute kinship relationship matrix"""
    n_samples, n_markers = genotypes.shape
    
    if n_markers == 0:
        raise ValueError("No markers available for kinship computation")
    
    # Standardize genotype data
    scaler = StandardScaler()
    genotypes_std = scaler.fit_transform(genotypes)
    
    if method == 'vanraden':
        M = genotypes_std
        kinship = M @ M.T / n_markers
    else:
        kinship = genotypes_std @ genotypes_std.T / n_markers
    
    # Ensure matrix symmetry
    kinship = (kinship + kinship.T) / 2
    np.fill_diagonal(kinship, kinship.diagonal() + 1e-8)
    
    return kinship.astype(Config.CPU_DTYPE)

def validate_kinship_matrix(kinship: np.ndarray) -> bool:
    """Validate kinship matrix properties"""
    if not np.allclose(kinship, kinship.T):
        print("Warning: Kinship matrix is not symmetric")
        return False
    
    try:
        eigenvalues = np.linalg.eigvalsh(kinship)
        if np.any(eigenvalues < -1e-8):
            print("Warning: Kinship matrix has negative eigenvalues")
            return False
    except:
        print("Warning: Unable to compute kinship matrix eigenvalues")
        return False
    
    return True

def save_kinship_matrix(kinship: np.ndarray, filename: str):
    """Save kinship matrix to file"""
    np.save(filename, kinship)
    print(f"Kinship matrix saved: {filename}")

def load_kinship_matrix(filename: str) -> np.ndarray:
    """Load kinship matrix from file"""
    kinship = np.load(filename)
    if not validate_kinship_matrix(kinship):
        print("Warning: Loaded kinship matrix may have issues")
    return kinship