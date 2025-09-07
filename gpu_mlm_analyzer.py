import cupy as cp
import numpy as np
from scipy.optimize import minimize_scalar
import scipy.stats as stats
from typing import Dict, Optional, Union, List
from tqdm import tqdm
import time
from config import Config

#version:1.0

class GPUMLMAnalyzer:
    def __init__(self, device: str = None, mode: str = 'p3d', verbose: bool = False):
        self.device = Config.setup_device(device)
        self.mode = mode
        self.verbose = verbose
        self.xp = Config.xp
        self.timings = {}

    def fit(self, genotypes, phenotypes, kinship, covariates=None, 
            tol=Config.DEFAULT_TOL, batch_size=Config.DEFAULT_BATCH_SIZE) -> Dict:
        
        start_time = time.time()
        
        genotypes = self._to_array(genotypes)
        phenotypes = self._to_array(phenotypes)
        kinship = self._to_array(kinship)
        
        if covariates is not None:
            covariates = self._to_array(covariates)
        
        n_samples, n_markers = genotypes.shape
        if len(phenotypes.shape) == 1:
            phenotypes = phenotypes.reshape(-1, 1)
        n_traits = phenotypes.shape[1]
        
        if self.verbose:
            print(f"Data shape - Genotypes: {genotypes.shape}, Phenotypes: {phenotypes.shape}")
        
        # Design matrix (with intercept)
        ones = self.xp.ones((n_samples, 1), dtype=Config.DTYPE)
        X_base = self.xp.concatenate([ones, covariates], axis=1) if covariates is not None else ones
        n_covariates = X_base.shape[1]
        
        # Eigen decomposition of kinship matrix
        eig_start = time.time()
        eigenvalues, U = self.xp.linalg.eigh(kinship)
        eigenvalues = self.xp.clip(eigenvalues, 1e-8, None)
        self.timings['eigen_decomp'] = time.time() - eig_start
        
        # Transform data
        y_trans = U.T @ phenotypes  # [n, traits]
        X_trans_base = U.T @ X_base
        
        # Global h² estimation for P3D mode
        global_h2 = None
        if self.mode == 'p3d':
            if self.verbose:
                print("P3D mode: Estimating global variance components...")
            h2_start = time.time()
            global_h2 = self._estimate_h2(y_trans[:, 0], X_trans_base, eigenvalues, tol)
            self.timings['global_h2'] = time.time() - h2_start
            if self.verbose:
                print(f"Global h²: {global_h2:.4f}")
        
        # Process markers in batches
        batch_start = time.time()
        results = self._process_in_batches(
            genotypes, X_base, U, eigenvalues, y_trans, 
            global_h2, n_markers, batch_size, n_traits, n_covariates
        )
        self.timings['batch_processing'] = time.time() - batch_start
        
        total_time = time.time() - start_time
        self.timings['total'] = total_time
        
        if self.verbose:
            self._print_timings()
        
        return self._compile_results(results, n_markers)

    def _to_array(self, data):
        """Convert data to current device"""
        if isinstance(data, np.ndarray) and self.device == 'cuda':
            return cp.asarray(data, dtype=Config.DTYPE)
        elif isinstance(data, cp.ndarray) and self.device == 'cpu':
            return cp.asnumpy(data).astype(Config.CPU_DTYPE)
        return data

    def _process_in_batches(self, genotypes, X_base, U, eigenvalues, y_trans, 
                          global_h2, n_markers, batch_size, n_traits, n_covariates):
        """Process markers in batches"""
        results = []
        n_batches = (n_markers + batch_size - 1) // batch_size
        
        if self.verbose:
            pbar = tqdm(total=n_markers, desc="Processing markers")
        
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_markers)
            batch_geno = genotypes[:, start:end]
            
            # Process current batch
            batch_results = self._process_batch(
                batch_geno, X_base, U, eigenvalues, 
                y_trans, global_h2, n_traits, n_covariates
            )
            results.extend(batch_results)
            
            if self.verbose:
                pbar.update(batch_geno.shape[1])
        
        if self.verbose:
            pbar.close()
        
        return results

    def _process_batch(self, batch_geno, X_base, U, eigenvalues, y_trans, global_h2, n_traits, n_covariates):
        """Process a batch of markers - optimized version"""
        batch_size = batch_geno.shape[1]
        n_samples = batch_geno.shape[0]
        batch_results = []
        
        # Precompute transformed base design matrix
        X_trans_base = U.T @ X_base
        
        # Pre-allocate memory for the entire batch
        for i in range(batch_size):
            # Get current marker genotype
            geno_col = batch_geno[:, i]
            
            # Transform genotype data
            geno_trans = U.T @ geno_col
            
            # Build complete design matrix (in feature space)
            X_trans = self.xp.column_stack([X_trans_base, geno_trans])
            
            # Process each phenotype
            for t in range(n_traits):
                y_t = y_trans[:, t] if n_traits > 1 else y_trans.flatten()
                
                if self.mode == 'emma':
                    h2 = self._estimate_h2(y_t, X_trans, eigenvalues, 1e-6)
                else:
                    h2 = global_h2
                
                result = self._test_marker(y_t, X_trans, eigenvalues, h2)
                
                # Save result for first phenotype
                if t == 0:
                    batch_results.append(result)
        
        return batch_results

    def _estimate_h2(self, y_trans, X_trans, eigenvalues, tol):
        """Estimate heritability h²"""
        def neg_ll(delta):
            lamb = eigenvalues + delta
            Xt_L_inv = X_trans / lamb[:, self.xp.newaxis]
            S = Xt_L_inv.T @ X_trans
            
            try:
                S_inv = self.xp.linalg.inv(S)
            except:
                return float('inf')
                
            beta = S_inv @ (Xt_L_inv.T @ y_trans)
            resid = y_trans - X_trans @ beta
            df = len(y_trans) - X_trans.shape[1]
            yPy = (resid ** 2 / lamb).sum()
            sigma_sq = yPy / df
            
            # Negative log-likelihood (ignore constant)
            ll = -0.5 * (self.xp.sum(self.xp.log(lamb)) + 
                         self.xp.log(self.xp.linalg.det(S)) + 
                         df * self.xp.log(sigma_sq))
            return -ll.item()

        try:
            res = minimize_scalar(neg_ll, bounds=(1e-8, 1e8), method='bounded', options={'xatol': tol})
            h2 = 1 / (1 + res.x)
        except Exception as e:
            if self.verbose:
                print(f"Optimization failed: {e}, using default h²=0.5")
            h2 = 0.5
        
        return max(0.0, min(h2, 0.999))

    def _test_marker(self, y_trans, X_trans, eigenvalues, h2):
        """Test association for a single marker"""
        delta = (1 - h2) / max(h2, 1e-8)
        lamb = eigenvalues + delta
        
        Xt_L_inv = X_trans / lamb[:, self.xp.newaxis]
        S = Xt_L_inv.T @ X_trans
        
        try:
            S_inv = self.xp.linalg.inv(S)
        except:
            return {'beta': float('nan'), 'se_beta': float('nan'), 
                    't_stat': float('nan'), 'p_value': float('nan')}
        
        beta = S_inv @ (Xt_L_inv.T @ y_trans)
        resid = y_trans - X_trans @ beta
        df = len(y_trans) - X_trans.shape[1]
        yPy = (resid ** 2 / lamb).sum().item()
        sigma_sq = yPy / df
        
        var_beta = sigma_sq * S_inv.diagonal()
        se_beta = self.xp.sqrt(var_beta)
        t_stat = beta / se_beta
        
        # F-test
        marker_df = 1
        F_stat = (t_stat[-1] ** 2).item()
        p_value = stats.f.sf(F_stat, marker_df, df)
        
        return {
            'beta': beta[-1].item(),
            'se_beta': se_beta[-1].item(),
            't_stat': t_stat[-1].item(),
            'p_value': p_value
        }

    def _compile_results(self, results, n_markers):
        """Compile results"""
        valid_results = [r for r in results if not np.isnan(r['p_value'])]
        
        return {
            'marker_results': results,
            'summary': {
                'n_markers': n_markers,
                'n_valid': len(valid_results),
                'min_pvalue': min((r['p_value'] for r in valid_results), default=float('nan')),
                'max_pvalue': max((r['p_value'] for r in valid_results), default=float('nan')),
                'timings': self.timings
            }
        }

    def _print_timings(self):
        """Print timing statistics"""
        print("\n=== Timing Statistics ===")
        for stage, duration in self.timings.items():
            print(f"{stage}: {duration:.2f}s")
        print("================")