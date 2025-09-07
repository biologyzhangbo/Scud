import numpy as np
import pandas as pd
from cyvcf2 import VCF
from typing import Tuple, List, Dict
from config import Config

class VCFReader:
    """VCF file reader"""
    
    def __init__(self, vcf_path: str):
        self.vcf_path = vcf_path
    
    def read_vcf(self, 
                max_markers: int = None, 
                maf_threshold: float = 0.01,
                missing_threshold: float = 0.2) -> Tuple[np.ndarray, List[str], List[Dict]]:
        
        vcf = VCF(self.vcf_path)
        sample_ids = vcf.samples
        marker_info = []
        genotypes_list = []
        
        count = 0
        filtered_count = 0
        
        for variant in vcf:
            if max_markers and count >= max_markers:
                break
            
            gt_data = variant.genotype.array()
            genotypes = gt_data[:, :-1]
            
            maf, missing_rate = self._calculate_variant_stats(genotypes)
            
            if maf < maf_threshold or missing_rate > missing_threshold:
                filtered_count += 1
                continue
            
            encoded_gt = self._encode_genotypes(genotypes)
            genotypes_list.append(encoded_gt)
            
            marker_info.append({
                'chrom': variant.CHROM,
                'pos': variant.POS,
                'id': variant.ID if variant.ID else f"{variant.CHROM}:{variant.POS}",
                'ref': variant.REF,
                'alt': variant.ALT[0] if variant.ALT else '.'
            })
            
            count += 1
            if count % 10000 == 0:
                print(f"Processed {count} markers...")
        
        vcf.close()
        
        genotypes_matrix = np.array(genotypes_list, dtype=Config.CPU_DTYPE).T
        
        print(f"Successfully read {count} markers, {len(sample_ids)} samples")
        print(f"Filtered {filtered_count} markers based on MAF/missing rate")
        
        return genotypes_matrix, sample_ids, marker_info
    
    def _calculate_variant_stats(self, genotypes: np.ndarray) -> Tuple[float, float]:
        valid_genotypes = genotypes[genotypes[:, 0] >= 0]
        
        if len(valid_genotypes) == 0:
            return 0.0, 1.0
        
        allele_counts = np.sum(valid_genotypes, axis=0)
        total_alleles = len(valid_genotypes) * 2
        alt_freq = np.sum(allele_counts) / total_alleles
        maf = min(alt_freq, 1 - alt_freq)
        
        missing_rate = 1 - len(valid_genotypes) / len(genotypes)
        
        return maf, missing_rate
    
    def _encode_genotypes(self, genotypes: np.ndarray) -> np.ndarray:
        encoded = np.zeros(len(genotypes), dtype=Config.CPU_DTYPE)
        
        for i, gt in enumerate(genotypes):
            if gt[0] < 0:
                encoded[i] = np.nan
            else:
                encoded[i] = np.sum(gt[:2])
        
        mean_val = np.nanmean(encoded)
        encoded = np.nan_to_num(encoded, nan=mean_val)
        
        return encoded