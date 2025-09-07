#!/usr/bin/env python3
"""
GPU-accelerated Mixed Linear Model for Genome-Wide Association Studies (GWAS)
CuPy optimized version
"""

import argparse
import numpy as np
import pandas as pd
import time
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from vcf_reader import VCFReader
from kinship import compute_kinship_matrix, validate_kinship_matrix
from gpu_mlm_analyzer import GPUMLMAnalyzer
from config import Config

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated Mixed Linear Model GWAS Analysis Tool (CuPy Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gpu-mlm-gwas --vcf data.vcf --pheno phenotypes.txt --out results
  gpu-mlm-gwas --vcf data.vcf --pheno phenotypes.txt --mode emma --out results_emma
  gpu-mlm-gwas --vcf data.vcf --pheno phenotypes.txt --device cpu --out results_cpu
  gpu-mlm-gwas --vcf data.vcf --pheno phenotypes.txt --max-markers 1000 --out test_results
"""
    )
    
    parser.add_argument('--vcf', required=True, help='Input VCF file path')
    parser.add_argument('--pheno', required=True, help='Phenotype file path')
    parser.add_argument('--out', required=True, help='Output file prefix')
    parser.add_argument('--covar', help='Covariate file path')
    parser.add_argument('--kinship', help='Pre-computed kinship matrix file')
    parser.add_argument('--device', default=None, choices=['cuda', 'cpu'], help='Compute device')
    parser.add_argument('--mode', default='p3d', choices=['p3d', 'emma'], help='Analysis mode')
    parser.add_argument('--max-markers', type=int, default=None, help='Maximum number of markers')
    parser.add_argument('--batch-size', type=int, default=Config.DEFAULT_BATCH_SIZE, help='Batch size')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--no-plot', action='store_true', help='Do not generate Manhattan plot')
    
    return parser.parse_args()

def prepare_data(args):
    vcf_reader = VCFReader(args.vcf)
    genotypes, sample_ids, marker_info = vcf_reader.read_vcf(max_markers=args.max_markers)

    print(f"Successfully read {len(marker_info)} markers, {len(sample_ids)} samples")

    pheno_df = pd.read_csv(args.pheno, sep='\t', index_col=0)
    phenotypes = pheno_df.values.astype(np.float32)
    pheno_samples = pheno_df.index.tolist()

    if np.any(np.isnan(phenotypes)):
        raise ValueError("Missing values (NaN) found in phenotype data")
    
    common_samples = list(set(sample_ids) & set(pheno_samples))
    print(f"Common samples: {len(common_samples)}")

    geno_mask = [sample_ids.index(s) for s in common_samples]
    pheno_mask = [pheno_samples.index(s) for s in common_samples]
    genotypes = genotypes[geno_mask]
    phenotypes = phenotypes[pheno_mask]

    print(f"Filtered genotype shape: {genotypes.shape}")
    print(f"Filtered phenotype shape: {phenotypes.shape}")

    phenotypes -= phenotypes.mean(axis=0)
    pheno_std = phenotypes.std(axis=0)
    if np.any(pheno_std == 0):
        raise ValueError("Phenotype variance is 0, cannot standardize")
    phenotypes /= pheno_std

    if args.kinship:
        print("Loading pre-computed kinship matrix...")
        kinship = np.load(args.kinship)
    else:
        print("Computing kinship matrix...")
        kinship = compute_kinship_matrix(genotypes)

    if not validate_kinship_matrix(kinship):
        print("Fixing kinship matrix...")
        eigenvalues, eigenvectors = np.linalg.eigh(kinship)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        kinship = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    covariates = None
    if args.covar:
        covar_df = pd.read_csv(args.covar, sep='\t', index_col=0)
        covariates = covar_df.loc[common_samples].values.astype(np.float32)
        print(f"Loaded {covariates.shape[1]} covariates")

    if args.max_markers and args.max_markers < len(marker_info):
        marker_info = marker_info[:args.max_markers]
        genotypes = genotypes[:, :args.max_markers]

    return {
        'genotypes': genotypes,
        'phenotypes': phenotypes,
        'kinship': kinship,
        'covariates': covariates,
        'marker_info': marker_info
    }

def main():
    args = parse_arguments()
    
    try:
        start_time = time.time()
        data = prepare_data(args)
        
        analyzer = GPUMLMAnalyzer(
            device=args.device, 
            mode=args.mode, 
            verbose=args.verbose
        )

        print("Starting GWAS analysis...")
        results = analyzer.fit(
            genotypes=data['genotypes'],
            phenotypes=data['phenotypes'],
            kinship=data['kinship'],
            covariates=data['covariates'],
            batch_size=args.batch_size
        )

        end_time = time.time()
        print(f"Analysis completed! Total time: {end_time - start_time:.2f} seconds")
        
        save_results(results, data['marker_info'], args.out)
        
        print(f"Results saved to: {args.out}_results.csv")
        
        if not args.no_plot:
            try:
                generate_manhattan_plot(results, data['marker_info'], args.out)
                print(f"Manhattan plot saved to: {args.out}_manhattan.png")
            except Exception as e:
                print(f"Failed to generate Manhattan plot: {e}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def save_results(results, marker_info, output_prefix, language='en'):
    """Save analysis results to file, handling encoding issues"""
    # Create results DataFrame
    results_df = pd.DataFrame({
        'CHROM': [info['chrom'] for info in marker_info],
        'POS': [info['pos'] for info in marker_info],
        'ID': [info['id'] for info in marker_info],
        'REF': [info['ref'] for info in marker_info],
        'ALT': [info['alt'] for info in marker_info],
        'BETA': [r['beta'] for r in results['marker_results']],
        'SE': [r['se_beta'] for r in results['marker_results']],
        'T_STAT': [r['t_stat'] for r in results['marker_results']],
        'P_VALUE': [r['p_value'] for r in results['marker_results']]
    })
    
    # Save CSV results (using UTF-8 encoding)
    results_df.to_csv(f"{output_prefix}_results.csv", index=False, encoding='utf-8')
    
    # Save summary statistics (using UTF-8 encoding)
    summary = results['summary']
    with open(f"{output_prefix}_summary.txt", 'w', encoding='utf-8') as f:
        f.write("GWAS Analysis Summary\n")
        f.write("=====================\n")
        f.write(f"Total markers: {summary['n_markers']}\n")
        f.write(f"Valid markers: {summary['n_valid']}\n")
        f.write(f"Min p-value: {summary['min_pvalue']:.2e}\n")
        f.write(f"Max p-value: {summary['max_pvalue']:.2e}\n")
        f.write("\nTime statistics:\n")
        for stage, duration in summary.get('timings', {}).items():
            f.write(f"{stage}: {duration:.2f}s\n")

def generate_manhattan_plot(results, marker_info, output_prefix):
    """Generate Manhattan plot, handling encoding issues"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set fonts to avoid Chinese character issues
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Prepare plot data
    plot_data = []
    for i, info in enumerate(marker_info):
        if i < len(results['marker_results']):
            result = results['marker_results'][i]
            if not np.isnan(result['p_value']):
                plot_data.append({
                    'CHROM': info['chrom'],
                    'POS': info['pos'],
                    'P_VALUE': result['p_value']
                })
    
    if not plot_data:
        print("Warning: No valid data for Manhattan plot")
        return
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure
    plt.figure(figsize=(14, 6))
    
    # Plot by chromosome groups
    chroms = sorted(plot_df['CHROM'].unique())
    colors = plt.cm.tab20.colors
    
    for i, chrom in enumerate(chroms):
        chrom_data = plot_df[plot_df['CHROM'] == chrom]
        color = colors[i % len(colors)]
        plt.scatter(chrom_data['POS'], -np.log10(chrom_data['P_VALUE']), 
                   color=color, alpha=0.7, s=15, label=f'Chr{chrom}')
    
    # Add significance threshold lines
    plt.axhline(y=-np.log10(5e-8), color='red', linestyle='--', 
                linewidth=1, label='Genome-wide significant (5e-8)')
    plt.axhline(y=-np.log10(1e-6), color='orange', linestyle='--', 
                linewidth=1, label='Suggestive (1e-6)')
    
    plt.xlabel('Chromosome Position')
    plt.ylabel('-log10(p-value)')
    plt.title('Manhattan Plot - GPU-MLM GWAS Results')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save image
    plt.savefig(f"{output_prefix}_manhattan.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()