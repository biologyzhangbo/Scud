"""
GPU-MLM-GWAS: GPU-accelerated Mixed Linear Model GWAS Analysis Tool
"""

__version__ = "0.1.0"
__author__ = "Bo Zhang"

from .gpu_mlm import main
from .gpu_mlm_analyzer import GPUMLMAnalyzer
from .vcf_reader import VCFReader
from .kinship import compute_kinship_matrix

__all__ = [
    'main',
    'GPUMLMAnalyzer', 
    'VCFReader',
    'compute_kinship_matrix'
]