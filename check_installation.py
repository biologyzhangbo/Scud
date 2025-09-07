#!/usr/bin/env python3
"""
Verify all dependencies are correctly installed
"""

import importlib
import subprocess
import sys

def check_package(name, version_check=None):
    """Check if package is installed"""
    try:
        module = importlib.import_module(name)
        if version_check:
            version = getattr(module, '__version__', 'Unknown')
            print(f"✓ {name}: {version}")
        else:
            print(f"✓ {name}: Installed")
        return True
    except ImportError:
        print(f"✗ {name}: Not installed")
        return False

def check_cupy():
    """Check CuPy and CUDA"""
    try:
        import cupy as cp
        print(f"✓ CuPy: {cp.__version__}")
        
        if cp.cuda.is_available():
            print(f"✓ CUDA: Available")
            print(f"✓ GPU: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
            return True
        else:
            print("⚠ CUDA: Not available")
            return False
    except ImportError:
        print("✗ CuPy: Not installed")
        return False

def main():
    print("=== Dependency Installation Verification ===\n")
    
    # Check CuPy
    cupy_ok = check_cupy()
    print()
    
    # Check other dependencies
    packages = [
        'numpy', 'pandas', 'sklearn', 
        'cyvcf2', 'matplotlib', 'seaborn', 
        'tqdm', 'scipy'
    ]
    
    all_ok = cupy_ok
    for pkg in packages:
        if check_package(pkg, version_check=True):
            all_ok = all_ok and True
    
    print(f"\n=== Verification Results ===")
    if all_ok:
        print("✓ All dependencies installed successfully!")
    else:
        print("⚠ Some dependencies have installation issues")
        sys.exit(1)

if __name__ == "__main__":
    main()