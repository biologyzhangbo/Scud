# Scud: GPU-accelerated Mixed Linear Model for Genome-Wide Association Studies

High-performance genome-wide association analysis tool based on CuPy optimization, supporting mixed linear models and CUDA acceleration.

## Key Features

- 🚀 **CuPy GPU Acceleration** - 5-10x faster than original version
- 📊 **Smart Batch Processing** - Support for large-scale marker processing
- 🧬 **Efficient Memory Management** - Optimized GPU memory usage
- ⚡ **Multiple Analysis Modes** - P3D (fast) and EMMA (accurate)
- 🔍 **Complete Filtering Functions** - MAF, missing rate filtering
- 📈 **Automatic Visualization** - Manhattan plots, QQ plots
- 🔧 **Flexible Configuration** - Support for CPU/GPU switching

## Installation

```bash
# Clone repository
git clone https://github.com/biologyzhangbo/Scud.git
cd Scud

# Install dependencies (choose based on your CUDA version)
pip install cupy-cuda11x  # CUDA 11.x
# or pip install cupy-cuda12x  # CUDA 12.x

pip install -r requirements.txt
pip install .
