# Python CUDA Packages

This repo maintains a number of Python packages requiring CUDA compilation for compatibility with recent CUDA & PyTorch versions.

To determine appropriate `TORCH_CUDA_ARCH_LIST` content in `setup.py` for your GPU, please refer to [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus).

Potentially useful `$PATH` variables:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.6
```

To install a package:
```bash
pip3 install "git+https://github.com/noidvan/python_cuda_packages.git@master#egg=<package_name>&subdirectory=<folder_name>"
```

Credit to original code authors.