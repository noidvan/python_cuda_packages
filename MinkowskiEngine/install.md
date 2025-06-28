Install [Intel oneAPI Math Kernel Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html?operatingsystem=linux&linux-install=apt) (`intel-oneapi-mkl-devel`)
```bash
pip3 install ninja
python3 setup.py install --blas=mkl --fast_math --blas_include_dirs=/opt/intel/oneapi/mkl/latest/include --blas_library_dirs=/opt/intel/oneapi/mkl/latest/lib/intel64
```