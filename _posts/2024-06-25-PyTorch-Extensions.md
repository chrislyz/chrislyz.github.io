---
layout: post
---

> [!quote] Rationale
> ..., PyTorch code like above will often be fast enough. However, we can also see why, under certain circumstances, there is room for further performance improvements. The most obvious reason is that PyTorch has no knowledge of the _algorithm_ you are implementing. It knows only of the individual operations you use to compose your algorithm. As such, PyTorch must execute your operations individually, one after the other. Since each individual call to the implementation (or _kernel_) of an operation, which may involve the launch of a CUDA kernel, has a certain amount of overhead, this overhead may become significant across many function calls.

And hence, PyTorch provides with a solution for speeding up the some unknown algorithms or processes by manually launching C++ or CUDA kernel implementations.

## C++ Extension

### Building a package with `setuptools`

```python
# In file: setup.py

from setuptools import setup, find_packages, Extension
from torch.utils.cpp_extension import BuildExtension

extra_compile_args = {
    "nvcc": [
        "-O3",
        "-std=c++11",
        "--use_fast_math",
        "--thread=8",
    ]
}

# Only essential parameters are shown, more of them can be found
# in https://setuptools.pypa.io/en/latest/references/keywords.html
setup(name='package_name',
      # list of instances for Python Extensions to be built
      packages=find_packages(),
      ext_modules=[
          CUDAExtension(
              name='package_name',
              source=[
                  "csrc/pybind.cpp",
                  "csrc/path/to/kernel.cu"
              ],
              extra_compile_args=extra_compile_args,
          ),
      ],
      cmdclass={'build_ext': BuildExtension},
      # specify the PYTHON dependency along with implementation
      install_requires=['torch']
)

      
(include_dirs=cpp_extension.include_paths(),
      language='c++')
```

#### Packaging with `pyproject.toml`

In a nutshell, `pyproject.toml` is a configuration file used by python packaging, containing three main components,

- `[build-system]`: specifies the backend for building the package,
    - `requires=[]`: a list of modules required (i.e., dependencies) during building, in our case, `["setuptools", "torch"]`.
    - `build-backend`: a library that builds a source distribution or built distribution, in our case, `"setuptools.build_meta"`.
- `[project]`: specifies the project's basic meta data, such as the author, README and etc.
- `[tool]`: specifies the tool-specific configs, such as `[tools.poetry]`, `[tools.black]` and etc, optional.

```toml
[build-system]
requires = ["setuptools", "torch"]
build-backend = "setuptools.build_meta"

[project]
name = "llmfs"
version = "0.0.1"
depedencies = [
    # if any
]
requires-python = ">= 3.8"

# [tools.poetry] # if any
```

### Implementing C++ functions

```cpp
// In file csrc/path/to/cpu.cpp

#include <torch/extension.h>

#include <iostream>

// Rewrite derivative of sigmoid function in C++
torch::Tensor d_sigmoid(torch::Tensor z) {
    auto s = torch::sigmoid(z);
    return (1 - s) * s;
}
```

`<torch/extension.h>` is the essential header file contains all the necessary declarations for PyTorch with C++ extensions, including `ATen` library (i.e., contains foundational tensor and mathematical operations), `pybind11` (i.e., create Python bindings for existing C++ code) and public APIs for both.

### Exposing C++ implementations

```cpp
// In file csrc/path/to/cpu.h

#include <torch/extension.h>

torch::Tensor d_sigmoid(torch::Tensor z);
```

### Binding to Python

```cpp
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "csrc/path/cpu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def()
}
```