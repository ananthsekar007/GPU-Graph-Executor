# GPU Graph Executor — Session Handoff

Last updated: 2026-09-25

## Goal

Complete the GPU graph executor assignment. Only these two implementation files are submitted:

- `python/dlsys/autodiff.py`
- `src/gpu_op.cu`

## Working environment

- Cluster project path: `/home/asunda45/Desktop/GPU-Graph-Executor`
- GPU: NVIDIA A100-SXM4-80GB
- Compute capability target: `sm_80`
- Driver: 595.71.05
- CUDA module selected: `cuda-11.8.0-gcc-11.2.0`
- Python virtual environment: `env`
- Python: 3.6.8
- pip: 21.3.1 (do not upgrade beyond this on Python 3.6)

Activate the environment and project configuration with:

```bash
cd /home/asunda45/Desktop/GPU-Graph-Executor
module load cuda-11.8.0-gcc-11.2.0
source env/bin/activate
export PYTHONPATH="$(pwd)/python:${PYTHONPATH}"
export LD_LIBRARY_PATH="$(pwd)/build/lib:${LD_LIBRARY_PATH}"
```

Python dependencies:

```text
numpy==1.19.5
six==1.16.0
pytest==6.2.5
```

Install them with:

```bash
python -m pip install -r requirements.txt
```

## Build configuration

The original Makefile incorrectly assumed CUDA was at `/usr/local/cuda`. The working Makefile must discover the module installation or receive its path explicitly.

Recommended definitions:

```makefile
CUDA_DIR := $(abspath $(dir $(shell command -v nvcc))/..)
ARCH = -gencode arch=compute_80,code=sm_80
```

The obsolete `compute_30`, `compute_35`, `compute_50`, and `compute_52` targets should not be used for the A100 build.

`make` now succeeds and creates:

- `build/obj/*.o`: intermediate compiled objects
- `build/lib/libc_runtime_api.so`: shared library loaded by Python through `ctypes`

## Current test status

Command:

```bash
python -m pytest -v tests/test_gpu_op.py
```

Current result:

- `test_softmax_cross_entropy` passes because its CUDA implementation was provided in the starter code.
- All other GPU-operation tests fail because their functions in `src/gpu_op.cu` are still TODO stubs.
- This is the expected baseline and does not indicate a CUDA installation problem.

## Remaining GPU work

There are 11 missing GPU operations:

1. Array set — easy CUDA kernel
2. Elementwise add — easy CUDA kernel
3. Add by constant — easy CUDA kernel
4. Elementwise multiply — easy CUDA kernel
5. Multiply by constant — easy CUDA kernel
6. ReLU — easy CUDA kernel
7. ReLU gradient — easy CUDA kernel
8. Broadcast-to — easy/medium CUDA indexing
9. Reduce-sum-axis-zero — medium reduction/indexing
10. Softmax — medium row-wise stable softmax
11. Matrix multiplication — cuBLAS wrapper; hardest because of row-major versus column-major layout

Softmax cross-entropy is already implemented.

## Remaining Python work

In `python/dlsys/autodiff.py`:

- Implement every operation's `infer_shape()` TODO.
- Implement `Executor.infer_shape()`.
- Implement `Executor.memory_plan()` so GPU arrays persist and are reused between `run()` calls.

Shape inference should be completed before GPU model training. Validate it first with one NumPy MLP epoch:

```bash
python tests/mnist_dlsys.py -l -m mlp -c numpy -e 1
```

## Current implementation progress

`DLGpuArraySet` has been implemented in the local repository with:

- A reusable `GetArraySize` helper.
- A one-dimensional, bounds-checked `array_set_kernel`.
- 256 threads per block and ceiling division for the grid size.

It still needs to be copied/synchronized to the cluster and verified there because the local machine has no `nvcc`:

```bash
make
python -m pytest -v tests/test_gpu_op.py::test_array_set
```

After it passes, the next implementation target is elementwise add-by-constant.

## Recommended implementation order

```text
Shape inference and Executor methods
    -> NumPy MLP one-epoch test
    -> array_set
    -> four elementwise arithmetic operations
    -> ReLU and ReLU gradient
    -> broadcast
    -> reduce-sum-axis-zero
    -> softmax
    -> cuBLAS matrix multiplication
    -> complete GPU test suite
    -> GPU MNIST training
```

Recompile and run the individual corresponding test after each CUDA operation. A successful `make` checks only compilation/linking, not numerical correctness.
