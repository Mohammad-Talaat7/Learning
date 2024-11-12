---
Creation: 2024-11-12
tags:
  - GPU
  - Parallel
  - CPP
Type:
  - Course
Reference: Coursera
---
# Module-1
## Kernel Execution
- This material focuses on the architecture of GPUs, specifically the Ampere architecture, and how it relates to kernel execution in CUDA programming.
- High-level architecture of the GPU:
	- The GPU consists of streaming multiprocessors (SMs) that perform computations, memory controllers for data management, and interfaces for communication with the CPU and other GPUs.
	- Each SM contains warps, which are groups of threads, and various cores dedicated to different types of operations, including floating-point and integer calculations.
- Mapping software to hardware in CUDA:
	- In CUDA, a thread is the smallest computational unit, and multiple threads form a block, which is part of a grid that can be executed across multiple GPUs.
	- The kernel execution involves specifying the number of blocks and threads per block, as well as shared memory usage and data streams.
- Kernel execution structure:
	- A kernel starts with a specific syntax and includes input arguments that allow for mapping into data structures, such as matrices, to perform operations like addition.
	- Understanding the organization of threads and blocks is crucial for efficient kernel execution and data processing in CUDA.