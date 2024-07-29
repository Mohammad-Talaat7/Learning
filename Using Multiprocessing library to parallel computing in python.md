---
Creation: 2023-07-29
tags:
  - Articles
  - Parallel
Reference: https://www.machinelearningplus.com/python/parallel-processing-python/
Type:
  - Article
---

- Parallel Computing is a strong tool to keep it in your knowledge base it makes the program more efficient by running its processes in parallel and We gonna discuss how to do it with `multiprocessing` python library
## How many Processes can I run in parallel?
- the answer for this question is that it's limited by the number of processors in your machine which you can get by running
```python
import multiprocessing as mp
print("Number of Processors: ", mp.cpu_count())
```
## What is Synchronous and Asynchronous execution?
- In the field of parallel computing there are two categories of executing program's processes:
	- first of all the __Synchronous__ execution when we use it we lock critical parts of code from the main program means that the rest of processes can't access the shared variables until the __Synchronous__ process finish
	- second of all the __Asynchronous__ execution when we use it we set no locks the program runs and finish all the process when he can as a result the order of results can mixed up but usually finish faster
## Python Implementation
- there are two main objects in `multiprocessing` to implement parallel execution:
	- `Pool` Class
		- Synchronous Execution
		- `Pool.map()` and `Pool.starmap()`
		- `Pool.apply()`
		- 
