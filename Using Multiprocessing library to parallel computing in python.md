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
### Problem Statement: Count how many numbers exist between a given range in each row?
- The first problem is: Given a 2D matrix (or list of lists), count how many numbers are present between a given range in each row. We will work on the list prepared below.
```python
import numpy as np
from time import time

# Prepare data
np.random.RandomState(100)
arr = np.random.randint(0, 10, size=[200000, 5])
data = arr.tolist()
data[:5]
```
- Solution without parallelization:
```python
# Solution Without Paralleization

def howmany_within_range(row, minimum, maximum):
    """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return count

results = []
for row in data:
    results.append(howmany_within_range(row, minimum=4, maximum=8))

print(results[:10])
#> [3, 1, 4, 4, 4, 2, 1, 1, 3, 3]
```
- The general way to Parallelize any operation is to take a particular function that should run multiple times and make it run parallelly in different processors
- To do this we initialize a `Pool` with n number of processors and pass the function you want to parallelize to one of `Pool` parallelization methods
- the `Pool` class offer `Pool.map()` , `Pool.starmap()`, and `Pool.apply()`methods to execute the function in __Synchronous__ theme and `Pool.map_async()`, `Pool.starmap_async()`, and `Pool.apply_async()` methods to execute the function in __Asynchronous__ theme 
- Both `Pool.map()` and `Pool.apply()` take the function-desired-toParallyze as the main argument but the main difference that `Pool.apply()` takes an `args` Argument that accepts the parameters passed to the function whereas `Pool.map()` takes only one iterable as an argument
- In simpler words `Pool.map()` is more suitable for simpler iterable operations but does the job faster 
### Parallel Programming with `Pool.apply()`
```python
# Parallelizing using Pool.apply()

import multiprocessing as mp

# Step 1: Init multiprocessing.Pool()
pool = mp.Pool(mp.cpu_count())

# Step 2: `pool.apply` the `howmany_within_range()`
results = [pool.apply(howmany_within_range, args=(row, 4, 8)) for row in data]

# Step 3: Don't forget to close
pool.close()    

print(results[:10])
#> [3, 1, 4, 4, 4, 2, 1, 1, 3, 3]
```
### Parallel Programming using `Pool.map()`
- as we mentioned `Pool.map()` 