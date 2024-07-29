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
	- second of all the __Asynchronous__ execution when we use it we set no locks the program runs and finish all the process when it can as a result the order of results can mixed up but usually finish faster
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
- as we mentioned `Pool.map()` qccepts only one iterable as an argument so we gonna modify our function by giving it default values
```python
# Parallelizing using Pool.map()
import multiprocessing as mp

# Redefine, with only 1 mandatory argument.
def howmany_within_range_rowonly(row, minimum=4, maximum=8):
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return count

pool = mp.Pool(mp.cpu_count())

results = pool.map(howmany_within_range_rowonly, [row for row in data])

pool.close()

print(results[:10])
#> [3, 1, 4, 4, 4, 2, 1, 1, 3, 3]
```
### Parallel Programming using `Pool.starmap()`
- In pervious example we had to modify our function by giving it default values due to `Pool.map()`accepting only one iterable as parameter
- Using `Pool.starmap()`we can avoid this as it also accepts one iterable as parameter but each element in that iterable is also an iterable 
- We can provide function’s arguments is the same order in this inner iterable element, will in turn be unpacked during execution 
- In simpler words `Pool.starmap()`is a variation of `Pool.map()`that accepts argument
```python
# Parallelizing with Pool.starmap()
import multiprocessing as mp

pool = mp.Pool(mp.cpu_count())

results = pool.starmap(howmany_within_range, [(row, 4, 8) for row in data])

pool.close()

print(results[:10])
#> [3, 1, 4, 4, 4, 2, 1, 1, 3, 3]
```
### Asynchronous Parallel Processing
- The asynchronous equivalents `apply_async()`, `map_async()`and `starmap_async()` lets you do execute the processes in parallel asynchronously, that is the next process can start as soon as previous one gets over without regard for the starting order.
- As a result, there is no guarantee that the result will be in the same order as the input.
- `apply_async()` is very similar to `apply()` except that you need to provide a callback function that tells how the computed results should be stored.
- However, a caveat with `apply_async()` is, the order of numbers in the result gets jumbled up indicating the processes did not complete in the order it was started.
- A workaround for this is, we redefine a new `howmany_within_range2()` to accept and return the iteration number (`i`) as well and then sort the final results.
```python
# Parallel processing with Pool.apply_async()

import multiprocessing as mp
pool = mp.Pool(mp.cpu_count())

results = []

# Step 1: Redefine, to accept `i`, the iteration number
def howmany_within_range2(i, row, minimum, maximum):
    """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return (i, count)


# Step 2: Define callback function to collect the output in `results`
def collect_result(result):
    global results
    results.append(result)


# Step 3: Use loop to parallelize
for i, row in enumerate(data):
    pool.apply_async(howmany_within_range2, args=(i, row, 4, 8), callback=collect_result)

# Step 4: Close Pool and let all the processes complete    
pool.close()
pool.join()  # postpones the execution of next line of code until all processes in the queue are done.

# Step 5: Sort results [OPTIONAL]
results.sort(key=lambda x: x[0])
results_final = [r for i, r in results]

print(results_final[:10])
#> [3, 1, 4, 4, 4, 2, 1, 1, 3, 3]
```
- It is possible to use `apply_async()`without providing a `callback`function.
- Only that, if you don’t provide a callback, then you get a list of `pool.ApplyResult` objects which contains the computed output values from each process.
- From this, you need to use the `pool.ApplyResult.get()` method to retrieve the desired final result.
```python
# Parallel processing with Pool.apply_async() without callback function

import multiprocessing as mp
pool = mp.Pool(mp.cpu_count())

results = []

# call apply_async() without callback
result_objects = [pool.apply_async(howmany_within_range2, args=(i, row, 4, 8)) for i, row in enumerate(data)]

# result_objects is a list of pool.ApplyResult objects
results = [r.get()[1] for r in result_objects]

pool.close()
pool.join()
print(results[:10])
#> [3, 1, 4, 4, 4, 2, 1, 1, 3, 3]
```
- Implementation using `Pool.starmap_async()`
```python
# Parallelizing with Pool.starmap_async()

import multiprocessing as mp
pool = mp.Pool(mp.cpu_count())

results = []

results = pool.starmap_async(howmany_within_range2, [(i, row, 4, 8) for i, row in enumerate(data)]).get()

# With map, use `howmany_within_range_rowonly` instead
# results = pool.map_async(howmany_within_range_rowonly, [row for row in data]).get()

pool.close()
print(results[:10])
#> [3, 1, 4, 4, 4, 2, 1, 1, 3, 3]
```
## Parallize Pandas Implementation
- First, lets create a sample dataframe and see how to do row-wise and column-wise paralleization.
- Something like using `pd.apply()`on a user defined function but in parallel.
```python
import numpy as np
import pandas as pd
import multiprocessing as mp

df = pd.DataFrame(np.random.randint(3, 10, size=[5, 2]))
print(df.head())
#>    0  1
#> 0  8  5
#> 1  5  3
#> 2  3  4
#> 3  4  4
#> 4  7  9
```
- We have a dataframe. Let’s apply the `hypotenuse` function on each row, but running 4 processes at a time
- To do this, we exploit the `df.itertuples(name=False)`.
- By setting `name=False`, you are passing each row of the dataframe as a simple tuple to the `hypotenuse` function.
```python
# Row wise Operation
def hypotenuse(row):
    return round(row[1]**2 + row[2]**2, 2)**0.5

with mp.Pool(4) as pool:
    result = pool.imap(hypotenuse, df.itertuples(name=False), chunksize=10)
    output = [round(x, 2) for x in result]

print(output)
#> [9.43, 5.83, 5.0, 5.66, 11.4]
```
- That was an example of row-wise parallelization.
- Let’s also do a column-wise parallelization.
```python
# Column wise Operation
def sum_of_squares(column):
    return sum([i**2 for i in column[1]])

with mp.Pool(2) as pool:
    result = pool.imap(sum_of_squares, df.iteritems(), chunksize=10)
    output = [x for x in result]

print(output) 
#> [163, 147]
```
- Now comes the third part – Parallelizing a function that accepts a Pandas Dataframe, NumPy Array, etc. Pathos follows the `multiprocessing` style of: Pool > Map > Close > Join > Clear.
```python
import numpy as np
import pandas as pd
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool

df = pd.DataFrame(np.random.randint(3, 10, size=[500, 2]))

def func(df):
    return df.shape

cores=mp.cpu_count()

df_split = np.array_split(df, cores, axis=0)

# create the multiprocessing pool
pool = Pool(cores)

# process the DataFrame by mapping function to each df across the pool
df_out = np.vstack(pool.map(func, df_split))

# close down the pool and join
pool.close()
pool.join()
pool.clear()
```
