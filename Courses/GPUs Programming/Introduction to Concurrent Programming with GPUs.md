---
Creation: 2024-07-26T18:14:00
tags:
  - GPU
  - Python
  - CPP
Type:
  - Course
Reference: Coursera
---

# Description
T​he purpose of this course is to give a base level of knowledge in topics required to develop well-written and performant GPU code. You will learn how to write C/C++ and Python 3 code that operates asynchronously, through threading, queues, and other concurrent programming techniques. These concepts may be familiar to you or completely new. This course will give an algorithmic and programmatic background, while also challenging you to complete practical implementations of well-known patterns.

At the very end of the course, you will be introduced to the leading GPU programming framework, CUDA. While this will be the main framework used for this specialization and is a great skill to learn, this knowledge can be applied to other frameworks such as OpenCL, Metal, and OpenAcc. Also, there are abstractions in other languages, such as PyCUDA and jCuda, for those wanting to access Nvidia GPUs in their favorite languages. Also, there are frameworks such as TensorFlow and PyTorch that allow you to use machine learning techniques built around neural networks on Nvidia GPUs for improved performance. Nvidia GPUs are not the only accelerators that can be used, there are other manufacturers and technologies including AMD and FPGAs. This course will provide a basis for using them in other programming contexts.

# Week 1 
We discussed [[Introduction to Concurrent Programming with GPUs Outline]]

# Week 2
## Concurrent Programming Pitfalls
### Race Conditions
- One of the most occurring pitfalls that happens in parallel computing is Race Conditions
- Happens When the expected order of thread Operations is not followed and issues arise 
- Ex. Imagine having two threads each one read the variable `x = 0` And increase its value by one so at the end we should have `x = 2` But in fact the two threads read the variable at the same time and increased it at the same time which result in `x = 1` Instead
- It’s often the best approach is to minimizing global and shared variables or if it’s necessary the have a plan to ensure that each threads access is atomic
### Resource Contention
- Same to Race Conditions but morally to the memory and not the order of operations execution
- The more threads and shared resources the more often this will occur 
- This is more asynchronous version of Race Conditions, Which can extends from threads to completely separate machines
- Resources needed to be accessed in different ways by different independent threads and they access the same resource (memory, file, etc) This can lead to conflict in constant rewriting of values 
### Dead Lock
- Dead Lock is similar to Resource Contention it happens when a thread need multiple resources to execute and they held on resources required for another thread that need it to execute and the final result is that we have two threads can’t execute as each one of them need resources the other thread held on 
### Live Lock
- It’s like Dead Lock but processes are actively running 
### Resource Over/Under Utilization
- It happens when we have too few or too many threads and they have no thing to do or performing a series of computations that can be broken down and run in parallel therefore more efficiently 
- If too many threads are going from ACTIVE to INACTIVE status due to not having work to do, the cost for context switching can outweigh the speed offered when many threads are being used
- This can be handled by scaling the number of threads on the number of data being processed or CPU utilization 
- When threads is too few they maybe in constant use and small memory leaks or inefficiencies can compound and CPU utilization may spike, which makes all running threads sour 
- If a thread request a lot of data or lots of instructions cache hits can occur more frequently and therefore the system will be slowed by data transfers to and from the cache or RAM or even in the worst case, the hard drive
## Concurrent Programming Problems
### Dining Philosophers
- We have 5 philosophers each of them has a fork on his right and his left (5 forks)
- They want to eat eggs and side dish
- To eat each philosopher needs both forks on the right and left
- They can do one thing at a time (pick up the fork - eat - but the fork down, etc)
- We need an Algorithm that allows all philosophers to eat
### Producer-Consumer
- Producer-Consumer (Reader-Writer) pattern is very popular tool these days
- We often use it in message queues 
- Race Conditions can easily happen as the producer needs to write data in sequential way and the consumer needs to write data in sequential way or as they become available 
- Ex. If the shared counter is updated, read, then the memory is updated
### Sleeping Barber 
- the sleeping Barber Problem visualize the data queue as the waiting room and the threads as the barbers each barber have one seat and therefore can work on one customer (process) at a time
- There are two possibilities here:
	- firstly the customers number are getting bigger and bigger and there is no room in the waiting room although the barber take their time with the customer in chair which lead to live-lock or over-utilization problem 
	- Secondly there are no customers and the barbers just chatting about life and sports which can lead to under-utilization problem
- Keep in mind that if the waiting room is full and new customer arrives one customer leaves
### Data and Code Synchronization
- most of programming languages have synchronization mechanism these days it means to block access for specific code or dat until it has finished all its operations
- If we sync all data we end up having a dead-lock as all the processes will wait for the data and on the other hand the data can’t be reached until the former process end using it
## Concurrent Programming Patterns
Many of the solutions fit into these five patterns:
### Divide and Conquer
- the main idea about Divide and Conquer Pattern is to split the large dataset or large process to smaller ones each of them running through a thread and each thread return a response then taking in account all responses we got we may answer the main question
- Used in Sorting and Searching Algorithms
- If recursion is not allowed or really inefficient which is the case in CUDA then this shouldn’t been used frequently
### Map-Reduce
- A form of Divide and Conquer Pattern its main idea that for each iteration tha mapper take N data points and split it and then search for the wanted value and the reducer retun a single value (whether we found the desired value or not)
### Repository
- The Repository pattern ensures that the Repository is the only role that manages the access to the shared data to allow multiple process to work on the same data atomically
### Pipelines / Workflows
- A pattern that divide the process into number of steps each step get input and its output goes as as input to the next step
- Workflows works in more circular way (fan in / fan out) means that the same input is an output to different logical steps or data is divided up and sent to the same logical code
### Recursion
- Many complex problems can be solved via recursive calls which is simply a function that calls it self in the subsequent of its own data 
- Although it’s not the most efficient solution 
- Recursion isn’t recommended when you work on large data on local or distributed CPUs also doesn’t work well on GPUs
- Functional Programming Languages and frameworks like Lisp, Closure are build around some level of recursion and in that case data is divided into head and tail
- Functions operate on current data and call themselves with the rest
- Recursion requires a good management to get it to its final state to ensure that the recursive calls will continue in an infinite way

## Flynn’s Taxonomy
- Flynn's Taxonomy classifies all parallel operations into four categories each category start with SI (Singe-Instruction) or MI (Multi-Instruction) and end with SD (Single-Data) or MD (Multi-Data)
- ![[Pasted image 20240729023825.png]]
- Single-Instruction and Single-Data
	- Traditional Von Neumann Single CPU computer
	- An SISD computing system is a uniprocessor machine which is capable of executing a single instruction, operating on a single data stream
	- machine instructions are processed in a sequential manner and computers adopting this model are popularly called sequential computers
	- All the instructions and data to be processed have to be stored in primary memory
	- The speed of the processing element in the SISD model is limited(dependent) by the rate at which the computer can transfer information internally
	- Dominant representative SISD systems are IBM PC, workstations.
- Single-Instruction and Multi-Data
	- Vector Processors fine-grained data Parallel Computers
	- An SIMD system is a multiprocessor machine capable of executing the same instruction on all the CPUs but operating on different data streams
	- Machines based on an SIMD model are well suited to scientific computing since they involve lots of vector and matrix operations
	- So that the information can be passed to all the processing elements (PEs) organized data elements of vectors can be divided into multiple sets(N-sets for N PE systems) and each PE can process one data set.
	- Dominant representative SIMD systems is Cray’s vector processing machine
- Multi-Instruction and Single-Data
	- Maybe pipelined Computers
	- An MISD computing system is a multiprocessor machine capable of executing different instructions on different PEs but all of them operating on the same dataset
	- Example $Z = sin(x)+cos(x)+tan(x)$
	- The system performs different operations on the same data set.
	- Machines built using the MISD model are not useful in most of the application, a few machines are built, but none of them are available commercially. 
- Multi-Instruction and Multi-Data
	- Multi Computers Multi-Processors
	- An MIMD system is a multiprocessor machine which is capable of executing multiple instructions on multiple data sets.
	- Each PE in the MIMD model has separate instruction and data streams; therefore machines built using this model are capable to any kind of application.
	- Unlike SIMD and MISD machines, PEs in MIMD machines work asynchronously

# Week 3
## Python 3 Syntax
- _ thread / threading libraries [[Intro to Python Threading]]
- asyncio library [[Async IO in Python A Complete Walkthrough]]
- multiprocessing library [[Using Multiprocessing library to parallel computing in python]]
## C++ Syntax
- atomic library
- future library
- mutex library
- thread library
# Week 4
## Nivida GPU architecture
### Tesla
- produced 2007-2011
- CUDA framework
- Tesla G80 specifics:
	- 128 CUDA cores
	- 1280 MB of GDDR5 memory
	- 152 GB/s memory bandwidth
	- 150 watts
### Fermi
- Produced 2010-2012
- Introduction of 64bit floating point values
- Fermi specifics:
	- 512 CUDA cores
	- 2 GB of GDDR5 memory 
	- 192 GB/s memory bandwidth 
	- 210 watts
### Kepler
- Produced 2012-2016
- Improved programmability
- GTX 760 Specifics
	- 1152 CUDA cores
	- 4 GB of GDDR5 memory 
	- 80 GB/s memory bandwidth 
	- 170 watts
### Maxwell
- Produced 2014-2016
- Improved Performance/W
- GTX 980 Specifics
	- 2048 CUDA cores
	- 4 GB of GDDR5 memory
	- 224 GB/s memory bandwidth
	- 165 watts
### Pascal
- Produced 2016-2018
- Unified Memory and NVLink
- GTX 1080 Specifics
	- 2560 CUDA cores
	- 8 GB of GDDR5 memory
	- 320 GB/s memory bandwidth
	- 180 watts
### Turing
- Produced 2018-2020
- Tensor and Ray Tracing Cores
- GeForce RTX 2080 Specifics
	- 3072 CUDA cores
	- 8 GB of GDDR5 memory 
	- 448 GB/s memory bandwidth 
	- 215 watts
### Ampere
- Produced 2020-
- More RT and tensor cores and lower cores/W
- GeForce RTX 3070 Specifics
	- 5120 CUDA cores
	- 8 GB of GDDR6 memory 
	- 384 GB/s memory bandwidth 
	- 115 watts
## CUDA Software Layers
- Our CUDA application can communicate with either the CUDA runtime API layer (NVCC) or the driver API directly 
![[lt46q7vNR2-eOqu7zbdv8Q_eed25abbc36e4a4da116d131d4781b86_Lesson5_CUDA_Software_Layers-readonly.png]]
- If you have chosen to use the higher level CUDA runtime API it can imped the GPU code (in .ptx or .cubin extension)into the host code to do this we need to compile both types of codes via NVCC command and once done they can interoperate we still need to use a host oriented compiler like GCC or G++ which wore output in the executable thankfully via the NVCC command
- If you are targeting the driver API you have to compile the GPU code (.ptx-.fatbin) and the CPU code (host code) separately then have the host-executable interact with the GPU via the driver API
![[lt46q7vNR2-eOqu7zbdv8Q_eed25abbc36e4a4da116d131d4781b86_Lلللesson5_CUDA_Software_Layers-readonly.png]]
## CUDA Runtime Driver APIs
- determining how the code will be compiled will change the way the code is written so it’s important to have the vision of compilation in mind
### CUDA Runtime API
- The Runtime API is just an abstraction of the driver which save developers from tasks like initialization of modules and managing context 
- Another simplification is that in runtime API all kernels that were compiled into any associated GPU code are available to host code as well therefore no need to selectively load modules and kernels into the current context 
- To write code that utilize this API you will need to write in a C++ Language in general 
### CUDA Driver API
- Use it if you want more control on your code and how it works on GPU
- Can be programmed in assembly but also any language that can link and execute .cubin objects
- The driver api needs to be initialized at least once using cuInit function 
# Week 5
