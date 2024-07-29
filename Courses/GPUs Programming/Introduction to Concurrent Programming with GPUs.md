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
- 

# Week 3
## Python 3 Syntax
### _ thread / threading libraries 
### asyncio library
### multiprocessing library
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