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

### Data and Code Synchronization
