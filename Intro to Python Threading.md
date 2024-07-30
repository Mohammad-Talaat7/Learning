---
Creation: 2024-07-30
tags:
  - Articles
  - Python
  - Parallel
Reference: https://realpython.com/intro-to-python-threading/
Type:
  - Article
---

# What is a Thread?
- A thread is a separate flow of execution. This means that your program will have two things happening at once. But for most Python 3 implementations the different threads do not actually execute at the same time: they merely appear to.
- The expectation is that on a multi-core machine a multithreaded code should make use of these extra cores and thus increase overall performance. Unfortunately the internals of the main Python interpreter, CPython, negate the possibility of true multi-threading due to a process known as the Global Interpreter Lock (GIL).
- The GIL is necessary because the Python interpreter is not _thread safe_. This means that there is a globally enforced lock when trying to safely access Python objects from within threads. At any one time only a single thread can acquire a lock for a Python object or C API. The interpreter will reacquire this lock for every 100 byte codes of Python instructions and around (potentially) blocking I/O operations. Because of this lock _CPU-bound code_ will see no gain in performance when using the Threading library, but it will likely gain performance increases if the Multiprocessing library is used.
- Architecting your program to use threading can also provide gains in design clarity. Most of the examples you’ll learn about in this tutorial are not necessarily going to run faster because they use threads. Using threading in them helps to make the design cleaner and easier to reason about.
# Starting a Thread
- To start a separate thread, you create a `Thread` instance and then tell it to `.start()`:
```python
import logging
import threading
import time

def thread_function(name):
    logging.info("Thread %s: starting", name)
    time.sleep(2)
    logging.info("Thread %s: finishing", name)

if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format,
    level=logging.INFO,datefmt="%H:%M:%S")

    logging.info("Main    : before creating thread")
    x = threading.Thread(target=thread_function, args=(1,))
    logging.info("Main    : before running thread")
    x.start()
    logging.info("Main    : wait for the thread to finish")
    # x.join()
    logging.info("Main    : all done")
```
- When you create a `Thread`, you pass it a function and a list containing the arguments to that function. In this case, you’re telling the `Thread` to run `thread_function()` and to pass it `1` as an argument.
- For this article, you’ll use sequential integers as names for your threads. There is `threading.get_ident()`, which returns a unique name for each thread, but these are usually neither short nor easily readable.
- When you run this program as it is (with line twenty commented out), the output will look like this:
```bash
$ ./single_thread.py
Main    : before creating thread
Main    : before running thread
Thread 1: starting
Main    : wait for the thread to finish
Main    : all done
Thread 1: finishing
```
## Daemon Threads
- In computer science, a _daemon_ is a process that runs in the background.
- If a program is running `Threads` that are not `daemons`, then the program will wait for those threads to complete before it terminates. `Threads` that _are_ daemons, however, are just killed wherever they are when the program is exiting.
- Let’s repeat the program with a `daemon` thread. You do that by changing how you construct the `Thread`, adding the `daemon=True` flag:
```python
x = threading.Thread(target=thread_function, args=(1,), daemon=True)
```
- When you run the program now, you should see this output:
```bash
$ ./daemon_thread.py
Main    : before creating thread
Main    : before running thread
Thread 1: starting
Main    : wait for the thread to finish
Main    : all done
```
- The difference here is that the final line of the output is missing. `thread_function()` did not get a chance to complete. It was a `daemon` thread, so when `__main__` reached the end of its code and the program wanted to finish, the daemon was killed.
## `join()` a Thread
- Daemon threads are handy, but what about when you want to wait for a thread to stop? What about when you want to do that and not exit your program? Now let’s go back to your original program and look at that commented out line twenty `x.join()`
- To tell one thread to wait for another thread to finish, you call `.join()`. If you uncomment that line, the main thread will pause and wait for the thread `x` to complete running.
# Working with many th