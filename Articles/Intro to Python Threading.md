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
# Working with many threads
- The example code so far has only been working with two threads: the main thread and one you started with the `threading.Thread` object.
- Frequently, you’ll want to start a number of threads and have them do interesting work. Let’s start by looking at the harder way of doing that, and then you’ll move on to an easier method.
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
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    threads = list()
    for index in range(3):
        logging.info("Main    : create and start thread %d.", index)
        x = threading.Thread(target=thread_function, args=(index,))
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        logging.info("Main    : before joining thread %d.", index)
        thread.join()
        logging.info("Main    : thread %d done", index)
```
- This code uses the same mechanism you saw above to start a thread, create a `Thread` object, and then call `.start()`. The program keeps a list of `Thread` objects so that it can then wait for them later using `.join()`.
- Running this code multiple times will likely produce some interesting results. Here’s an example output from my machine:
```bash
$ ./multiple_threads.py
Main    : create and start thread 0.
Thread 0: starting
Main    : create and start thread 1.
Thread 1: starting
Main    : create and start thread 2.
Thread 2: starting
Main    : before joining thread 0.
Thread 2: finishing
Thread 1: finishing
Thread 0: finishing
Main    : thread 0 done
Main    : before joining thread 1.
Main    : thread 1 done
Main    : before joining thread 2.
Main    : thread 2 done
```
- If you walk through the output carefully, you’ll see all three threads getting started in the order you might expect, but in this case they finish in the opposite order! Multiple runs will produce different orderings. Look for the `Thread x: finishing` message to tell you when each thread is done.
- The order in which threads are run is determined by the operating system and can be quite hard to predict. It may (and likely will) vary from run to run, so you need to be aware of that when you design algorithms that use threading.
- Fortunately, Python gives you several primitives that you’ll look at later to help coordinate threads and get them running together. Before that, let’s look at how to make managing a group of threads a bit easier.
# Using a ThreadPoolExecuter
- There’s an easier way to start up a group of threads than the one you saw above. It’s called a `ThreadPoolExecutor`, and it’s part of the standard library in `concurrent.futures` (as of Python 3.2).
- The easiest way to create it is as a context manager, using the `with` to manage the creation and destruction of the pool.
- Here’s the `__main__` from the last example rewritten to use a `ThreadPoolExecutor`:
```python
import concurrent.futures

# [rest of code]

if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(thread_function, range(3))
```
- The code creates a `ThreadPoolExecutor` as a context manager, telling it how many worker threads it wants in the pool. It then uses `.map()` to step through an iterable of things, in your case `range(3)`, passing each one to a thread in the pool.
- The end of the `with` block causes the `ThreadPoolExecutor` to do a `.join()` on each of the threads in the pool. It is _strongly_ recommended that you use `ThreadPoolExecutor` as a context manager when you can so that you never forget to `.join()` the threads.
- Running your corrected example code will produce output that looks like this:
```bash
$ ./executor.py
Thread 0: starting
Thread 1: starting
Thread 2: starting
Thread 1: finishing
Thread 0: finishing
Thread 2: finishing
```
# Basic Synchronization Using `Lock`
- To solve your race condition, you need to find a way to allow only one thread at a time into the read-modify-write section of your code. The most common way to do this is called `Lock` in Python. In some other languages this same idea is called a `mutex`. Mutex comes from MUTual EXclusion, which is exactly what a `Lock` does.
- A `Lock` is an object that acts like a hall pass. Only one thread at a time can have the `Lock`. Any other thread that wants the `Lock` must wait until the owner of the `Lock` gives it up.
- The basic functions to do this are `.acquire()` and `.release()`. A thread will call `my_lock.acquire()` to get the lock. If the lock is already held, the calling thread will wait until it is released. There’s an important point here. If one thread gets the lock but never gives it back, your program will be stuck. You’ll read more about this later.
- Fortunately, Python’s `Lock` will also operate as a context manager, so you can use it in a `with` statement, and it gets released automatically when the `with` block exits for any reason.
```python
class FakeDatabase:
    def __init__(self):
        self.value = 0
        self._lock = threading.Lock()

    def locked_update(self, name):
        logging.info("Thread %s: starting update", name)
        logging.debug("Thread %s about to lock", name)
        with self._lock:
            logging.debug("Thread %s has lock", name)
            local_copy = self.value
            local_copy += 1
            time.sleep(0.1)
            self.value = local_copy
            logging.debug("Thread %s about to release lock", name)
        logging.debug("Thread %s after release", name)
        logging.info("Thread %s: finishing update", name)
```
- If you run this version with logging set to warning level, you’ll see this:
```bash
$ ./fixrace.py
Testing locked update. Starting value is 0.
Thread 0: starting update
Thread 1: starting update
Thread 0: finishing update
Thread 1: finishing update
Testing locked update. Ending value is 2.
```
- Look at that. Your program finally works!
- You can turn on full logging by setting the level to `DEBUG` by adding this statement after you configure the logging output in `__main__`:
```python
logging.getLogger().setLevel(logging.DEBUG)
```
- Running this program with `DEBUG` logging turned on looks like this:
```bash
$ ./fixrace.py
Testing locked update. Starting value is 0.
Thread 0: starting update
Thread 0 about to lock
Thread 0 has lock
Thread 1: starting update
Thread 1 about to lock
Thread 0 about to release lock
Thread 0 after release
Thread 0: finishing update
Thread 1 has lock
Thread 1 about to release lock
Thread 1 after release
Thread 1: finishing update
Testing locked update. Ending value is 2.
```
- In this output you can see `Thread 0` acquires the lock and is still holding it when it goes to sleep. `Thread 1` then starts and attempts to acquire the same lock. Because `Thread 0` is still holding it, `Thread 1` has to wait. This is the mutual exclusion that a `Lock` provides.
# Producer-Consumer Threading
- The Producer-Consumer Problem is a standard computer science problem used to look at threading or process synchronization issues. You’re going to look at a variant of it to get some ideas of what primitives the Python `threading` module provides.
- For this example, you’re going to imagine a program that needs to read messages from a network and write them to disk. The program does not request a message when it wants. It must be listening and accept messages as they come in. The messages will not come in at a regular pace, but will be coming in bursts. This part of the program is called the producer.
- On the other side, once you have a message, you need to write it to a database. The database access is slow, but fast enough to keep up to the average pace of messages. It is _not_ fast enough to keep up when a burst of messages comes in. This part is the consumer.
- In between the producer and the consumer, you will create a `Pipeline` that will be the part that changes as you learn about different synchronization objects.
- That’s the basic layout. Let’s look at a solution using `Lock`. It doesn’t work perfectly, but it uses tools you already know, so it’s a good place to start.
## Producer-Consumer Using `Lock`
- Since this is an article about Python `threading`, and since you just read about the `Lock` primitive, let’s try to solve this problem with two threads using a `Lock` or two.
- The general design is that there is a `producer` thread that reads from the fake network and puts the message into a `Pipeline`:
```python
import random 

SENTINEL = object()

def producer(pipeline):
    """Pretend we're getting a message from the network."""
    for index in range(10):
        message = random.randint(1, 101)
        logging.info("Producer got message: %s", message)
        pipeline.set_message(message, "Producer")

    # Send a sentinel message to tell consumer we're done
    pipeline.set_message(SENTINEL, "Producer")
```
- To generate a fake message, the `producer` gets a random number between one and one hundred. It calls `.set_message()` on the `pipeline` to send it to the `consumer`.\
- The `producer` also uses a `SENTINEL` value to signal the consumer to stop after it has sent ten values. This is a little awkward, but don’t worry, you’ll see ways to get rid of this `SENTINEL` value after you work through this example.
- On the other side of the `pipeline` is the consumer:
```python
def consumer(pipeline):
    """Pretend we're saving a number in the database."""
    message = 0
    while message is not SENTINEL:
        message = pipeline.get_message("Consumer")
        if message is not SENTINEL:
            logging.info("Consumer storing message: %s", message)
```
- The `consumer` reads a message from the `pipeline` and writes it to a fake database, which in this case is just printing it to the display. If it gets the `SENTINEL` value, it returns from the function, which will terminate the thread.
- Before you look at the really interesting part, the `Pipeline`, here’s the `__main__` section, which spawns these threads:
```python
if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")
    # logging.getLogger().setLevel(logging.DEBUG)

    pipeline = Pipeline()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(producer, pipeline)
        executor.submit(consumer, pipeline)
```
- It can be worthwhile to walk through the `DEBUG` logging messages to see exactly where each thread acquires and releases the locks.
- Now let’s take a look at the `Pipeline` that passes messages from the `producer` to the `consumer`:
```python
class Pipeline:
    """
    Class to allow a single element pipeline between producer and consumer.
    """
    def __init__(self):
        self.message = 0
        self.producer_lock = threading.Lock()
        self.consumer_lock = threading.Lock()
        self.consumer_lock.acquire()

    def get_message(self, name):
        logging.debug("%s:about to acquire getlock", name)
        self.consumer_lock.acquire()
        logging.debug("%s:have getlock", name)
        message = self.message
        logging.debug("%s:about to release setlock", name)
        self.producer_lock.release()
        logging.debug("%s:setlock released", name)
        return message

    def set_message(self, message, name):
        logging.debug("%s:about to acquire setlock", name)
        self.producer_lock.acquire()
        logging.debug("%s:have setlock", name)
        self.message = message
        logging.debug("%s:about to release getlock", name)
        self.consumer_lock.release()
        logging.debug("%s:getlock released", name)
```
- That seems a bit more manageable. The `Pipeline` in this version of your code has three members:
	- `.message` stores the message to pass.
	- `.producer_lock` is a `threading.Lock` object that restricts access to the message by the `producer` thread.
	- `.consumer_lock` is also a `threading.Lock` that restricts access to the message by the `consumer` thread.
- `__init__()` initializes these three members and then calls `.acquire()` on the `.consumer_lock`. This is the state you want to start in. The `producer` is allowed to add a new message, but the `consumer` needs to wait until a message is present.
- `.get_message()` and `.set_messages()` are nearly opposites. `.get_message()` calls `.acquire()` on the `consumer_lock`. This is the call that will make the `consumer` wait until a message is ready.
- Once the `consumer` has acquired the `.consumer_lock`, it copies out the value in `.message` and then calls `.release()` on the `.producer_lock`. Releasing this lock is what allows the `producer` to insert the next message into the `pipeline`.
- Before you go on to `.set_message()`, there’s something subtle going on in `.get_message()` that’s pretty easy to miss. It might seem tempting to get rid of `message` and just have the function end with `return self.message`. See if you can figure out why you don’t want to do that before moving on.
- Here’s the answer. As soon as the `consumer` calls `.producer_lock.release()`, it can be swapped out, and the `producer` can start running. That could happen before `.release()` returns! This means that there is a slight possibility that when the function returns `self.message`, that could actually be the _next_ message generated, so you would lose the first message. This is another example of a __race condition__.
- Moving on to `.set_message()`, you can see the opposite side of the transaction. The `producer` will call this with a message. It will acquire the `.producer_lock`, set the `.message`, and the call `.release()` on then `consumer_lock`, which will allow the `consumer` to read that value.
- Let’s run the code that has logging set to `WARNING` and see what it looks like:
```bash
$ ./prodcom_lock.py
Producer got data 43
Producer got data 45
Consumer storing data: 43
Producer got data 86
Consumer storing data: 45
Producer got data 40
Consumer storing data: 86
Producer got data 62
Consumer storing data: 40
Producer got data 15
Consumer storing data: 62
Producer got data 16
Consumer storing data: 15
Producer got data 61
Consumer storing data: 16
Producer got data 73
Consumer storing data: 61
Producer got data 22
Consumer storing data: 73
Consumer storing data: 22
```
- At first, you might find it odd that the producer gets two messages before the consumer even runs. If you look back at the `producer` and `.set_message()`, you will notice that the only place it will wait for a `Lock` is when it attempts to put the message into the pipeline. This is done after the `producer` gets the message and logs that it has it.
- When the `producer` attempts to send this second message, it will call `.set_message()` the second time and it will block.
- The operating system can swap threads at any time, but it generally lets each thread have a reasonable amount of time to run before swapping it out. That’s why the `producer` usually runs until it blocks in the second call to `.set_message()`.
- Once a thread is blocked, however, the operating system will always swap it out and find a different thread to run. In this case, the only other thread with anything to do is the `consumer`.
- The `consumer` calls `.get_message()`, which reads the message and calls `.release()` on the `.producer_lock`, thus allowing the `producer` to run again the next time threads are swapped.
- Notice that the first message was `43`, and that is exactly what the `consumer` read, even though the `producer` had already generated the `45` message.
- While it works for this limited test, it is not a great solution to the producer-consumer problem in general because it only allows a single value in the pipeline at a time. When the `producer` gets a burst of messages, it will have nowhere to put them.
- Let’s move on to a better way to solve this problem, using a `Queue`.
## Producer-Consumer Using Queue
- If you want to be able to handle more than one value in the pipeline at a time, you’ll need a data structure for the pipeline that allows the number to grow and shrink as data backs up from the `producer`.
- Python’s standard library has a `queue` module which, in turn, has a `Queue` class. Let’s change the `Pipeline` to use a `Queue` instead of just a variable protected by a `Lock`. You’ll also use a different way to stop the worker threads by using a different primitive from Python `threading`, an `Event`.
- Let’s start with the `Event`. The `threading.Event` object allows one thread to signal an `event` while many other threads can be waiting for that `event` to happen. The key usage in this code is that the threads that are waiting for the event do not necessarily need to stop what they are doing, they can just check the status of the `Event` every once in a while.
- The triggering of the event can be many things. In this example, the main thread will simply sleep for a while and then `.set()` it:
```python
if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")
    # logging.getLogger().setLevel(logging.DEBUG)

    pipeline = Pipeline()
    event = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(producer, pipeline, event)
        executor.submit(consumer, pipeline, event)

        time.sleep(0.1)
        logging.info("Main: about to set event")
        event.set()
```
- The only changes here are the creation of the `event` object on line 8, passing the `event` as a parameter on lines 10 and 11, and the final section on lines 13 to 15, which sleep for a second, log a message, and then call `.set()` on the event.
- The `producer` also did not have to change too much:
```python
def producer(pipeline, event):
    """Pretend we're getting a number from the network."""
    while not event.is_set():
        message = random.randint(1, 101)
        logging.info("Producer got message: %s", message)
        pipeline.set_message(message, "Producer")

    logging.info("Producer received EXIT event. Exiting")
```
- It now will loop until it sees that the event was set on line 3. It also no longer puts the `SENTINEL` value into the `pipeline`.
- `consumer` had to change a little more:
```python
def consumer(pipeline, event):
    """Pretend we're saving a number in the database."""
    while not event.is_set() or not pipeline.empty():
        message = pipeline.get_message("Consumer")
        logging.info(
            "Consumer storing message: %s  (queue size=%s)",
            message,
            pipeline.qsize(),
        )

    logging.info("Consumer received EXIT event. Exiting")
```
- While you got to take out the code related to the `SENTINEL` value, you did have to do a slightly more complicated `while` condition. Not only does it loop until the `event` is set, but it also needs to keep looping until the `pipeline` has been emptied.
- Making sure the queue is empty before the consumer finishes prevents another fun issue. If the `consumer` does exit while the `pipeline` has messages in it, there are two bad things that can happen. The first is that you lose those final messages, but the more serious one is that the `producer` can get caught attempting to add a message to a full queue and never return.
- This happens if the `event` gets triggered after the `producer` has checked the `.is_set()` condition but before it calls `pipeline.set_message()`.
- If that happens, it’s possible for the consumer to wake up and exit with the queue still completely full. The `producer` will then call `.set_message()` which will wait until there is space on the queue for the new message. The `consumer` has already exited, so this will not happen and the `producer` will not exit.
- The `Pipeline` has changed dramatically, however:
```python
class Pipeline(queue.Queue):
    def __init__(self):
        super().__init__(maxsize=10)

    def get_message(self, name):
        logging.debug("%s:about to get from queue", name)
        value = self.get()
        logging.debug("%s:got %d from queue", name, value)
        return value

    def set_message(self, value, name):
        logging.debug("%s:about to add %d to queue", name, value)
        self.put(value)
        logging.debug("%s:added %d to queue", name, value)
```
- You can see that `Pipeline` is a subclass of `queue.Queue`. `Queue` has an optional parameter when initializing to specify a maximum size of the queue.
- If you give a positive number for `maxsize`, it will limit the queue to that number of elements, causing `.put()` to block until there are fewer than `maxsize` elements. If you don’t specify `maxsize`, then the queue will grow to the limits of your computer’s memory.
- `.get_message()` and `.set_message()` got much smaller. They basically wrap `.get()` and `.put()` on the `Queue`. You might be wondering where all of the locking code that prevents the threads from causing race conditions went.
- The core devs who wrote the standard library knew that a `Queue` is frequently used in multi-threading environments and incorporated all of that locking code inside the `Queue` itself. `Queue` is thread-safe.
- Running this program looks like the following:
```bash
$ ./prodcom_queue.py
Producer got message: 32
Producer got message: 51
Producer got message: 25
Producer got message: 94
Producer got message: 29
Consumer storing message: 32 (queue size=3)
Producer got message: 96
Consumer storing message: 51 (queue size=3)
Producer got message: 6
Consumer storing message: 25 (queue size=3)
Producer got message: 31

[many lines deleted]

Producer got message: 80
Consumer storing message: 94 (queue size=6)
Producer got message: 33
Consumer storing message: 20 (queue size=6)
Producer got message: 48
Consumer storing message: 31 (queue size=6)
Producer got message: 52
Consumer storing message: 98 (queue size=6)
Main: about to set event
Producer got message: 13
Consumer storing message: 59 (queue size=6)
Producer received EXIT event. Exiting
Consumer storing message: 75 (queue size=6)
Consumer storing message: 97 (queue size=5)
Consumer storing message: 80 (queue size=4)
Consumer storing message: 33 (queue size=3)
Consumer storing message: 48 (queue size=2)
Consumer storing message: 52 (queue size=1)
Consumer storing message: 13 (queue size=0)
Consumer received EXIT event. Exiting
```
- If you read through the output in my example, you can see some interesting things happening. Right at the top, you can see the `producer` got to create five messages and place four of them on the queue. It got swapped out by the operating system before it could place the fifth one.
- The `consumer` then ran and pulled off the first message. It printed out that message as well as how deep the queue was at that point
- This is how you know that the fifth message hasn’t made it into the `pipeline` yet. The queue is down to size three after a single message was removed. You also know that the `queue` can hold ten messages, so the `producer` thread didn’t get blocked by the `queue`. It was swapped out by the OS.
# Threading Objects
- There are a few more primitives offered by the Python `threading` module. While you didn’t need these for the examples above, they can come in handy in different use cases, so it’s good to be familiar with them.
## Semaphore
- The first Python `threading` object to look at is `threading.Semaphore`. A `Semaphore` is a counter with a few special properties. The first one is that the counting is atomic. This means that there is a guarantee that the operating system will not swap out the thread in the middle of incrementing or decrementing the counter.
- The internal counter is incremented when you call `.release()` and decremented when you call `.acquire()`.
- The next special property is that if a thread calls `.acquire()` when the counter is zero, that thread will block until a different thread calls `.release()` and increments the counter to one.
- Semaphores are frequently used to protect a resource that has a limited capacity. An example would be if you have a pool of connections and want to limit the size of that pool to a specific number.
##  Timer
