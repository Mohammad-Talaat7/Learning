---
Creation: 2024-07-30
tags:
  - Articles
  - Parallel
Reference: https://realpython.com/async-io-python/
Type:
  - Article
---
# Where does Async IO Fit In?
- Concurrency and parallelism are expansive subjects that are not easy to wade into. While this article focuses on async IO and its implementation in Python, it’s worth taking a minute to compare async IO to its counterparts in order to have context about how async IO fits into the larger, sometimes dizzying puzzle.
- To recap, concurrency encompasses both multiprocessing (ideal for CPU-bound tasks) and threading (suited for IO-bound tasks). Multiprocessing is a form of parallelism, with parallelism being a specific type (subset) of concurrency. The Python standard library has offered longstanding support for both of these through its `multiprocessing`, `threading`, and `concurrent.futures` packages.
- The `asyncio` package is billed by the Python documentation as a library to write concurrent code. However, async IO is not threading, nor is it multiprocessing. It is not built on top of either of these.
- In fact, async IO is a single-threaded, single-process design: it uses **cooperative multitasking**, a term that you’ll flesh out by the end of this tutorial. It has been said in other words that async IO gives a feeling of concurrency despite using a single thread in a single process. Coroutines (a central feature of async IO) can be scheduled concurrently, but they are not inherently concurrent.
- I’d like to paraphrase one from Miguel Grinberg’s 2017 [PyCon](https://realpython.com/pycon-guide/) talk, which explains everything quite beautifully:


> [!example] Miguel Grinberg's 2017 PyCon Talk
>  Chess master Judit Polgár hosts a chess exhibition in which she plays multiple amateur players. She has two ways of conducting the exhibition: synchronously and asynchronously.
> 
> Assumptions:
> 
> - 24 opponents
> - Judit makes each chess move in 5 seconds
> - Opponents each take 55 seconds to make a move
> - Games average 30 pair-moves (60 moves total)
> 
> **Synchronous version**: Judit plays one game at a time, never two at the same time, until the game is complete. Each game takes _(55 + 5) * 30 == 1800_ seconds, or 30 minutes. The entire exhibition takes _24 * 30 == 720_ minutes, or **12 hours**.
> 
> **Asynchronous version**: Judit moves from table to table, making one move at each table. She leaves the table and lets the opponent make their next move during the wait time. One move on all 24 games takes Judit _24 * 5 == 120_ seconds, or 2 minutes. The entire exhibition is now cut down to _120 * 30 == 3600_ seconds, or just **1 hour**. [(Source)](https://youtu.be/iG6fr81xHKA?t=4m29s)
> 
> There is only one Judit Polgár, who has only two hands and makes only one move at a time by herself. But playing asynchronously cuts the exhibition time down from 12 hours to one. So, cooperative multitasking is a fancy way of saying that a program’s event loop (more on that later) communicates with multiple tasks to let each take turns running at the optimal time.

- Async IO takes long waiting periods in which functions would otherwise be blocking and allows other functions to run during that downtime. (A function that blocks effectively forbids others from running from the time that it starts until the time that it returns.)
