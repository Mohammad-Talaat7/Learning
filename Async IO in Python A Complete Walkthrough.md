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
# Miguel Grinberg's 2017 PyCon talk (Example on Async IO)
- Chess master Judit Polgár hosts a chess exhibition in which she plays multiple amateur players. She has two ways of conducting the exhibition: synchronously and asynchronously.
- Assumptions:
	- 24 opponents
	- Judit makes each chess move in 5 seconds
	- Opponents each take 55 seconds to make a move
	- Games average 30 pair-moves (60 moves total)
- **Synchronous version**: Judit plays one game at a time, never two at the same time, until the game is complete. Each game takes _(55 + 5) * 30 == 1800_ seconds, or 30 minutes. The entire exhibition takes _24 * 30 == 720_ minutes, or **12 hours**.
- **Asynchronous version**: Judit moves from table to table, making one move at each table. She leaves the table and lets the opponent make their next move during the wait time. One move on all 24 games takes Judit _24 * 5 == 120_ seconds, or 2 minutes. The entire exhibition is now cut down to _120 * 30 == 3600_ seconds, or just **1 hour**. [(Source)](https://youtu.be/iG6fr81xHKA?t=4m29s)
- There is only one Judit Polgár, who has only two hands and makes only one move at a time by herself. But playing asynchronously cuts the exhibition time down from 12 hours to one. So, cooperative multitasking is a fancy way of saying that a program’s event loop (more on that later) communicates with multiple tasks to let each take turns running at the optimal time.
- Async IO takes long waiting periods in which functions would otherwise be blocking and allows other functions to run during that downtime. (A function that blocks effectively forbids others from running from the time that it starts until the time that it returns.)
# The `asyncio` Package and `async/await`
- At the heart of async IO are coroutines. A coroutine is a specialized version of a Python generator function. Let’s start with a baseline definition and then build off of it as you progress here: a coroutine is a function that can suspend its execution before reaching `return`, and it can indirectly pass control to another coroutine for some time.
- Let’s take the immersive approach and write some async IO code. This short program is the `Hello World` of async IO but goes a long way towards illustrating its core functionality:
```python
#!/usr/bin/env python3
# countasync.py

import asyncio

async def count():
    print("One")
    await asyncio.sleep(1)
    print("Two")

async def main():
    await asyncio.gather(count(), count(), count())

if __name__ == "__main__":
    import time
    s = time.perf_counter()
    asyncio.run(main())
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")
```
- When you execute this file, take note of what looks different than if you were to define the functions with just `def` and `time.sleep()`:
```bash
$ python3 countasync.py
One
One
One
Two
Two
Two
countasync.py executed in 1.01 seconds.
```
- The order of this output is the heart of async IO. Talking to each of the calls to `count()` is a single event loop, or coordinator. When each task reaches `await asyncio.sleep(1)`, the function yells up to the event loop and gives control back to it, saying, “I’m going to be sleeping for 1 second. Go ahead and let something else meaningful be done in the meantime.”
- Contrast this to the synchronous version:
```python
#!/usr/bin/env python3
# countsync.py

import time

def count():
    print("One")
    time.sleep(1)
    print("Two")

def main():
    for _ in range(3):
        count()

if __name__ == "__main__":
    s = time.perf_counter()
    main()
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")
```
- When executed, there is a slight but critical change in order and execution time:
```bash
$ python3 countsync.py
One
Two
One
Two
One
Two
countsync.py executed in 3.01 seconds.
```
- While using `time.sleep()` and `asyncio.sleep()` may seem banal, they are used as stand-ins for any time-intensive processes that involve wait time. (The most mundane thing you can wait on is a `sleep()` call that does basically nothing.) That is, `time.sleep()` can represent any time-consuming blocking function call, while `asyncio.sleep()` is used to stand in for a non-blocking call (but one that also takes some time to complete).
- As you’ll see in the next section, the benefit of awaiting something, including `asyncio.sleep()`, is that the surrounding function can temporarily cede control to another function that’s more readily able to do something immediately. In contrast, `time.sleep()` or any other blocking call is incompatible with asynchronous Python code, because it will stop everything in its tracks for the duration of the sleep time.
## The Rules of Async IO
- The syntax `async def` introduces either a **native coroutine** or an **asynchronous generator**. The expressions `async with` and `async for` are also valid, and you’ll see them later on.
- The keyword `await` passes function control back to the event loop. (It suspends the execution of the surrounding coroutine.) If Python encounters an `await f()` expression in the scope of `g()`, this is how `await` tells the event loop, “Suspend execution of `g()` until whatever I’m waiting on—the result of `f()`—is returned. In the meantime, go let something else run.”
- In code, that second bullet point looks roughly like this:
```python
async def g():
    # Pause here and come back to g() when f() is ready
    r = await f()
    return r
```
- There’s also a strict set of rules around when and how you can and cannot use `async`/`await`. These can be handy whether you are still picking up the syntax or already have exposure to using `async`/`await`
- A function that you introduce with `async def` is a coroutine. It may use `await`, `return`, or `yield`, but all of these are optional. Declaring `async def noop(): pass` is valid:
	- Using `await` and/or `return` creates a coroutine function. To call a coroutine function, you must `await` it to get its results.
	- It is less common (and only recently legal in Python) to use `yield` in an `async def` block. This creates an asynchronous generator, which you iterate over with `async for`. Forget about async generators for the time being and focus on getting down the syntax for coroutine functions, which use `await` and/or `return`.
	- Anything defined with `async def` may not use `yield from`, which will raise a `SyntaxError`.
- Just like it’s a `SyntaxError` to use `yield` outside of a `def` function, it is a `SyntaxError` to use `await` outside of an `async def` coroutine. You can only use `await` in the body of coroutines.
- Here are some terse examples meant to summarize the above few rules:
```python
async def f(x):
    y = await z(x)  # OK - `await` and `return` allowed in coroutines
    return y

async def g(x):
    yield x  # OK - this is an async generator

async def m(x):
    yield from gen(x)  # No - SyntaxError

def m(x):
    y = await z(x)  # Still no - SyntaxError (no `async def` here)
    return y
```
- Finally, when you use `await f()`, it’s required that `f()` be an object that is awaitable. Well, that’s not very helpful, is it? For now, just know that an awaitable object is either:
	- another coroutine
	- an object defining an `.__await__()` dunder method that returns an iterator.
- If you’re writing a program, for the large majority of purposes, you should only need to worry about case #1.
# Async IO Design Patterns
- Async IO comes with its own set of possible script designs, which you’ll get introduced to in this section.
## Chaining Coroutines
- A key feature of coroutines is that they can be chained together. (Remember, a coroutine object is awaitable, so another coroutine can `await` it.) This allows you to break programs into smaller, manageable, recyclable coroutines:
```python
#!/usr/bin/env python3
# chained.py

import asyncio
import random
import time

async def part1(n: int) -> str:
    i = random.randint(0, 10)
    print(f"part1({n}) sleeping for {i} seconds.")
    await asyncio.sleep(i)
    result = f"result{n}-1"
    print(f"Returning part1({n}) == {result}.")
    return result

async def part2(n: int, arg: str) -> str:
    i = random.randint(0, 10)
    print(f"part2{n, arg} sleeping for {i} seconds.")
    await asyncio.sleep(i)
    result = f"result{n}-2 derived from {arg}"
    print(f"Returning part2{n, arg} == {result}.")
    return result

async def chain(n: int) -> None:
    start = time.perf_counter()
    p1 = await part1(n)
    p2 = await part2(n, p1)
    end = time.perf_counter() - start
    print(f"-->Chained result{n} => {p2} (took {end:0.2f} seconds).")

async def main(*args):
    await asyncio.gather(*(chain(n) for n in args))

if __name__ == "__main__":
    import sys
    random.seed(444)
    args = [1, 2, 3] if len(sys.argv) == 1 else map(int, sys.argv[1:])
    start = time.perf_counter()
    asyncio.run(main(*args))
    end = time.perf_counter() - start
    print(f"Program finished in {end:0.2f} seconds.")
```
- Pay careful attention to the output, where `part1()` sleeps for a variable amount of time, and `part2()` begins working with the results as they become available:
```bash
$ python3 chained.py 9 6 3
part1(9) sleeping for 4 seconds.
part1(6) sleeping for 4 seconds.
part1(3) sleeping for 0 seconds.
Returning part1(3) == result3-1.
part2(3, 'result3-1') sleeping for 4 seconds.
Returning part1(9) == result9-1.
part2(9, 'result9-1') sleeping for 7 seconds.
Returning part1(6) == result6-1.
part2(6, 'result6-1') sleeping for 4 seconds.
Returning part2(3, 'result3-1') == result3-2 derived from result3-1.
-->Chained result3 => result3-2 derived from result3-1 (took 4.00 seconds).
Returning part2(6, 'result6-1') == result6-2 derived from result6-1.
-->Chained result6 => result6-2 derived from result6-1 (took 8.01 seconds).
Returning part2(9, 'result9-1') == result9-2 derived from result9-1.
-->Chained result9 => result9-2 derived from result9-1 (took 11.01 seconds).
Program finished in 11.01 seconds.
```
- In this setup, the runtime of `main()` will be equal to the maximum runtime of the tasks that it gathers together and schedules.
## A Full Program: Asynchronous Requests
- You’ve made it this far, and now it’s time for the fun and painless part. In this section, you’ll build a web-scraping URL collector, `areq.py`, using `aiohttp`, a blazingly fast async HTTP client/server framework. (We just need the client part.) Such a tool could be used to map connections between a cluster of sites, with the links forming a directed graph.
- You may be wondering why Python’s `requests` package isn’t compatible with async IO. `requests` is built on top of `urllib3`, which in turn uses Python’s `http` and `socket` modules.
- By default, socket operations are blocking. This means that Python won’t like `await requests.get(url)` because `.get()` is not awaitable. In contrast, almost everything in `aiohttp` is an awaitable coroutine, such as [`session.request()`](https://github.com/aio-libs/aiohttp/blob/508adbb656da2e9ae660da5e98e1e5fa6669a3f4/aiohttp/client.py#L225) and [`response.text()`](https://github.com/aio-libs/aiohttp/blob/da75122f6089a250128d2736f2bd88d10e97ca17/aiohttp/client_reqrep.py#L913). It’s a great package otherwise, but you’re doing yourself a disservice by using `requests` in asynchronous code.
- The high-level program structure will look like this:
	- Read a sequence of URLs from a local file, `urls.txt`.
	- Send GET requests for the URLs and decode the resulting content. If this fails, stop there for a URL.
	- Search for the URLs within `href` tags in the HTML of the responses.
	- Write the results to `foundurls.txt`.
	- Do all of the above as asynchronously and concurrently as possible. (Use `aiohttp` for the requests, and `aiofiles` for the file-appends. These are two primary examples of IO that are well-suited for the async IO model.)
- Here are the contents of `urls.txt`. It’s not huge, and contains mostly highly trafficked sites:
```bash
$ cat urls.txt
https://regex101.com/
https://docs.python.org/3/this-url-will-404.html
https://www.nytimes.com/guides/
https://www.mediamatters.org/
https://1.1.1.1/
https://www.politico.com/tipsheets/morning-money
https://www.bloomberg.com/markets/economics
https://www.ietf.org/rfc/rfc2616.txt
```
- The second URL in the list should return a 404 response, which you’ll need to handle gracefully. If you’re running an expanded version of this program, you’ll probably need to deal with much hairier problems than this, such a server disconnections and endless redirects.
- The requests themselves should be made using a single session, to take advantage of reusage of the session’s internal connection pool.
- Let’s take a look at the full program. We’ll walk through things step-by-step after:
```python
#!/usr/bin/env python3
# areq.py

"""Asynchronously get links embedded in multiple pages' HMTL."""

import asyncio
import logging
import re
import sys
from typing import IO
import urllib.error
import urllib.parse

import aiofiles
import aiohttp
from aiohttp import ClientSession

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    level=logging.DEBUG,
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("areq")
logging.getLogger("chardet.charsetprober").disabled = True

HREF_RE = re.compile(r'href="(.*?)"')

async def fetch_html(url: str, session: ClientSession, **kwargs) -> str:
    """GET request wrapper to fetch page HTML.

    kwargs are passed to `session.request()`.
    """

    resp = await session.request(method="GET", url=url, **kwargs)
    resp.raise_for_status()
    logger.info("Got response [%s] for URL: %s", resp.status, url)
    html = await resp.text()
    return html

async def parse(url: str, session: ClientSession, **kwargs) -> set:
    """Find HREFs in the HTML of `url`."""
    found = set()
    try:
        html = await fetch_html(url=url, session=session, **kwargs)
    except (
        aiohttp.ClientError,
        aiohttp.http_exceptions.HttpProcessingError,
    ) as e:
        logger.error(
            "aiohttp exception for %s [%s]: %s",
            url,
            getattr(e, "status", None),
            getattr(e, "message", None),
        )
        return found
    except Exception as e:
        logger.exception(
            "Non-aiohttp exception occured:  %s", getattr(e, "__dict__", {})
        )
        return found
    else:
        for link in HREF_RE.findall(html):
            try:
                abslink = urllib.parse.urljoin(url, link)
            except (urllib.error.URLError, ValueError):
                logger.exception("Error parsing URL: %s", link)
                pass
            else:
                found.add(abslink)
        logger.info("Found %d links for %s", len(found), url)
        return found

async def write_one(file: IO, url: str, **kwargs) -> None:
    """Write the found HREFs from `url` to `file`."""
    res = await parse(url=url, **kwargs)
    if not res:
        return None
    async with aiofiles.open(file, "a") as f:
        for p in res:
            await f.write(f"{url}\t{p}\n")
        logger.info("Wrote results for source URL: %s", url)

async def bulk_crawl_and_write(file: IO, urls: set, **kwargs) -> None:
    """Crawl & write concurrently to `file` for multiple `urls`."""
    async with ClientSession() as session:
        tasks = []
        for url in urls:
            tasks.append(
                write_one(file=file, url=url, session=session, **kwargs)
            )
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    import pathlib
    import sys

    assert sys.version_info >= (3, 7), "Script requires Python 3.7+."
    here = pathlib.Path(__file__).parent

    with open(here.joinpath("urls.txt")) as infile:
        urls = set(map(str.strip, infile))

    outpath = here.joinpath("foundurls.txt")
    with open(outpath, "w") as outfile:
        outfile.write("source_url\tparsed_url\n")

    asyncio.run(bulk_crawl_and_write(file=outpath, urls=urls))
```
- The constant `HREF_RE` is a [regular expression](https://realpython.com/regex-python/) to extract what we’re ultimately searching for, `href` tags within HTML:
```python
>>> HREF_RE.search('Go to <a href="https://realpython.com/">Real Python</a>')
<re.Match object; span=(15, 45), match='href="https://realpython.com/"'>
```
- The coroutine `fetch_html()` is a wrapper around a GET request to make the request and decode the resulting page HTML. It makes the request, awaits the response, and raises right away in the case of a non-200 status:
```python
resp = await session.request(method="GET", url=url, **kwargs)
resp.raise_for_status()
```
- If the status is okay, `fetch_html()` returns the page HTML (a `str`). Notably, there is no exception handling done in this function. The logic is to propagate that exception to the caller and let it be handled there:
```python
html = await resp.text()
```
- We `await` `session.request()` and `resp.text()` because they’re awaitable coroutines. The request/response cycle would otherwise be the long-tailed, time-hogging portion of the application, but with async IO, `fetch_html()` lets the event loop work on other readily available jobs such as parsing and writing URLs that have already been fetched.
- Next in the chain of coroutines comes `parse()`, which waits on `fetch_html()` for a given URL, and then extracts all of the `href` tags from that page’s HTML, making sure that each is valid and formatting it as an absolute path.
- Admittedly, the second portion of `parse()` is blocking, but it consists of a quick regex match and ensuring that the links discovered are made into absolute paths.
- In this specific case, this synchronous code should be quick and inconspicuous. But just remember that any line within a given coroutine will block other coroutines unless that line uses `yield`, `await`, or `return`. If the parsing was a more intensive process, you might want to consider running this portion in its own process with `loop.run_in_executor()`.
- Next, the coroutine `write()` takes a file object and a single URL, and waits on `parse()` to return a `set` of the parsed URLs, writing each to the file asynchronously along with its source URL through use of `aiofiles`, a package for async file IO.
- Lastly, `bulk_crawl_and_write()` serves as the main entry point into the script’s chain of coroutines. It uses a single session, and a task is created for each URL that is ultimately read from `urls.txt`.
- Here are a few additional points that deserve mention:
	- The default `ClientSession` has an adapter with a maximum of 100 open connections. To change that, pass an instance of `asyncio.connector.TCPConnector` to `ClientSession`. You can also specify limits on a per-host basis.
	- You can specify max timeouts for both the session as a whole and for individual requests.
	- This script also uses `async with`, which works with an asynchronous context manager. I haven’t devoted a whole section to this concept because the transition from synchronous to asynchronous context managers is fairly straightforward. The latter has to define `.__aenter__()` and `.__aexit__()` rather than `.__exit__()` and `.__enter__()`. As you might expect, `async with` can only be used inside a coroutine function declared with `async def`.
- If you’d like to explore a bit more, the [companion files](https://github.com/realpython/materials/tree/master/asyncio-walkthrough) for this tutorial up at GitHub have comments and docstrings attached as well.
- Here’s the execution in all of its glory, as `areq.py` gets, parses, and saves results for 9 URLs in under a second:
```bash
$ python3 areq.py
21:33:22 DEBUG:asyncio: Using selector: KqueueSelector
21:33:22 INFO:areq: Got response [200] for URL: https://www.mediamatters.org/
21:33:22 INFO:areq: Found 115 links for https://www.mediamatters.org/
21:33:22 INFO:areq: Got response [200] for URL: https://www.nytimes.com/guides/
21:33:22 INFO:areq: Got response [200] for URL: https://www.politico.com/tipsheets/morning-money
21:33:22 INFO:areq: Got response [200] for URL: https://www.ietf.org/rfc/rfc2616.txt
21:33:22 ERROR:areq: aiohttp exception for https://docs.python.org/3/this-url-will-404.html [404]: Not Found
21:33:22 INFO:areq: Found 120 links for https://www.nytimes.com/guides/
21:33:22 INFO:areq: Found 143 links for https://www.politico.com/tipsheets/morning-money
21:33:22 INFO:areq: Wrote results for source URL: https://www.mediamatters.org/
21:33:22 INFO:areq: Found 0 links for https://www.ietf.org/rfc/rfc2616.txt
21:33:22 INFO:areq: Got response [200] for URL: https://1.1.1.1/
21:33:22 INFO:areq: Wrote results for source URL: https://www.nytimes.com/guides/
21:33:22 INFO:areq: Wrote results for source URL: https://www.politico.com/tipsheets/morning-money
21:33:22 INFO:areq: Got response [200] for URL: https://www.bloomberg.com/markets/economics
21:33:22 INFO:areq: Found 3 links for https://www.bloomberg.com/markets/economics
21:33:22 INFO:areq: Wrote results for source URL: https://www.bloomberg.com/markets/economics
21:33:23 INFO:areq: Found 36 links for https://1.1.1.1/
21:33:23 INFO:areq: Got response [200] for URL: https://regex101.com/
21:33:23 INFO:areq: Found 23 links for https://regex101.com/
21:33:23 INFO:areq: Wrote results for source URL: https://regex101.com/
21:33:23 INFO:areq: Wrote results for source URL: https://1.1.1.1/
```
