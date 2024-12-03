---
title: Python timeit Module
date: 2023-09-01
---



# [Module] timeit

## [class] Timer

Class for measuring execution time of a **small piece of code**.

```python
timeit.Timer(stmt='pass', setup='pass', timer=<default_timer>, globals=None)
```

***stmt*** contains the code snippets to be timed. If some extra preparation needs to be done, ***setup*** holds the work should be done before executing ***stmt***.

> The code snippet defined in ***setup*** is executed exactly once. And any modification holds among ***number*** of executing ***stmt***.[^bignote1]

[^bignote1]: Examples:

    ```python
    stmt = """
    print(len(queue))
    while queue:
         queue.popleft()
    """
    setup = """
    import collections
    queue =  collections.deque(range(1000000))
    """
    # print(len(queue)) outputs 1000000 for the very and only first time
    # then it falls back to 0 because of popleft()
    timeit.timeit(stmt, setup, number=10)
    ```

> The code snippets for either ***stmt*** or ***setup*** could be multiple statements as long as they are semi-colon[^1] or newline separated. But they cannot contain multi-line string literals as if ***stmt*** and ***setup*** could also be multi-line string literals.
> The execution time for ***setup*** is excluded from the result.
>
> The ***timer*** defaults to `time.perf_counter()`.

[^1]: Although semi-colon can be used to separate statements, it is limited to code snippets with same indentations (or simple statements that are comprised within a single logical line). As to those conditional snippets, newline is more of a comprehensive and legitimate separator.

The statement are executed within timeit namespace by default. However, statements in global namespace can also be imported and/or included into timeit namespace without further redo. There currently are two approaches to do so:

1. Group such statements in one function and form a namespace. Then import resulting function into ***setup*** in the form of `from __main__ import function()`
2. Directly include global namespace into ***globals*** in the form of `globals=globals()`

### [Method] timeit

```python
Timer().timeit(number=1000000)
```

Time given `stmt` with corresponding `setup`, `timer function` in total `number` executions. The `global` is optional used to specify a namespace. The measured time is returned in seconds as floating number.

> Note: `timeit()` temporarily disables garbage collection during timing. GC can then be enabled by `gc.enable()`

### [Function] timeit

```python
timeit.timeit(stmt='pass', setup='pass', timer=<default_timer>, number=1000000, globals=None)
```

An interface that creates a ***Timer*** instance with given arguments and run its `timeit()` method.

### [Method] repeat

```python
timeit.repeat(stmt='pass', setup='pass', timer=<default_timer>, repeat=5, number=1000000, globals=None)
```

It can be seen as sort of a wrapper function. Call `timeit()` in ***repeat*** times returning a list of results. Before preforming any statistic analysis, it is important to be noted that, *"the lowest value gives a lower bound of execution speed, while higher values could be caused by other processes interfering (i.e., OS context switching and etc.)"*.

### [Function] repeat

An interface that creates a ***Timer*** instance with given arguments and run its `repeat()` method.

```python
timeit.repeat(stmt='pass', setup='pass', timer=<default_timer>, repeat=5, number=1000000, globals=None)
```

### [Method] autorange

```python
autorange(callback=None)
```

It can be seen as sort of a wrapper function. Automatically determine how many times to call `timeit()`. It aims to help `timeit()` return a readable result. Therefore, `autorange()` keeps increasing ***number*** argument (i.e., number=1/2/5/10/20 and etc.) so that time is $\ge$ 0.2 seconds essentially.
The callback function receives two arguments, `callback(number, time_taken)`, and is called after each trial.

### [Method] print_exc

Helper to print a traceback from the timed code.

```python
print_exc(file=None)
```

 Compared to standard traceback, `print_exc()` source lines are also displayed to the I/O. The optional ***file*** argument defaults to `sys.stderr`.


## [CLI] timeit

```shell
python3 -m timeit [-n N] [-r N] [-u U] [-s S] [-h] [statement ...]
```

| Arguments            | Description                                       |
| -------------------- | ------------------------------------------------- |
| **-n** N, --number=N | how many times to execute statement               |
| **-r** N, --repeat=N | how many times to repeat the timer                |
| **-s** S, --setup=S  | statement to be executed statement                |
| **-p**, --process    | measure process time using `time.process_timer()` |
| **-u**, --unit=u     | time unit: nsec, usec, msec, or sec               |
| **-v**, --verbose    | print raw timing results                          |
| **-h**, --help       | help information                                  |


## [IPython] %timeit

Time of executing a Python statement or expression using the [[#[Module ] timeit|timeit]] module.

### Line mode

Time a single-line statement.

```python
%timeit [-n <N> -r<R> [-t|-c] -q -p<P> -o] statement
```

***-n & -r*** are the same as defined in CLI section.

- **-t**: use `time.time` to measure wall time (default on Unix platform)
- **-c**: use `time.clock` to measure (default on Windows platform)
- **-p**: use a precision of \<P\> digits, default to 3
- **-q**: quiet
- **-o**: return a result can be stored in a variable

### Cell mode

Time multi-line statements. the first line is used as setup statements, and the rest of them are timed. They have access to any variables created during setup.

```python
%%timeit [-n<N> -r<R> [-t|-c] -q -p<P> -o]
```

## Examples

- A pythonic example

  ```python
  import timeit
  
  # Multi-line string literals
  stmt = """
  while queue:
      queue.popleft()"""
  
  setup = """
  import collections
  queue = collections.deque(range(1000000))
  """
  # 0.04570029999999997
  print(timeit.timeit(stmt=stmt, setup=setup, number=1000))
  
  # newlineseparated single-line string literals
  stmt = 'while queue2:\n  queue2.popleft()'
  setup = 'import collections\nqueue2=collections.deque(range(1000000))'
  # 0.043947400000000025
  print(timeit.timeit(stmt=stmt, setup=setup, number=1000))
  
  # semicolon separated single-line string literals without indentation
  
  stmt = 'queue2=collections.deque([queue.popleft() for _ in range(len(queue))])'
  setup = 'import collections ; queue=collections.deque(range(1000000)) ;'
  # simulating while statement in one-liner 
  # 0.08338979999999996
  print(timeit.timeit(stmt=stmt, setup=setup, number=1000))
  
  #stmt = 'while queue2: ;  queue2.popleft() ;'
  #                       ^
  # SyntaxError: invalid syntax
  #Moreover on footnote[^1]
  #setup = 'import collections ; queue2=collections.deque(range(1000000)) ;'
  #print(timeit.timeit(stmt=stmt, setup=setup, number=1000))
  
  # directly executing function with timing
  def foo(queue):
      while queue:
          queue.popleft()
  # 0.04536580000000001
  print(timeit.timeit(stmt='foo(queue)',
                      setup='from __main__ import foo ; import collections ; queue=collections.deque(range(1000000))',
                      number=1))
  ```

- A CLI example

  ```shell
  # Use ' ' as separator/delimiter for multi-line statements
  $ python3 -m timeit 'data=[1,2]' 'if data:' '  data.pop()' '  if data:' '    data.pop()'
  ```

- An IPython example

  ```python
  #-----------------------------------------------------------------
  # The following code snippets represent a single code cell iPython
  # Code cell 1
  In [1]: import time
     ...: %timeit -n1 time.sleep(2)
  #-----------------------------------------------------------------
  1.01 s ± 5.91 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
  ```


  #-----------------------------------------------------------------

  # The following code snippets represent a single code cell iPython

  # Code cell 2

  ```python
  In [2]: %%timeit
     ...: import time
     ...: for _ in range(100):
     ...:    time.sleep(0.01)  # sleep 1 second in total
  #-----------------------------------------------------------------
  1.52 s ± 4.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

  ```
## Reference
- [1] [IPython official documentation](https://ipython.readthedocs.io/en/stable/interactive/magics.html)
- [2] [Python official documentation](https://docs.python.org/3/library/timeit.html)
  ```