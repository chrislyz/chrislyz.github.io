---
title: Python Decorators
date: 2023-09-07
math: true
---

## Function/Method Decorators

### Background

Prior to python 2.4, there were only several methods used for transforming functions and methods (e.g., `staticmethod()` and `classmethod()`). There is no syntax for such intensions.
The drawbacks are listed as following,

1. Difficulties to understand (the actual transformation could be far from declaration)

   ```python
   def foo(self):
       pass
   op1()
   op2()
   opN()
   foo = classmethod(foo)
   ```

2. Less readable with longer methods 

   ```python
   my_new_function_with_very_long_name = 
           classmethod(my_new_function_with_very_long_name)
   ```

3. Less pythonic to name the function multiple times for conceptually a single declaration

   ```python
   def foo(self):
       pass
   foo = synchronized(lock)(foo)
   foo = classmethod(foo)
   ```

To address above drawbacks, the transformation syntax is hence ...

```python
@classmethod
@synchronized(lock)
def foo(cls):
    pass
```

> [!INFO]
>
> There are two ways of modifying classes, class decorators and metaclasses. Class decorators were proposed in Python 2.6 by [PEP 3129](https://peps.python.org/pep-3129).

## The `@syntax`

```python
def decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)  # return results to func
    return wrapper
```

The `@syntax `

```python
@dec2
@dec1
def func(arg1, arg2, ...):
    pass
fun(arg1, arg2)
```

is equivalent to

```python
def func(arg1, arg2, ...):
    pass
func = dec2(dec1(func))
```

The bottom to top ordering is clearly follows the ordering that the function is transformed from inside all the way to outside (i.e., running `func` then `dec1(func)` and etc.)

> [!INFO]
>
> There was a pierid of discussing different forms of the syntax of decorators. Each of the proposal had its reason. The community summarized pros and cons and made the final decision.
>
> On the one hand, intuitively, a decorator ought to stay outside the function/method body given the fact that they are executed at the time the function is defined.
>
> Refer to other languages, Java uses `@` as markers and annotations.
>
> If you are keen on a closer look on each form, you can read the [syntax-alternatives](https://peps.python.org/pep-0318/#syntax-alternatives) section listed in the standard.

#### Some terminology

- **Transformation function**: a function that is used to apply a series of transformations to other function, denoted by $Y = f(X)$, where $f$ is the transformation function.
- **Target function**: a function that accepts transformations, denoted by $X$.
- **Decorator syntax**: a python syntax denoted by `@`.

### Confusion

You may ask what decorator does? Why do all the examples of the decorator have a inner function *wrapper*. Consider an example,

```python
def decorator(func):
    def wrapper(*args, **kwargs):
        print('Entering Decorator')
        res = func(*args, **kwargs)
        return res
    return wrapper

# What a decorator really does is assignment
# assigning wrapper (returned by decorator) to target function
@decorator
def foo(arg):
    print('Welcome to foo()')
# is essentially equivalent to take transformation
foo = decorator(foo)
# then interpreted to
foo(arg) ::= wrapper(arg)
         ::= res = func(arg)  # func is provided in decorator scope
           | return res
```

Therefore, we now know, decorator is just a function that accepts functions as argument and returns the target (i.e., to be decorated) function with a bit manipulation. 
Then can we have a decorator that does not have an inner wrapper function? What is the difference between two of them?

```python
# Consider following example
def decorator_wht_wrapper(func):
    print('Decorator without wrapper')
    return func

@decorator_wht_wrapper
def foo():  # output: Decorator without wrapper
    print('Welcome to foo()')

# Running this script will automatically produce one-line output
# 'Decorator without wrapper'
# The reason is that decorate a function (i.e., @decorator) is equivalent to
foo = decorator_wht_wrapper(foo)
# It can be seen that `decorator_wht_wrapper` is executed anyway
# hence, it is obvious an one-line output will be poped up

# Now, what will be the output of following lines
foo()  # output: Welcome to foo()
foo()  # output: Welcome to foo()
foo()  # output: Welcome to foo()

# While the decorate with wrapper result differently
foo()  # output: Entering Decorator\nWelcome to foo()
foo()  # output: Entering Decorator\nWelcome to foo()
foo()  # output: Entering Decorator\nWelcome to foo()
```

To summarize, you can decorate a function without wrapper. However, you need to be aware of the fact that a decorator is by definition...

### Nesting decorators

### Decorators with argument

### Conclusion

In conclusion, a decorator requires two important component, including a transformation function (i.e., $f$) and a decorator syntax (i.e., @). By decorating a function, it always means that you agree with executing the transformation function with the target function as an argument (i.e., execute `X = f(X)`).

### Metadata

Decorators can be as simple as replacing one function with another function. Unfortunately, the wrapper functions don't update the [[Builtin Variable|metadata]] of the returned function automatically, during the replacement. For instance,

```python
def decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@decorator
def foo():
    pass

# Instead of outputing name of foo(), it outputs name of wrapper
print(foo.__name__)  # output: wrapper
print(*[getattr(foo, x) for x in dir(foo)])
```

To address this problem, [[Functools#^3306f0|@functools.wraps]] decorator updates the metadata of the original function and reflects it to the wrapper function.

```python
import functools
def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@decorator
def foo():
    pass

# Instead of outputing name of foo(), it outputs name of wrapper
print(foo.__name__)  # output: foo
print(*[getattr(foo, x) for x in dir(foo)])  # You can also see the changes
                                             # reflected on other field
```

### Class as decorators

Any object with `__call__` method defined is considered to be a callable object. A callable class can work the same way as functions.

```python
class Decorator:
    def __init__(self, func):
        self.func = func
        self.n_calls = 0
    # wrapper function as to a class object
    def __call__(self, *args, **kwargs):
        self.n_calls += 1
        print(f'{self.func.__name__} exectued {self.n_calls} times')
        return self.func(*args, **kwargs)

@Decorator
def foo():
    print('Hello')

foo()  # output: Hello\nfoo executed 1 times
foo()  # output: Hello\nfoo exectued 2 times
foo()  # output: Hello\nfoo executed 3 times

# output: AttributeError: 'A' object has no attribute '__name__'
#print(foo.__name__)
```

Everything seems to be fine, as `self.func` is directly copied from `func` itself. Then you may wonder there couldn't be anything left. But you are being cautious by double-checking `dir(foo)`. Alright, there must be some metadata information missing.

> At least three attributes - `__annotations__`, `__qualname__` and `__name__` - are missing.

In this case, you need to update the metadata from the wrapped function to the wrapper function. In terms of the wrapped and wrapper function, as we consider the whole class is a decorator function, the wrapper function is the class itself while the function parameter is the wrapped function.

```python
class Decorator:
    def __init__(self, func):
        self.func = func
        self.n_calls = 0
        functools.update_wrapper(self, func)
    ...

print(foo.__name__)  # output: foo
```

## Examples

### Timing function

```python
from functools import wraps
import time

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        duration = (time.perf_counter() - start) * 1000
        print(f"Function[{func.__name__!r}]: takes {duration:.2f}ms")
        return res
    return wrapper

@timeit
def somefunc(...):
    pass
```

### Singleton instance

```python
def singleton(cls):
    instances = {}
    def getinstance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return getinstance

@singleton
class MyClass:
    ...
```

### Filter input (decorator with arguments)

```python
from functools import wraps
def accepts(*types):
    def check_accepts(f):
        # co_argcount doese not support varargs
        assert len(types) == f.__code__.co_argcount
        @wraps
        def new_f(*args, **kwargs):
            for (a, t) in zip(args, types):
                assert isinstance(a, t), \
                        "arg %r does not match %s" % (a,t)
            return f(*args, **kwargs)
        return new_f
    return check_accepts

@accepts(int, (int, float))
def func(arg1, arg2):
    return arg1 * args2
```

### Gathering information

```python
from functools import wraps

def func_info(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.n_calls += 1
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__!r}({signature}) {wrapper.n_calls} times")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")
        return value

    wrapper.n_calls = 0  # definition of function attribute
                         # needs to be in its scope
    return wrapper

sum = func_info(sum)
a = [i for i in range(10)]
for i, v in enumerate(a):
    sum(a[:i+1])


# caching intermediate result
def cache_inter(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = ','.joint([repr(x) for x in args])
        if not wrapper.seen.get(key):
            wrapper.seen[key] = func(*args, **kwargs)
        return wrapper.seen[key]
    wrapper.seen = {}
    return wrapper

@func_info
@cache_inter
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1 or n == 2:
        return 1
    return fibonacci(n - 2) + fibonacci(n - 1)
# By comment and uncomment @cache_inter, you will be able to see
# a tremendous drop on the number of invocations
```

> Further reading about [[PEP 232 - Function Attributes]]

### Register an entry

```python
TABLE = dict()

def register(func):
    TABLE[func.__name__] = func
    return func

@register  # register happens only once when the function is defined
def add():
    ...

@register
def minus():
    ...

TABLE['add'](...)
TABLE['minus'](...)
```


## Class Decorators

The semantics and design goals of class decorators are the same as for function decorators; the only difference is that class decorators are decorating a class instead of a function.

```python
class A:
    pass
A = foo(bar(A))
# is semantically equivalent to
@foo
@bar
class A:
    pass
```

## Reference

-  [1] *Decorators for Functions and Methods*, PEP 318, June 2003.  [Available](https://peps.python.org/pep-0318/).
-  [2] *Class Decorators*, PEP 3128, May 2007. [Available](https://peps.python.org/pep-3129/).
-  [3] *Function Attributes*, PEP 232, December 2000. [Available](https://peps.python.org/pep-0232/). %%FIXME: This should not appear here%%