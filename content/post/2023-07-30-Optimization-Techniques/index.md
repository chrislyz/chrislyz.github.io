---
title: Optimization Techniques
description: Continuous optimization method collections
date: 2023-07-30
---



# Optimization Techniques

- [Branch Prediction](#branch-prediction)



## Branch Prediction

[Why is processing a sorted array faster than processing an unsorted array?](https://stackoverflow.com/a/11227902)



### Example: Random Access vs. Ordered Access

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ARRAY_SIZE 100000000

int main() {
    static int data[ARRAY_SIZE];

    for (int i = 0; i < ARRAY_SIZE; ++i) {
        data[i] = rand() % 256;
    }

    long long sums = 0;
    clock_t tic = clock();
    // Snippet 1: random access
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        if (data[i] > 128) {
            sums += data[i];
        }
    }
    clock_t toc = clock();
    printf("Random access took %lf seconds\n", (double)(toc - tic)/CLOCKS_PER_SEC);

    for (int i = 0; i < ARRAY_SIZE; ++i) {
        data[i] = i;
    }
    sums = 0;
    tic = clock();
    // Snippet 2: ordered access
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        if (data[i] > 128) {
            sums += data[i];
        }
    }
    toc = clock();
    printf("Ordered access took %lf seconds\n", (double)(toc - tic)/CLOCKS_PER_SEC);
}

// Random access took 0.503551 seconds
// Ordered access took 0.138453 seconds
```

### Branchless Solutions

> [!warning]
>
> Branchless codes are usually a hacking way of removing conditional statements, which may reduce readability and flexibility.

### Simple if-else clause

Consider you have a simple `if-else` clause, e.g.,

```c
if (someCond) a = 0;
else a = 1;
```

We prefer,

```c
a = someCond ? a : b;
```

which involves less `jump` operations.

> [!info]
>
> if-else statements are compiled to
> ![[Pasted image 20230731110906.png]]
> while the ternary operation is compiled to
> ![[Pasted image 20230731110951.png]]
> https://godbolt.org/z/oEn3Evrve

### Conditional Assignment

Consider a scenario where you have a simple conditional assignment, e.g.,

```c
if (val >= 128) {
    sums += val;
}
```

One possible way is to use bitwise operation and mathematics to remove conditional statements, e.g.,

```c
int t = (val - 128) >> 31;
sums += ~t & val;
```

### Switch-case

Consider a scenario where you need to execute different workflows according to some simple options, e.g.,

```c
// applies the same to if-statement
switch (option) {
    case 0:
        workflow1();
        break;
    case 1:
        workflow2();
        break;
    ...
    default:
        baseflow();
        break;
}
```

One possible way to deal with these jumps is to use a jump table, e.g.,

```c
// function prototypes
void workflow1(void *);
void workflow2(void *);
...
void baseflow(void *);

void (*table[n]) (void *);

// snippet
table[0] = workflow1();
table[1] = workflow2();
...
table[n-1] = baseflow();
```