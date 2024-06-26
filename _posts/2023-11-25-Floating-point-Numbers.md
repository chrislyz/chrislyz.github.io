---
layout: post
title: "Single-precision Floating-point Numbers"
---

## Single-precision Floating-point

Single-precision floating-point format is also known as FP32 or float32, since it occupies 32 bits in memory.

> Prior Knowledge
> Scientific notation for decimal numbers is straightforward, where nonzero numbers are written as $m \times 10^n$.
>
> Similarly, binary numbers with base $2$ are written as $m\_b \times 2\_d^{n_d}$, where $m_b$ is the coefficient of binary base, $n_d$ is the exponential digits of decimal base. For example, $(0.0001\ 1011)\_2$ is written as $(1.1011)\_2 \times (2\_{10})^{4\_{10}}$ in scientific notation.

### Composition

FP32 is composed by
- Sign bit: 1 bit,
    - $\text{sign} = (-1)^{b_{31}}$
- Exponent: 8 bits, read right-to-left,
    - stored in bias form with offset $-127$ added to the original range $[0, 255]$, which gives $[-127,128]$,
    - $\text{exponent} = b_{23} \times 2^{23-23} + b_{24} \times 2^{24-23} + b_{i} \times 2^{i-23} + \cdots b_{30} \times 2^{7} - 127$, where $b_i \in [0,1]$ is the bit value of bit index $i$.
- Mantissa (Significand precision): 24 bits (23 bits actually stored), read left-to-right,
    - $\text{fraction} = 1 + b_{22} \times 2^{23-22} + b_{21} \times 2^{23-21} + b_i \times 2^{23-i} + \cdots + b_0 \times 2^{23}$, where $b_i \in [0, 1]$ is the bit value of bit index $i$.

> [!info]
> Fun fact about IEEE 754: [Link 1](https://qr.ae/pyZonh)
> For the exponent part, offset by 127 naturally centers the mean of zero, so that all bits zero just represent 0.
> For the fraction part, engineers found about that a leading 1 always appears in the scientific notation for nonzero exponents. As a result, they decided to save 1 bit for implicitly adding 1 to the fraction part and name this as **leading bit convention**.

![image]({{site.baseurl}}/assets/media/Pasted image 20230727115116.png)

FP32 is calculated in decimal by,
$$
\begin{align*}
\text{value} &= \underbrace{(-1)^{b_{31}}\vphantom{\sum_j}}_{sign} \times \underbrace{\left(\text{pow}\left(2,(\sum_{i=23}^{30}b_i^{(i-23)}-127)\right)\right)\vphantom{\sum_j}}_{exponent} \times \underbrace{\left(1+\sum_{j=22}^{0} b_j^{-(23-j)}\right)}_{fraction}\\[3ex]
    &= (-1) \times \text{pow}(2,124-127) \times 1.25\\
    &= -1 \times 0.125 \times 1.25\\
    &= 0.156250
\end{align*}
$$

### Possible Implementation

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CARRAY_LENGTH 32
#define value(c) (int) (c - '0')

float fp32(char s[CARRAY_LENGTH]) {
    float sign = pow(-1, value(s[31]));
    printf("sign value: %f\n", sign);
    
    float exponent = 0.f;
    for (int i = 23; i <= 30; ++i)
        if (value(s[i]))
            exponent += pow(2, (i - 23));
    exponent -= 127.f;
    printf("exponent value: %f\n", exponent);

    float fraction = 0.f;
    for (int i = 22; i >= 0; --i)
        if (value(s[i]))
            fraction += pow(2, -(23 - i));
    fraction += 1.f;
    printf("fraction value: %f\n", fraction);

    return sign * pow(2, exponent) * fraction;

}

int main(int argc, char *argv[]) {
    char s[CARRAY_LENGTH] = "00000000000000000000010001111100";
    float result = fp32(s);
    printf("%f\n", result);
}
```

### Subnormal Numbers

Recall *leading bit convention* that a leading bit always appears for **nonzero** exponents. Then single-precision floating point number with zero exponents is called a **subnormal numbers (or denormal numbers)**.
