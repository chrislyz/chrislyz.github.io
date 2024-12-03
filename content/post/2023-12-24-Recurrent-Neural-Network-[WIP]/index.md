---
title: Recurrent Neural Networks
date: 2023-12-24
math: true
---



## History

Long Short-term memory was first introduced in the paper with the same name by Hochreiter and Schmidhuber [1].

## Prior Knowledge

### Difference Equation

A difference equation is an equation describing the relationship between a sequential variable (e.g., the present state of a system and its previous states). A trivial example is


$$
\begin{align}
f(n) = f(n-1) + f(n-2) + 2\ &&\text{given that }f(1)=1, f(2)=2
\end{align}
$$


where the given premise is called *initial condition*. The *order* of the equation refers to number of previous values that the present depends on, where the example above is hence a second-order difference equation. Accordingly, a difference equation can be categorized into first-order, second-order and higher-order difference equations. 

Besides, based on whether the equation can be written in the form that all its arguments are multiplied by a scalar (i.e., $f(s_x1,\ldots,sx_n)=sf(x_1,\ldots,x_n)$), we characterize a difference equation to be homogeneous and inhomogeneous respectively. Collaborate with the definition above, it is obvious that the example is inhomogeneous.

> [!info]
> The relationship between difference equations and differential equation can be seen as if discreteness and continum. That is being said, difference equations range from discrete time steps whereas differential equations range from a continuous time interval.

## Problem Statement

The traditional feedforward neural network does not comprehend nor can process the sequential data, not to mention the variable-length sequential data. The two most important reasons are, first traditional feedforward neural networks can not handle endless sequential data in practical, as they are finite. Moreover, the independence between neurons is difficult for MLP to persist (context) information during sequential events.

> [!note]
> The independence of neurons can be explained as that the model learns the mapping of semantic information independently in each position of sequential data. Not only does the method significantly increase the computational complexity (i.e., $n$ tokens in maximum $k$-length sequential data O($n^k$)).
>
> Take the sentences "In 2009, blablabla" and "Blablabla in 2009" as example. The date information can be learnt independently twice only because they are in different position in two sentences. Hence, no context information will be persistent across time.

## Solution

By definition, a Recurrent Neural Network (RNN) is a type of neural network that is specialized for processing a sequence of values $x^{(1)},\ldots,x^{(\tau)}$. In addition, most RNNs are not constrained to the fixed length sequences.

RNN extends the idea of parameter sharing across different parts of a model which is essential to the relevant information occurring at multiple positions with the sequence.

Consider the classical dynamic system, we have

![]({{site.baseurl}}/assets/media/Pasted image 20230925120515.png)


$$
s^{(t)} = f(s^{(t-1)}; \theta)
$$



Similarly, each member of the output in RNN is a function of the previous member of the output, except that RNN also accepts external inputs, i.e.,

![]({{site.baseurl}}/assets/media/Pasted image 20230925160528.png)


$$
h(t) = f(h^{(t-1)}, x^{(t)}; \theta)
$$


where $h^{(t)}$ is the hidden unit at time $t$, and $x^{(t)}$ is the external input at time $t$.

> [!note]
> $h^{(t)}$ is the latent variable storing the information from time $0$ to $t$. In other words, RNN models the conditional probability by introducing the latent variable,
>
>
> $$
> P(x_t|x_{t-1},\cdots,x_1) \approx P(x_t|h_{t-1})
> $$

The unfolded computational graph (on the right) shows that RNN produces an output $\mathbf{o}^{(t)}$ at each time $t$. The loss $L$ computes $\mathbf{\hat{y}}=\mathrm{softmax}(\mathbf{o})$ and compares this to the target $\mathbf{y}$. RNN has three important parameter matrices: hidden-to-hidden recurrent connections $W$, input-to-hidden connections $U$ and hidden-to-output connections $V$.

![]({{site.baseurl}}/assets/media/Pasted image 20230925161147.png)

### Forward propagation


$$
\begin{align}
h^{(t)}&=\tanh(\mathbf{b}+Wh^{(t-1)}+Ux^{(t)})\\
o^{(t)}&=\mathbf{c}+Vh^{(t)}\\
\hat{y}^{(t)}&=\mathrm{softmax}(o^{(t)})
\end{align}
$$


where $\mathbf{b}$ and $\mathbf{c}$ are bias vectors, $\tanh$ is the activation function for hidden units, and $\mathrm{softmax}$ is the activation function for output units. As the current state is strongly dependent on the previous state, the runtime complexity is $O(t)$ and can not be further reduced.

### Loss

The loss becomes the sum of total loss at each timestamp, i.e.,


$$
\begin{align}
L\left(\{x^{(1)},\ldots,x^{(t)}\},\{y^{(1)},\ldots,y^{(t)}\}\right)&=\sum_tL^{(t)}\\
&=-\sum_t\log p_{\mathrm{model}}\left(y^{(t)}|\{x^{(1)},\ldots,x^{(t)}\}\right)
\end{align}
$$


where loss $L$ is calculated with negative $\log$-likelihood.

## Representation of Input

As the Recurrent Neural Network models with numeric values, the input of the model should be collaborated in such a condition.

Some famous modeling methods are listed as following,

- [[Bag-of-Words]]
- [[Word Embeddings]]

## Bidirectional RNN

## Other Variants

- [[Long Short Term Memory]]
- [[Gated Recurrent Unit]]

## Implementation

### PyTorch

```python
class RNNBase:
    def __init__(self, mode: str, input_size: int, hidden_size: int,
                 dropout: float = 0) -> None:
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = float(dropout)

    # ...
    # parameter validation
    # ...

class RNN(RNNBase):
    @overload
    def __init__(self, input_size: int, hidden_size: int,
                 dropout: float = 0,
                 nonlinearity: str = 'tanh') -> None:
        ...

    @overload
    def __init__(self, *args, **kwargs) -> None:
        ...

    def __init__(self, *args, **kwargs) -> None:
        self.nonlinearity = kwargs.get('nonlinearity', 'tanh')
        if self.nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif self.nonlinearity == 'relu':
            mode = 'RNN_RELU'
        super().__init__(mode, *args, **kwargs)

    def forward(self, input, hx=None):
        num_directions = 1
        if hx is None:
            hx = torch.zeros(self.num_layers * num_directions,
                             max_batch_size, self.hidden_size)

        if self.mode == 'RNN_TANH':
            result = torch.tanh()
        elif self.mode == 'RNN_RELU':
            result = torch.relu()
```

## Suffering

Long term memory dependencies.

## Reference

- [1] A. Sherstinsky, “Fundamentals of Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) Network,” _Physica D: Nonlinear Phenomena_, vol. 404, p. 132306, Mar. 2020, doi: [10.1016/j.physd.2019.132306](https://doi.org/10.1016/j.physd.2019.132306).
- [2] S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” _Neural Computation_, vol. 9, no. 8, pp. 1735–1780, Nov. 1997, doi: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735).
- [3] 7 - Difference Equations. Url: https://www.cl.cam.ac.uk/teaching/2003/Probability/prob07.pdf
- [4] Ian Deep learning.