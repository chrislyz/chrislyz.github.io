---
title: Error Backpropagation
description: Error Backpropagation
date: 2023-06-25 00:00:00+0000
math: true
---

Error backpropagation (EBP) is an efficient algorithm perform gradient-based optimizations to the neural network.

## Derivation

> We will use the example and notations mentioned in [1] to derive the error backpropagation algorithm.

Let us start with an arbitrary deep feedforward neural network as illustrated in the below figure. Recall [Feedforward Neural Network Feedforward Process feedforward process], $z_k^{(L-1)}$ denotes the output of $k\text{-th}$ neuron in the last layer, and $w_{kj}^{(L)}$ denotes the weights connected between $j\text{-th}$ neuron and $k\text{-th}$ neuron in the successive layers $(L-1)$ and $(L)$. The lost function $\mathcal{E}$ is defined by the sum of mean squared error function (MSE) over the last layer, i.e.,

$$
\mathcal{E} = \sum_n\frac{1}{2}(\hat{z}_{n}^{(L)}-y)^2 \tag{1}
$$

And to minimize the loss (or error), we need to calculate the derivative of the corresponding loss function, i.e.,


$$
\nabla \mathcal{E} = \sum_n(\hat{z}_{n}^{(L)} - y) \tag{2}
$$

> [!note]
> Please note that $z_k^{(L)}$ generalizes the relationship in which any neuron in $(L)$ layer is connected to the $j\text{-th}$ neuron in $(L-1)$ layer.

![]({{site.baseurl}}/assets/media/Pasted image 20230810140946.png)

In the feedforward process, we have a weighted sum $a_k$ and the output of an activated neuron $z_k$,


$$
\begin{align}
&a_k^{(L)} = \sum_j w_{kj}^{(L)}z_j^{(L-1)} + b_j^{(L-1)} \tag{3}\\
&z_k^{(L)} = h(a_k^{(L)}) \tag{4}
\end{align}
$$


where $h(\cdot)$ is an [[Activation Functions|activation function]]. As the network is learning the weights of parameters, the loss function should be minimized with respect to the weights, i.e.,


$$
\nabla_{w^{(L)}}\mathcal{E} = \dfrac{\partial E(w^{(L)})}{\partial w^{(L)}}
$$


Since the loss function is also a function of output of all neurons given by $(1)$, the partial derivative $\partial \mathcal{E}/\partial z_k$ can further be expressed by the chain rule,


$$
\begin{align}
\nabla\mathcal{E} &= \dfrac{\partial E(w^{(L)})}{\partial w^{(L)}}\\[1ex]
&= \sum_k\dfrac{\partial \mathcal{E}}{\partial z_k^{(L)}}\dfrac{\partial z_k^{(L)}}{\partial w^{(L)}}\\[1ex]
\end{align}
$$


Let us introduce a notation $\delta_k^{(L)}$ referring to the partial derivative of loss function given the $k\text{-th}$ neuron in $(L)$ layer to show the backward process. And given by $(2)$, we have,


$$
\delta_k^{(L)} \equiv \dfrac{\partial \mathcal{E}}{\partial z_k^{(L)}} = \hat{z}_k^{(L)} - y_k
$$


Similarly, the output of neuron $z_k$ is a function of weighted sum $a_j$ given by $(2)$ that is a function of desired independent variable $w$. Hence, applying the chain rule to $\partial z_j / \partial w$ gives the final derivative,


$$
\begin{align}
\nabla \mathcal{E}&=\dfrac{\partial\mathcal{E}}{\partial z_k^{(L)}} \dfrac{\partial z_k^{(L)}}{\partial a_k^{(L)}}\dfrac{\partial a_k^{(L)}}{\partial w_{kj}^{(L)}}\\
&= \sum_k\delta_k^{(L)}h^{\prime}(a_k^{(L)})z_j^{(L-1)}
\end{align}
$$



$$
\begin{align}
\delta_j^{(L-1)} \equiv \dfrac{\partial \mathcal{E}}{\partial z_j^{(L-1)}} &= \sum_k \dfrac{\partial\mathcal{E}}{\partial z_k^{(L)}}\dfrac{\partial z_k^{(L)}}{\partial a_k^{(L)}}\dfrac{\partial a_k^{(L)}}{\partial z_j^{(L-1)}}\\
&= \sum_k \delta_k^{(L)} h^{\prime}(a_k^{(L)})w_{kj}^{(L)}
\end{align}
$$


It can be seen that each partial derivative of the loss function with respect to the previous layer is derived backwards. Therefore, backpropagation algorithm, as the name suggests, is a backward algorithm.

> [!note]
> In the textbook [1], $\delta$ is the partial derivative of the loss function with respect to the weighted sum given by $(5.51)$. Additionally, the textbook omit the activation function $h(a_k)$ for the final layer which is included here.

Likewise, the derivative of the loss function in layer $(L-1)$ with respect to weights is given by,


$$
\begin{align}
\nabla\mathcal{E}&=\dfrac{\partial\mathcal{E}}{\partial z_k}\dfrac{\partial z_k}{\partial a_k}\dfrac{\partial a_k}{\partial z_j}\dfrac{\partial z_j}{\partial a_j}\dfrac{\partial a_j}{\partial w_{ji}}\\

\end{align}
$$


The derivative can be computed all the way back to the first hidden layer. in a way that gradients of each layer is flowing backwards.

A tree shows the computational structure of each layer [2].
![]({{site.baseurl}}/assets/media/Pasted image 20230810145453.png)

## Reference

- [1] C. M. Bishop, _Pattern recognition and machine learning_. in Information science and statistics. New York: Springer, 2006.
- [2] https://www.3blue1brown.com/lessons/backpropagation-calculus