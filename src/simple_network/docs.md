# Simple Neural Network

*This text is inspired by chapters 1 & 2 of Michael Nielsens free online book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html). I recommend checking it out if you want more detailed descriptions.*

This document should help explain the basic concepts of how a neural network works and give you a better understanding how the [implementation](src/simple_network/simple_network.py) works.

**Topics covered**

- Neural Network description
- Notation
- Some Math
- Backpropagation

## Neural Networks description

Okay, so you probably already know that you can fit a line to some points using linear regression.
![Linear Regression](./img/neural_network_structure.HEIC)

Now, only fitting a line is not very useful in the real world, since most data is not linear. Neural networks help us fit more complex functions to data.
Thats where neural networks come in. They are able to fit arbitrarily complex functions to data, given enough neurons and training time. Really, neural networks are just function approximators.

## Gradient Descent, why it works

At the start, all the parameters of the network are random, we now need an update rule to change the parameters in a way that the output of the network gets closer to the expected output.

It can be said that in an example with two parameters $v_1, v_2$ a change in the cost function $C$ can be approximated as:
$$
\begin{align}
\Delta C &\approx \frac{\partial C}{\partial v_1} \Delta v_1 + \frac{\partial C}{\partial v_2} \Delta v_2 \\
\end{align}
$$
We now want to choose $\Delta v = (\Delta v_1, \Delta v_2)^T$ as to make $\Delta C$ negative, so that the cost function decreases.

If we have $\nabla C = ( \frac{\partial C}{\partial v_1},  \frac{\partial C}{\partial v_2})²$ then 1) can be rewritten as:
$$
\Delta C  \approx \nabla C \cdot \Delta v
$$

Now if we choose $\Delta v = -\eta \nabla C$ and some  small $\eta > 0$ (the learning rate) we get:
$$
\Delta C \approx -\eta ||\nabla C||^2
$$
And since $||\nabla C||²$ will always be positive, the $\Delta C$ will always be negative, and the cost function will decrease.

And so we have our update rule:
$$
v \rightarrow v - \eta \nabla C
$$
And this works for any number of parameters, not just two and so we can apply it to our neural network.

## Architecture

(Insert picture )

$w_{jk}^l$Weight connecting neuron $k$ in layer $l-1$ to neuron $j$ in layer-$l$

okay fuck this intro shit imma do backprop now.

## Backpropagation

We have a neural network, for example one with three layers follows this function:
$$
f(x; W,b) = \sigma(w² \sigma(w¹  \sigma(w⁰x + b⁰) + b¹) + b²)
$$
Now our goal is to minimize this function with respect to a Cost/Loss function. In our MNIST example this is simply the Squared Error
$$
C = \frac{1}{2n}\sum_x ||y(x) - a^L(x)||
$$
Where $y(x)$ is the expected output, $a^L(x)$ is the output of our network (the networks last activation layer) and $||x||$ is the Euclidean Norm.
We use $2n$ so that the derivative of the individual cost $C_x$ becomes a bit nicer and it does not affect the optimization since the cost is just scaled by a factor.

$$
C_x = \frac{1}{2}||y(x) - a^L(x)|| \quad \quad \nabla_a C_x=a^L(x)-y(x)
$$

To find the minimum of this function we can use Gradient Descent

### Matrix Form

Neural networks are actually all matrix multiplications, so we can write the earlier equations a bit simpler.

### Motivation
