from typing import List, Optional

import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

np.random.seed(42)


class Network:
    """
    Basic Neural Network, totally unoptimized
    Uses Stochastic Gradient Descent as the optimizer
    """

    def __init__(self, sizes: list[int]):
        self.num_layers = len(sizes)
        self.sizes = sizes

        # Where x is the size of the previous layer and y the size of the next layer
        self.w = [
            np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]
        self.b = [np.random.randn(y, 1) for y in self.sizes[1:]]

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid Activation Function"""
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_deriv(self, z: np.ndarray):
        """first derivative of sigmoid evaluated at z"""
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward(self, a: np.ndarray):
        """Forward Pass through the Network"""
        for w, b in zip(self.w, self.b):
            z = np.matmul(w, a) + b
            a = self.sigmoid(z)
        return a

    def cost_deriv(self, a_L, y):
        return a_L - y

    def evaluate(self, test_data: np.ndarray):
        """
        Nr of correctly classified test-samples
        """
        # Assumes one hot encoded targets
        test_results = [
            (np.argmax(self.forward(x)), np.argmax(y)) for x, y in test_data
        ]
        return sum(int(x == y) for x, y in test_results)

    def train(
        self,
        train_data: List[tuple[np.ndarray, np.ndarray]],
        epochs: int = 20,
        batch_size: int = 30,
        lr: float = 0.01,
        test_data: Optional[List[tuple[np.ndarray, np.ndarray]]] = None,
    ):
        """
        Training using Stochastic Gradient Descent.

        If test_data is passed, then the network is evaluated on the test_data after each epoch
        """
        if not train_data:
            raise ValueError("train_data can not be none")
        n_train = len(train_data)
        n_test = len(test_data) if test_data else None
        n_batches = n_train // batch_size

        test_results = []
        for epoch in range(epochs):
            for step in range(n_batches):
                batch = [
                    train_data[idx]
                    for idx in np.random.randint(low=0, high=n_train, size=batch_size)
                ]
                self.minibatch_update(batch, lr)

            if test_data:
                """Compute the cost on the test set"""
                test_result = self.evaluate(test_data)
                test_results.append(test_result)
                print(
                    f"Epoch: {epoch + 1} / {epochs},\t True Positives: {test_result}/{n_test}\t accuracy: {test_result / n_test:.5f}"
                )

    def minibatch_update(self, batch: List[tuple[np.ndarray, np.ndarray]], lr: float):
        """
        Runs one minibatch update

        Key Equations of backpropagation

        BP​1: δ^L =  ∇_a C ⊙ σ'(z^L)                         Get Error δ in last layer of network ∇_a C for quadratic cost 0.5(y(x) - a^L(x))² -> (a^L - y)
        BP2: δ^l = ((w^{l+1})^T δ^{l+1}) ⊙ σ'(z^L)          Propagate Errors from last layer (BP1) through rest of network to all layers
        BP3: ∂C/∂b^l_j = δ^l_j -> ∂C/∂b = δ                 Rate of change of cost wrp. to any bias
        BP4: ∂C/∂w^l_jk = a^{l-1}_k δ^l_j -> a_in δ_out     Rate of change of cost wrp. to any weight

        For all samples in minibatch
            1. Get the errors and activations for all nodes
            2. Calculate the gradients at the nodes and save them (accumulate gradients)

        Calculate average gradient at each node accumulated gradients/n_samples
        3. Update weights and biases according to update rule
            w_k' = w_k - lr * dC/dw_k
        """
        grad_w = [np.zeros(w.shape) for w in self.w]
        grad_b = [np.zeros(b.shape) for b in self.b]

        for x, y in batch:
            # Backprop
            # grad_w = [np.zeros(w.shape) for w in self.w]
            # grad_b = [np.zeros(b.shape) for b in self.b]
            activation = x
            activations = [activation]
            zs = []
            # forward pass
            for w, b in zip(self.w, self.b):
                z = np.dot(w, activation) + b
                zs.append(z)
                activation = self.sigmoid(z)
                activations.append(activation)

            # backward pass
            # Get all deltas
            delta = self.cost_deriv(activations[-1], y) * self.sigmoid_deriv(zs[-1])

            grad_b[-1] += delta
            grad_w[-1] += np.dot(delta, activations[-2].T)

            for l in range(2, self.num_layers):
                delta = np.dot(self.w[-l + 1].T, delta) * self.sigmoid_deriv(zs[-l])
                grad_b[-l] += delta
                grad_w[-l] += np.dot(delta, activations[-l - 1].T)

        # Accumulated gradients over all samples in batch
        # now update
        m = len(batch)

        self.w = [w - (lr / m) * dC_dw for w, dC_dw in zip(self.w, grad_w)]
        self.b = [b - (lr / m) * dC_db for b, dC_db in zip(self.b, grad_b)]

        return grad_w, grad_b


def load_data_MNIST():
    """
    Loads MNIST train and test data flattened, unbatched
    One Hot Encodes the targets
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # load mnist data as grayscale 28x28 images flattened to 784 element vectors
    x_train = x_train.reshape(60000, 784, 1).astype("float32") / 255
    x_test = x_test.reshape(10000, 784, 1).astype("float32") / 255
    y_train = to_categorical(y_train, 10).reshape(60000, 10, 1)
    y_test = to_categorical(y_test, 10).reshape(10000, 10, 1)

    train_data = list(zip(x_train, y_train))  # evtl also reshape to 10,1
    test_data = list(zip(x_test, y_test))

    return train_data, test_data


def main():
    train_data, test_data = load_data_MNIST()
    net = Network([784, 20, 10])
    net.train(train_data, epochs=20, batch_size=30, lr=0.01, test_data=test_data)


if __name__ == "__main__":
    main()
