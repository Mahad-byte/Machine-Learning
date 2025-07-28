import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def linear(x, a=1.0):
    return a * x


def linear_derivative(x, a=1.0):
    return np.ones_like(x) * a


def binary_step(x):
    return np.where(x >= 0, 1, 0)


def binary_step_derivative(x):
    return np.zeros_like(x)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)


def prelu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)


def prelu_derivative(x, alpha=0.1):
    return np.where(x > 0, 1, alpha)


def main():
    # Test values
    test_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    print("Input values:", test_values)

    print("\nSigmoid:")
    print("Function:", sigmoid(test_values))
    print("Derivative:", sigmoid_derivative(test_values))

    print("\nLinear:")
    print("Function:", linear(test_values, 0.5))
    print("Derivative:", linear_derivative(test_values, 0.5))

    print("\nBinary Step:")
    print("Function:", binary_step(test_values))
    print("Derivative:", binary_step_derivative(test_values))

    print("\nTanh:")
    print("Function:", tanh(test_values))
    print("Derivative:", tanh_derivative(test_values))

    print("\nReLU:")
    print("Function:", relu(test_values))
    print("Derivative:", relu_derivative(test_values))

    print("\nLeaky ReLU:")
    print("Function:", leaky_relu(test_values, 0.1))
    print("Derivative:", leaky_relu_derivative(test_values, 0.1))

    print("\nPReLU:")
    print("Function:", prelu(test_values, 0.25))
    print("Derivative:", prelu_derivative(test_values, 0.25))


if __name__ == "__main__":
    main()