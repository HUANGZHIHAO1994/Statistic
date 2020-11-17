import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def __call__(self, params, grads):
        params -= self.lr * grads


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def __call__(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)

        self.v = self.momentum * self.v + grads
        params -= self.lr * self.v


class RMSProp:
    def __init__(self, lr=0.01, alpha=0.9, eps=1e-08):
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.v = None

    def __call__(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)

        self.v = self.alpha * self.v
        self.v += (1 - self.alpha) * np.square(grads)
        eta = self.lr / (np.sqrt(self.v) + self.eps)
        params -= eta * grads


class Adam:
    def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-08):
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.m = None
        self.v = None
        self.n = 0

    def __call__(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
        if self.v is None:
            self.v = np.zeros_like(params)

        self.n += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grads)

        alpha = self.lr * np.sqrt(1 - np.power(self.beta2, self.n))
        alpha = alpha / (1 - np.power(self.beta1, self.n))

        params -= alpha * self.m / (np.sqrt(self.v) + self.eps)
