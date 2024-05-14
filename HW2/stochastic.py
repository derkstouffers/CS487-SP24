# Deric Shaffer
# CS487 - HW2
# Due Date - Feb. 11th, 2024
# Stochastic Gradient Descent Class

import numpy as np

class AdalineSGD:
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
    
    def fit(self, x, y):
        self._initialize_weights(x.shape[1])
        self.losses_ = []
        for _ in range(self.n_iter):
            if self.shuffle:
                x, y = self._shuffle(x, y)
            losses = []
            for xi, target in zip(x, y):
                losses.append(self._update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)

            prediction = self.predict(x)
            accuracy = np.mean(prediction == y)
            print(f'Epoch {_} accuracy = {accuracy}')
        return self
    
    def partial_fit(self, x, y):
        if not self.w_initialized:
            self._initialize_weights(x.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(x, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(x, y)
        return self
    
    def _shuffle(self, x, y):
        r = self.rgen.permutation(len(y))
        return x[r], y[r]
    
    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float_(0.)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_ += self.eta * 2.0 * xi * (error)
        self.b_ += self.eta * 2.0 * error
        loss = error ** 2
        return loss

    def net_input(self, x):
        return np.dot(x, self.w_) + self.b_

    def activation(self, x):
        return x

    def predict(self, x):
        return np.where(self.activation(self.net_input(x)) >= 0.5, 1, 0)