import numpy as np


class Neuron():
    def __init__(self, num_input):
        self.weights = np.ones(num_input)
        self.bias = 0.0

    def forward(self, X):
        '''Сумматорная функция'''
        return (X @ self.weights) + self.bias

    def activation(self, z):
        '''Активационная функция - сигмоида в моём случае'''
        return 1 / (1 + np.exp(-z))
    
    def loss(self, X, y):
        '''Функция потерь'''
        y = y.ravel()
        y_pred = self.activation(self.forward(X))
        return np.mean(0.5 * ((y_pred - y) ** 2))

    def gradient(self, X, y):
        '''Градиент аналитическим методом'''
        y = y.ravel()
        n = len(y)
        y_pred = self.activation(self.forward(X))
        
        nebula_J_w = (1 / n) * (X.T @ ((y_pred - y) * y_pred * (1 - y_pred)))
        nebula_J_b = (1 / n) * ((y_pred - y) * y_pred * (1 - y_pred)).sum()
        return nebula_J_w, nebula_J_b

    def update_mini_batch(self, X, y, learning_rate):
        nebula_J_w, nebula_J_b = self.gradient(X, y)
        self.weights -= nebula_J_w * learning_rate
        self.bias -= nebula_J_b * learning_rate

    def SGD(self, X, y, batch_size=10, learning_rate=0.01, epochs=5):
        '''Стохастический градиентный спуск'''
        n = len(y)
        idx = np.arange(0, n)
        for epoch in range(epochs):
            np.random.shuffle(idx)
            for i in range(0, n, batch_size):
                idx_l = i
                idx_r = min(i + batch_size, n)
                self.update_mini_batch(X[idx[idx_l:idx_r]], y[idx[idx_l:idx_r]], learning_rate=learning_rate)


if __name__ == '__main__':
    X = np.random.randn(100, 3)
    y = np.random.randint(0, 2, size=(100, 1))

    model = Neuron(num_input=3)
    model.SGD(X, y, batch_size=16, learning_rate=0.05, epochs=10)
    print("Веса после обучения 10 эпох:", model.weights)
    model.SGD(X, y, batch_size=16, learning_rate=0.05, epochs=100)
    print("Веса после обучения 100 эпох:", model.weights)
    model.SGD(X, y, batch_size=16, learning_rate=0.05, epochs=500)
    print("Веса после обучения 500 эпох:", model.weights)
