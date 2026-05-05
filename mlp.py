import numpy as np


class MLP():
    def __init__(self, layers_dims):
        self.layers_dims = layers_dims
        self.weights = []
        self.biases = []
        for l in range(len(layers_dims) - 1):
            n_in = layers_dims[l]
            n_out = layers_dims[l + 1]

            W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
            b = np.random.randn(1, n_out) * 0.01

            self.weights.append(W)
            self.biases.append(b)

        self.cache = {'A': [], 'Z': []}

    def activation(self, x):
        '''Функция активации - RelU'''
        return np.maximum(x, 0)

    def loss(self, y_pred, y):
        '''Функция потерь - MSE'''
        return 0.5 * ((y_pred - y) ** 2).sum() / len(y)

    def forward(self, X):
        '''Функция прохода по всей сети'''
        self.cache = {'A': [X], 'Z': []}
        A = X

        for l in range(len(self.weights)):
            z = (A @ self.weights[l].T) + self.biases[l]

            if l == len(self.weights) - 1:
                a = 1 / (1 + np.exp(-z))
            else:
                a = np.maximum(0, z)

            self.cache['A'].append(a)
            self.cache['Z'].append(z)
            A = a

        return a

    def backward(self, X, y, lambd):
        '''Функция обратного распространения ошибки'''
        m = X.shape[0]
        grads = {'dW': [], 'db': []}
        L = len(self.weights) - 1
        A = self.cache['A'][L + 1]
        A_prev = self.cache['A'][L]
        Z_last = self.cache['Z'][L]

        dA = (A - y) / m

        dZ = dA * A * (1 - A)

        dW = (dZ.T @ A_prev) / m
        dW += (lambd / m) * self.weights[L]

        db = np.sum(dZ, axis=0, keepdims=True) / m
        grads['dW'].append(dW)
        grads['db'].append(db)

        for l in range(len(self.weights) - 2, -1, -1):
            A = self.cache['A'][l + 1]
            A_prev = self.cache['A'][l]
            Z_curr = self.cache['Z'][l]
            W = self.weights[l + 1]

            dA = dZ @ W

            dZ = dA * (Z_curr > 0).astype(float)

            dW = (dZ.T @ A_prev) / m
            dW += (lambd / m) * self.weights[l]

            db = np.sum(dZ, axis=0, keepdims=True) / m
            grads['dW'].append(dW)
            grads['db'].append(db)

        grads['dW'] = grads['dW'][::-1]
        grads['db'] = grads['db'][::-1]
        return grads

    def update_params(self, grads, learning_rate=0.01):
        '''Обновление весов'''
        for l in range(len(self.weights)):
            self.weights[l] -= learning_rate * grads['dW'][l]
            self.biases[l]  -= learning_rate * grads['db'][l]
    
    def fit(self, X, y, batch_size=8, epoch=5, learning_rate=0.1, lambd=0.0):
        '''Процесс обучения'''
        n = len(y)
        idx = np.arange(n)
        
        for _ in range(epoch):
            np.random.shuffle(idx)
            epoch_loss = 0
            num_batches = 0

            for i in range(0, n, batch_size):
                idx_l, idx_r = i, min(i + batch_size, n)
                X_batch = X[idx[idx_l:idx_r]]
                y_batch = y[idx[idx_l:idx_r]]

                y_pred = self.forward(X_batch)

                epoch_loss += self.loss(y_pred, y_batch)
                num_batches += 1

                grads = self.backward(X_batch, y_batch, lambd)
                self.update_params(grads, learning_rate)

if __name__ == '__main__':
    X = np.array([[0, 0 ],
                 [0, 1],
                 [1, 0],
                 [1, 1]])
    y = np.array([[0],
                 [1],
                 [1],
                 [0]])

    model = MLP(layers_dims=[2, 4, 4, 1])
    print(f'Начало обучения')
    model.fit(
        X=X,
        y=y,
        epoch=1000,
        batch_size=4,
        learning_rate=1.5,
        lambd=0.005
        )
    print(f'Конец обучения')
    y_pred = model.forward(X)
    print(f'Loss: {model.loss(y_pred, y)}')
    print(f'Предсказания: {y_pred}')
