# This is to perform gradient descent for linear regression manually.

import numpy as np

# The function we want to learn : f = w * x
x = np.array([1, 2, 3, 4], dtype=np.float32)
y = np.array([2, 4, 6, 8], dtype=np.float32)

# Obviously, the w here must be 2.

w = 0.0

def forward(x):
    return w * x

# The loss being used here is MSE
def loss(y, y_pred):
   return np.mean((y - y_pred) * (y - y_pred))

# The loss function here J is MSE.
# J = 1 / n * (w * x - y) ** 2
# dJ / dW = 1 / n * wx(w * x - y)
def gradient(x, y, y_pred): 
    return np.dot(2 * x, y_pred - y).mean()

print(f'prediction before training : f(5) = {forward(5)}')

lr = 0.01
epochs = 10

for i in range(epochs):
    pred = forward(x)
    l = loss(y, pred)
    grad = gradient(x, y, pred)
    
    # Weight updation using GD.
    w -= lr * grad
    
    print(f'epoch {i + 1} and w {w}, loss = {l}')
    
print(f"prediction after training : {forward(5)}")