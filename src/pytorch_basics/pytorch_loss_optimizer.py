# This is to perform gradient descent for linear regression manually.

import numpy as np
import torch
import torch.nn as nn

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    
# The function we want to learn : f = w * x
x = torch.from_numpy(np.array([1, 2, 3, 4], dtype=np.float32))
y = torch.from_numpy(np.array([2, 4, 6, 8], dtype=np.float32))

# Obviously, the w here must be 2.

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
    return w * x

# The loss being used here is MSE
loss = nn.MSELoss()

# The loss function here J is MSE.
# J = 1 / n * (w * x - y) ** 2
# dJ / dW = 1 / n * wx(w * x - y)
optimizer = torch.optim.SGD(params=[w], lr=0.01)

print(f'prediction before training : f(5) = {forward(5)}')

lr = 0.01
epochs = 100

for i in range(epochs):
    pred = forward(x)
    l = loss(y, pred)
    # Gradient computation
    l.backward()
     
    optimizer.step()
    
    # Ensure gradients are cleared when we are going to the next epoch.
    optimizer.zero_grad()
     
    print(f'epoch {i + 1} and w {w}, loss = {l}')
    
print(f"prediction after training : {forward(5)}")