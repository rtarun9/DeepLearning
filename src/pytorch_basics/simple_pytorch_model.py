
import numpy as np
import torch
import torch.nn as nn

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    
# The function we want to learn : f = w * x
# Now, each sample will have a single feature, so the tensor shape will have to change.
x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

# Obviously, the w here must be 2.

n_samples, n_features = x.shape

x_test = torch.tensor([5], dtype=torch.float32)

class CustomLRModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomLRModel, self).__init__()
        self.ln = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.ln(x)
    
# The loss being used here is MSE
loss = nn.MSELoss()

# The loss function here J is MSE.
# J = 1 / n * (w * x - y) ** 2
# dJ / dW = 1 / n * wx(w * x - y)
model = CustomLRModel(n_features, n_features)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print(f'prediction before training : f(5) = {model(x_test).item()}')

lr = 0.01
epochs = 200

for i in range(epochs):
    pred = model(x)
    l = loss(y, pred)
    # Gradient computation
    l.backward()
     
    optimizer.step()
    
    # Ensure gradients are cleared when we are going to the next epoch.
    optimizer.zero_grad()
    
    [w, b] = model.parameters()
    w = w.detach() 
    print(f'epoch {i + 1} and w {w.item()}, loss = {l}')
    
print(f"prediction after training : {model(x_test)}")