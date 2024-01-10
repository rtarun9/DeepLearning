import torch 
import torch.nn as nn
import numpy as np
import sklearn 
import matplotlib.pyplot as plt

# The linear regression function we want to learn is 8x - 4.33
def compute_y(x):
    return 8 * x - 4.33

x_data = np.array(range(0, 120)).astype(np.float32)
y_data = np.array(compute_y(x_data)).astype(np.float32)

x_data = x_data.reshape((-1, 1))
y_data = y_data.reshape((-1, 1))

# Now, creating the tensors from this (along with reshaping).
x_tensor = torch.from_numpy(x_data)
y_tensor = torch.from_numpy(y_data)

print(f"x_tensor shape : {x_tensor.shape}, same for y : {y_tensor.shape}")

# Create the model.
class LR(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(LR, self).__init__()
        self.linear = nn.Linear(input_dims, output_dims)

    def forward(self, x):
        return self.linear(x)

n_samples, n_features = x_tensor.shape
lr = LR(n_features, n_features)

# Specify the loss function
loss = nn.MSELoss()
optimizer = torch.optim.SGD(params = lr.parameters(), lr=0.0001)

epochs = 1000
for epoch in range(epochs):
    # First step, forward pass.
    y_pred = lr(x_tensor)
    
    # Loss computation.
    l = loss(y_pred, y_tensor)
    
    # Gradient computation.
    l.backward()
    
    # Weight updation.
    optimizer.step()

    # Zero gradients after weight updation.
    optimizer.zero_grad()
    
    [w, b] = lr.parameters()
    # Print the current loss to console.
    print(f"epoch={epoch} and loss={l.detach()}")
    print("w and b : ", w.item(), b.item()) 
    
print("Test for model prediction : ")
print("model(1000) = ", lr(torch.tensor([1000], dtype=torch.float32)).item())

predicted = lr(x_tensor).detach().numpy()
plt.plot(x_data, y_data, 'b')
plt.plot(x_data, predicted, 'ro')
plt.show() 