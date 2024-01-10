import torch
print('cuda availability : ', torch.cuda.is_available())

a = torch.ones(2, requires_grad=True)

b = a + 2 # A computational graph is created when this add operation is done.
print(a, b)

# Now, peforming backpropogation and displaying the gradients.
v = torch.tensor([0.1, 0.2])
b.backward(gradient=v)

print(a.grad)

print('for testing and some times we will want to prevent gradient tracking.')
with torch.no_grad():
    b = a + 2
    print(a, b)