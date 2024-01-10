import torch
import numpy

print("So apparently, when the tensor is on the gpu, we cannot do .numpy()). Lets test this.")
device = 'cpu'

x = torch.ones(3, device=device)
print(x, x.numpy(), " this will work fine")

print("Now, x will be on the gpu (if cuda is available that is)")
if torch.cuda.is_available():
    device = 'cuda'
    
print(x, x.to(device), "This works fine")
# print(x.to(device).numpy(), " This will not work")
print(x.to(device).cpu().numpy(), " verbose, but this will work")