import numpy as np
import torch

a = torch.ones(3)
print("Tensor : ", a)

b = a.numpy()
print("Numpy nd array : ", b)

# Note : even if you do .numpy(), they both still point to the 
# same memory location.

b = b * 2
print("a and b : ", a, b)

a = a + a + a
print("a and b : ", a, b)

print("So far, they both seem to work independently")

print("That changes when inplace modification is done to tensor")

a = torch.rand(3)
b = a.numpy()
print("a and b before modifications : ", a, b)
a.add_(1)
print("a and b : ", a, b)
a = a + 4
print("a and b after a non-inplace operation", a, b)
print("What if change is done on b?")
b = b + 2
print("a and b after b was modified : ", a, b)

print("Moral of the story : with inplace operations (and change on the tensor), the numpy nd array will also change as they point to the same memory location") 

print("Converting a numpy array to a torch tensor")
a = np.ones(5)
b = torch.from_numpy(a).type(torch.float16)

print("torch tensor from numpy array : ", b)