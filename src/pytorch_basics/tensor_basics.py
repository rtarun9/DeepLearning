import torch

print("torch cuda availability : ", torch.cuda.is_available())

empty_tensor = torch.empty(1)
print('empty tensor of size 1 : ', empty_tensor)

empty_tensor = torch.empty(3)
print("empty tensor of size 3 : ", empty_tensor)

empty_tensor = torch.empty((3, 3))
print("empty tensor of size 3x3 : ", empty_tensor)

random_tensor = torch.rand((3, 3))
print("Random tensor : ", random_tensor)

tensor_with_float16_dt = torch.ones((3, 3), dtype=torch.float16)
print("Tensor with float16 datatype : ", tensor_with_float16_dt)
print("Type of tensor and shape : ", tensor_with_float16_dt.dtype, tensor_with_float16_dt.shape)

# Basic tensor operations:
x = torch.rand(2)
y = torch.rand(2)

z = x + y
print("x and y tensor : ", x, y)
print("torch element wise addition : ", z)
print("Same, but with torch.add function : ", torch.add(x, y))
y.add_(x)
print("Addition but inplace (actually modifies y as x + y ", y)
print("Element wise multiplication : ", torch.mul(x, y))

print("--------------------------")

print("For reshaping tensors : ")
x = torch.rand((3, 3))
print("Original x : ", x)

x = x.view(-1, 9)
print("Modified x : ", x)


