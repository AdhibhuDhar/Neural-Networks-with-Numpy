from tensor import Tensor
x=Tensor(2)
y=Tensor(3)
z=x*y+x
z.backward()
print("dz/dx:",x.grad)
print("dz/dy:",y.grad)