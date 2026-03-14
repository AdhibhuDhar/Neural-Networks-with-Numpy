#autograd func
#tensor obj stores val,grad,what func made it,parents
import numpy as np

class Tensor:

    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=float)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) #gradient loss wrt tensor

        self._backward = lambda: None #for leaf default rule is nothing

        self._prev = set() #stores parents

    def __add__(self, other):

        other = other if isinstance(other, Tensor) else Tensor(other) #eg x+3,converts 3 to tensor automatically,everything inside a graph is a tensor

        out = Tensor(self.data + other.data)#fwd result,out is a new tensor node

        def _backward():#defines grad rule for addn
            self.grad += out.grad#chain rule
            other.grad += out.grad#both parents recieve same gradient

        out._backward = _backward#attach grad node to output node
        out._prev = {self, other}#store parents


        return out

    def __mul__(self, other):

        other = other if isinstance(other, Tensor) else Tensor(other)#everything inside graph is tensor

        out = Tensor(self.data * other.data)#fwd mul

        def _backward():#define gradient rule
            self.grad += other.data * out.grad#chain rule
            other.grad += self.data * out.grad#both parents rcv same grad

        out._backward = _backward#attach grad rule to result node   
        out._prev = {self, other}#store parents

        return out

    def backward(self):

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()
