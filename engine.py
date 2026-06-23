# engine.py

import math

class Value:
    """
    During the forward pass we compute values.
    During the backward pass we compute influence.
    """
    def __init__(self, data, _children=(), _op="") -> None:
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op

        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad 
        
        out._backward = _backward
        return out
    
    def __neg__(self):
        return self * Value(-1)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))

        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        t = math.tanh(x)

        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
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
        
        self.grad = 1.0
        for node in reversed(topo):

            node._backward()
            