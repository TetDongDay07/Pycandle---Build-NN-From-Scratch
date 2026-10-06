import math
class Value:
    def __init__ (self, data, prev =(), op='', label=''):
        self.data=data
        self._prev=tuple(prev)
        self._op =op
        self.grad=0.0
        self.label=label
        self._backward=lambda: None

    def __repr__(self):
        return f'Value(data={self.data})'
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other),'+')
        def _backward():
            self.grad += out.grad
            other.grad +=  out.grad
        out._backward = _backward  
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self * (-1)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other),'*')
        def _backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        out._backward = _backward    
        return out
    
    def __rmul__(self, other):
        return self*other

    
    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data ** other.data, (self, other),'**')
        def _backward():
            self.grad += (math.pow(self.data, other.data-1) * other.data) * out.grad
        out._backward=_backward    
        return out
    
    def __truediv__(self, other):
        return self * (other**-1)
    
    def __rtruediv__(self, other):
        return other * (self**-1)
    
    
    def exp(self):
        out = Value(math.exp(self.data),(self,),op='exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward    
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    
    # In backward, a node only backward once the nodes that use it already done
    def backward(self):
        topo=[]
        visited=set()
        def build_topo(u):
            if u not in visited:
                visited.add(u)                
                for child in u._prev:
                    build_topo(child)
                topo.append(u)    
        build_topo(self)
        self.grad=1.0
        for node in reversed(topo):
            node._backward()
    