from pycandle.engine import Value
import random

class Module:
    def parameters(self):
        return []
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad=0.0

class Neuron(Module):
    def __init__ (self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
        self.nonlin=nonlin
    
    def __call__ (self, x):
        # w.T * x + b
        act = sum((wi*xi for wi, xi in zip(self.w,x)),start=self.b)
        return act.tanh() if self.nonlin else act
    
    def parameters(self):
        return self.w + [self.b]



class Layer(Module):
    def __init__ (self, nin, nout, nonlin=True):
        self.neurons=[Neuron(nin,nonlin) for _ in range(nout)]
        
    def __call__ (self,x):
        outs = [n(x) for n in self.neurons]
        return outs   
    
    def __len__(self):
        return len(self.neurons)
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
    
    
class MLP(Module):
    def __init__ (self, nin, nouts: list):
        struc = [nin] + nouts
        self.layers = [Layer(struc[i],struc[i+1], nonlin=(i!=len(nouts)-1)) for i in range(len(nouts))]
        
    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        return x[0] if len(self.layers[-1])==1 else x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]      
    
    