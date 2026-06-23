import random

from engine import Value

class Neuron:
    def __init__(self, n_inputs: int):
        self.weights = [Value(random.uniform(0, 1)) for _ in range(n_inputs)]
        self.bias = Value(random.uniform(0, 1))

    def __call__(self, inputs: list[Value]):
        assert len(inputs) == len(self.weights)
        weighted_sum = Value(0)
        for w, x in zip(self.weights, inputs):
            weighted_sum += w * x

        return (weighted_sum + self.bias).tanh()
    
    def parameters(self) -> list[Value]:
        return self.weights + [self.bias]
    

class Layer:
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        self.neurons = [Neuron(n_inputs) for _ in range(n_neurons)]

    def __call__(self, inputs: list[Value]) -> list[Value]:
        return [neuron(inputs) for neuron in self.neurons]
    
    def parameters(self) -> list[Value]:
        params = []
        for neuron in self.neurons:
            params += neuron.parameters()
        return params
    
class MLP:
    def __init__(self, n_inputs: int, layer_sizes: list[int]) -> None:
        assert layer_sizes[-1] == 1
        sizes = [n_inputs] + layer_sizes
        self.layers = [
            Layer(sizes[i], sizes[i + 1])
            for i in range(len(layer_sizes))
        ]

    def __call__(self, inputs: list[Value]) -> Value:
        for layer in self.layers:
            inputs = layer(inputs)

        assert len(inputs) == 1
        return inputs[0]

    def parameters(self) -> list[Value]:
        params = []
        for layer in self.layers:
            params += layer.parameters()

        return params