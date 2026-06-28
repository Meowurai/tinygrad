# demo.py

from network import MLP 
from train import train, predict

network = MLP(3, [4, 4, 1])

inputs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

targets = [1.0, -1.0, -1.0, 1.0]

train(
    network, 
    inputs, 
    targets, 
    epochs=500, 
    learning_rate=0.1,
)

predict(network, inputs, targets)