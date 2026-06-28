# train.py

from engine import Value
from network import MLP


def train(network: MLP, inputs: list[list[float]], targets: list[float], epochs: int, learning_rate: float = 0.1):
    for step in range(epochs):
        predictions = []

        for x in inputs:
            x_values = [Value(number) for number in x]
            prediction = network(x_values)
            predictions.append(prediction)

        loss = Value(0)
        for prediction, target_number in zip(predictions, targets):
            target = Value(target_number)
            loss += (prediction - target) ** 2

        for p in network.parameters():
            p.grad = 0.0

        loss.backward()

        for p in network.parameters():
            p.data -= learning_rate * p.grad

        print("step:", step, "loss:", loss.data)


def predict(network: MLP, inputs: list[list[float]], targets: list[float]):
    for x, y in zip(inputs, targets):
        x_values = [Value(number) for number in x]
        pred = network(x_values)
        print("pred:", pred.data, "target:", y)