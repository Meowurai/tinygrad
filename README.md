# Tinygrad

> **Status:** Complete

A small learning lab for building the core ideas behind automatic differentiation and neural networks from scratch.

This project implements a tiny scalar-based autograd engine, then uses it to train a small multi-layer perceptron (MLP). The goal was not performance or completeness, but understanding how forward passes, computational graphs, backpropagation, gradients, and parameter updates fit together.

## Outcome

This project achieved its original goal: understanding how a neural network can be built from simple scalar values and trained using reverse-mode automatic differentiation.

Tinygrad became the foundation for the next project, TinyLM, where the same autograd ideas were reused to train a character-level language model.

## What This Project Builds

Tinygrad consists of four main parts:

- **Value** — a scalar object that tracks data, gradients, parents, and the operation that produced it.
- **Autograd Engine** — builds a computation graph during the forward pass and propagates gradients backward through it.
- **Neural Network Components** — `Neuron`, `Layer`, and `MLP` abstractions built on top of `Value`.
- **Training Loop** — computes predictions, loss, gradients, and parameter updates.

## Architecture

```text
Inputs
    │
    ▼
MLP
    │
    ▼
Prediction
    │
    ▼
Loss
    │
    ▼
Backward Pass
    │
    ▼
Gradients
    │
    ▼
Parameter Update
```

## Autograd Engine

The core abstraction is `Value`.

A `Value` stores:

- `data` — the scalar value computed during the forward pass.
- `grad` — the gradient accumulated during the backward pass.
- `_prev` — the parent values that produced it.
- `_op` — the operation that created it.
- `_backward` — the local gradient rule for that operation.

Supported operations include:

- Addition
- Negation
- Subtraction
- Multiplication
- Power
- `tanh`
- Reverse-mode `backward()`

## Neural Network Components

The project builds a small neural network from the autograd engine.

### Neuron

A neuron owns:

- one weight per input
- one bias

It computes a weighted sum and applies `tanh`.

### Layer

A layer owns multiple neurons and returns one output per neuron.

### MLP

An MLP owns multiple layers and composes them into a full network.

Example:

```python
network = MLP(3, [4, 4, 1])
```

This creates:

```text
3 inputs
    ↓
4 neurons
    ↓
4 neurons
    ↓
1 output
```

## Training

Training follows the same basic loop used in larger neural network frameworks:

```text
1. Run a forward pass
2. Compute loss
3. Zero gradients
4. Run backward pass
5. Update parameters using gradient descent
```

The update rule is:

```python
p.data -= learning_rate * p.grad
```

## Demo

Run:

```bash
python demo.py
```

The demo trains a small MLP on a tiny dataset and prints the loss over time, followed by predictions compared to the target values.

## Key Concepts Learned

- A neural network is a composition of simple mathematical operations.
- Forward pass computes values.
- Backward pass computes influence.
- Gradients must accumulate because a value can influence the loss through multiple paths.
- Backpropagation requires traversing the computation graph in reverse topological order.
- Parameters are just `Value` objects that get updated by gradient descent.
- A model learns when parameter updates reduce the loss over time.

## Conclusion

Tinygrad is intentionally complete.

Its purpose was to understand the foundations of automatic differentiation and neural network training by implementing every core component from scratch.

By the end of the project, the following questions have been answered:

- How are computational graphs constructed during the forward pass?
- How does reverse-mode automatic differentiation compute gradients?
- Why must gradients accumulate?
- How does gradient descent update parameters?
- How do neurons, layers, and multi-layer perceptrons fit together?

This project is intentionally limited to scalar-valued computation and a small multi-layer perceptron. It serves as a complete educational implementation of the core mechanics behind neural network training.
