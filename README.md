# Pycandle

A tiny neural network library built from scratch in pure Python, to learn how backpropagation works.

- **Autograd engine**: a scalar `Value` class that records operations and computes gradients with `backward()`.
- **Neural network**: `Neuron`, `Layer` and `MLP` built on top of the engine.
- **Training**: plain batch gradient descent with a sum-of-squared-errors loss.

## Project structure

```
Build-NN-From-Scratch
├──README.md
├──src/pycandle/
	├── notebook.ipynb
	├── engine.py       # Value: operations + backward()
	├── nn.py           # Module, Neuron, Layer, MLP
	├── GD.py           # GD (gradient descent), accuracy
	├── data_prep.py    # CSV loading, train/test split
	├── graph_viz.py    # draw the computation graph (graphviz)
	├── data/           # moons.csv
	└── example/
	    ├── train_moon.py
	    └── visualize_graph.py
├──.python-version
├──pyproject.toml
├──uv.lock	    
    
```

## Setup

```bash
uv sync
```

`graphviz` is only needed for `graph_viz.py` (it also requires the Graphviz program installed on your system).

## Quick start

```python
from pycandle.nn import MLP
from pycandle.GD import GD, accuracy
from pycandle.data_prep import load_moon, train_test_split

X, y = load_moon()
X_train, y_train, X_test, y_test = train_test_split(X, y)

model = MLP(2, [6, 1])          # 2 inputs, 6 hidden neurons, 1 output
GD(X_train, y_train, model, epochs=100, lr=0.002)

print(accuracy(X_test, y_test, model))
```

Or run the full example:

```bash
uv run python3 src/pycandle/example/train_moon.py
```

## Notes

- Labels are `+1` / `-1`; a prediction is positive when the output is `> 0`.
- Hidden layers use `tanh`; the output layer is linear.
- The loss is a **sum** over samples, so gradients are large. Keep the learning rate small (about `0.002`), otherwise the loss diverges. 