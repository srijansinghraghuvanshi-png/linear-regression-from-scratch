# Linear Regression from Scratch

A beginner-friendly machine-learning project that implements **linear regression using NumPy and gradient descent**. The goal is to understand the mathematics behind a model instead of using scikit-learn or another ML library.

## What this project demonstrates

- The linear model: y_hat = w*x + b
- Mean squared error (MSE) as the loss function
- Gradients for the weight and bias
- Gradient descent updates
- Making predictions and evaluating training error
- Basic input validation and unit tests

## Project structure

```text
linear-regression-from-scratch/
├── linear_regression.py
├── demo.py
├── test_linear_regression.py
├── requirements.txt
└── README.md
```

## Run it

Python 3.10+ is recommended. From this directory:

```bash
python -m pip install -r requirements.txt
python demo.py
python -m unittest -v
```

On Windows, if `python` is not recognized, try `py` instead.

## The math, in plain language

Our model predicts a value using:

```text
prediction = weight * x + bias
```

For each training example, the error is:

```text
error = prediction - actual_value
```

The mean squared error is the average squared error:

```text
MSE = (1 / n) * sum((prediction - actual_value) ** 2)
```

Gradient descent calculates how the loss changes when the weight or bias changes, then updates both parameters in the direction that reduces the loss:

```text
weight = weight - learning_rate * dMSE/dweight
bias   = bias   - learning_rate * dMSE/dbias
```

In this project, the derivatives are:

```text
dMSE/dweight = (2 / n) * sum((prediction - actual) * x)
dMSE/dbias   = (2 / n) * sum(prediction - actual)
```

## Experiment

The demo creates reproducible synthetic data close to `y = 4x + 7`, adds random noise, and trains the model. The learned values should be close to a weight of 4 and a bias of 7; exact values vary with the generated noise.

## Limitations

This first version handles **one numerical feature** and uses batch gradient descent. It is an educational implementation, not a production-ready ML package. Feature scaling, multiple features, train/test splitting, and visualizing the loss curve are good next improvements.

## Learning note

The model and gradient-descent logic are implemented here rather than delegated to scikit-learn. NumPy is used for numerical arrays and vectorized calculations.

Built as part of my BCA learning journey toward machine learning.
