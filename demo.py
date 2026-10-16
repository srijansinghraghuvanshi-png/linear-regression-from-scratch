"""Run a small reproducible linear-regression experiment."""
import numpy as np

from linear_regression import LinearRegression


def main() -> None:
    # Synthetic relationship: y is approximately 4x + 7.
    rng = np.random.default_rng(42)
    x = np.linspace(0, 10, 100)
    noise = rng.normal(loc=0.0, scale=2.0, size=x.size)
    y = 4.0 * x + 7.0 + noise

    model = LinearRegression(learning_rate=0.01, n_iterations=2_000)
    model.fit(x, y)

    print("Linear Regression from Scratch")
    print("-" * 34)
    print(f"Learned weight: {model.weight:.3f} (expected near 4)")
    print(f"Learned bias:   {model.bias:.3f} (expected near 7)")
    print(f"Training MSE:   {model.mean_squared_error(x, y):.3f}")
    print(f"First loss:     {model.loss_history[0]:.3f}")
    print(f"Final loss:     {model.loss_history[-1]:.3f}")
    print()
    sample_x = np.array([2.0, 5.0, 8.0])
    for value, prediction in zip(sample_x, model.predict(sample_x)):
        print(f"x={value:.1f} -> predicted y={prediction:.2f}")


if __name__ == "__main__":
    main()
