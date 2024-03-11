import numpy as np
import matplotlib.pyplot as plt

# Activation Functions
def bipolar_step_activation(x):
    # Bipolar step activation function
    return 1 if x >= 0 else -1

def sigmoid_activation(x):
    # Sigmoid activation function
    return 1 / (1 + np.exp(-x))

def relu_activation(x):
    # ReLU (Rectified Linear Unit) activation function
    return max(0, x)

# Perceptron Function
def perceptron(inputs, weights, activation_fn):
    # Calculate the weighted sum of inputs
    weighted_sum = np.dot(inputs, weights[1:]) + weights[0]
    # Apply the activation function
    return activation_fn(weighted_sum)

# Weight Initialization
def initialize_weights(num_inputs):
    # Initialize weights with random values
    return np.random.uniform(-0.5, 0.5, size=num_inputs + 1)

# Training
def train_perceptron(training_inputs, training_outputs, weights, activation_fn, learning_rate, convergence_error=0.002, max_epochs=1000):
    errors = []
    epochs = 0
    while True:
        total_error = 0
        # Iterate through training data
        for inputs, target in zip(training_inputs, training_outputs):
            # Forward pass
            prediction = perceptron(inputs, weights, activation_fn)
            # Calculate error
            error = target - prediction
            total_error += error ** 2
            # Update weights
            weights[1:] += learning_rate * error * inputs
            weights[0] += learning_rate * error
        # Track error over epochs
        errors.append(total_error)
        epochs += 1
        # Check for convergence
        if total_error <= convergence_error or epochs >= max_epochs:
            break
    return errors, epochs

# Error Plotting
def plot_error(errors, epochs, activation):
    # Plot epochs vs error
    plt.plot(range(1, epochs + 1), errors)
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title(f'Epochs vs Error ({activation} Activation)')
    plt.show()

def main():
    # Define training data
    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    training_outputs = np.array([0, 1, 1, 0])

    # Initialize weights and set learning rate
    initial_weights = initialize_weights(training_inputs.shape[1])
    learning_rate = 0.05

    # Training with Bi-Polar Step Activation
    errors_bipolar, num_epochs_bipolar = train_perceptron(training_inputs, training_outputs, initial_weights.copy(), bipolar_step_activation, learning_rate)
    plot_error(errors_bipolar, num_epochs_bipolar, 'Bi-Polar Step')
    print("Converged with Bi-Polar Step activation in", num_epochs_bipolar, "epochs.")

    # Training with Sigmoid Activation
    errors_sigmoid, num_epochs_sigmoid = train_perceptron(training_inputs, training_outputs, initial_weights.copy(), sigmoid_activation, learning_rate)
    plot_error(errors_sigmoid, num_epochs_sigmoid, 'Sigmoid')
    print("Converged with Sigmoid activation in", num_epochs_sigmoid, "epochs.")

    # Training with ReLU Activation
    errors_relu, num_epochs_relu = train_perceptron(training_inputs, training_outputs, initial_weights.copy(), relu_activation, learning_rate)
    plot_error(errors_relu, num_epochs_relu, 'ReLU')
    print("Converged with ReLU activation in", num_epochs_relu, "epochs.")

if __name__ == "__main__":
    main()
