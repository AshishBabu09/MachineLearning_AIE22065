import numpy as np
import matplotlib.pyplot as plt

def bipolar_step(x):
    return np.where(x >= 0, 1, -1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def perceptron(x, w, b, activation):
    # Calculate the dot product of the input and weights, add the bias, and apply the activation function.
    return activation(np.dot(x, w) + b)

def train_perceptron(activation):
    # Initialize the weights and bias with the correct number of weights.
    w = np.array([10, 0.2])  # Two weights corresponding to the two inputs.
    b = 0

    # Define the learning rate.
    learning_rate = 0.05

    # Define the training data (inputs and expected outputs for the AND gate).
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    expected_outputs = np.array([0, 0, 0, 1])

    # Initialize lists to store errors and epochs.
    errors = []
    epochs = []

    # Train the perceptron using a loop.
    for epoch in range(1000):
        epoch_error = 0  # Initialize error for the epoch

        # Iterate over the training data.
        for i in range(len(inputs)):
            # Calculate the actual output of the perceptron.
            actual_output = perceptron(inputs[i], w, b, activation)

            # Calculate the error between the actual and expected output.
            error = expected_outputs[i] - actual_output

            # Accumulate the error for the epoch
            epoch_error += abs(error)

            # Update the weights and bias using the learning rate and the error.
            w += learning_rate * error * inputs[i]
            b += learning_rate * error

        # Store the total error for the epoch
        errors.append(epoch_error)

        # Add the current epoch to the epochs list.
        epochs.append(epoch)

        # Check if the convergence criterion is met.
        if epoch_error <= 0.002:
            break

    return w, b, errors, epochs, epoch + 1

# Train perceptron with Bipolar Step function
w_bipolar, b_bipolar, errors_bipolar, epochs_bipolar, iterations_bipolar = train_perceptron(bipolar_step)

# Train perceptron with Sigmoid function
w_sigmoid, b_sigmoid, errors_sigmoid, epochs_sigmoid, iterations_sigmoid = train_perceptron(sigmoid)

# Train perceptron with ReLU function
w_relu, b_relu, errors_relu, epochs_relu, iterations_relu = train_perceptron(relu)

# Print the number of iterations taken to converge for each activation function
print("Iterations taken to converge (Bipolar Step):", iterations_bipolar)
print("Iterations taken to converge (Sigmoid):", iterations_sigmoid)
print("Iterations taken to converge (ReLU):", iterations_relu)

# Plot the errors vs. epochs for each activation function
plt.figure(figsize=(10, 6))
plt.plot(epochs_bipolar, errors_bipolar, label="Bipolar Step")
plt.plot(epochs_sigmoid, errors_sigmoid, label="Sigmoid")
plt.plot(epochs_relu, errors_relu, label="ReLU")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.title("Error vs. Epochs for Different Activation Functions")
plt.legend()
plt.grid(True)
plt.show()
