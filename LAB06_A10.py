#10
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler  # Import StandardScaler for data scaling

# Define training data (mapped outputs for 0 and 1)
def get_mapped_data(logic_gate):
  if logic_gate == "AND":
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[1, 0], [1, 0], [1, 0], [0, 1]])  # 1 for 0, 0 for 1
  elif logic_gate == "XOR":
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])  # 0 for 0, 1 for 1
  else:
    raise ValueError("Invalid logic gate specified. Choose 'AND' or 'XOR'.")
  return inputs, targets

# Train and evaluate the MLPClassifier
def train_and_evaluate(logic_gate):
  inputs, targets = get_mapped_data(logic_gate)

  # Standardize the input data (may improve convergence for XOR)
  scaler = StandardScaler()
  inputs_scaled = scaler.fit_transform(inputs)

  # Create the MLPClassifier with a single hidden layer, ReLU activation, and increased max_iter
  clf = MLPClassifier(solver='lbfgs', alpha=0.01, hidden_layer_sizes=(4,), activation='relu', random_state=1, max_iter=1000)
  clf.fit(inputs_scaled, targets)

  # Make predictions and calculate accuracy
  predictions = clf.predict(inputs_scaled)
  accuracy = np.mean(predictions == targets) * 100

  print(f"** {logic_gate.upper()} Gate Results **")
  print(f"Accuracy: {accuracy:.2f}%")

# Run for both AND and XOR gates
train_and_evaluate("AND")
train_and_evaluate("XOR")
