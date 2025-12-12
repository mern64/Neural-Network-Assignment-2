#STINK3014 Neural Networks
#Baseline code - Multilayer Perceptron for assignment #2

import math
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Initialization
# -----------------------------
X1, X2 = 1.1, 0.95
target = 1

# Initial weights
w1 = 0.7   # X1 --> Hidden
w2 = 0.5   # X2 --> Hidden
w3 = 0.4   # Hidden --> Output

# Learning parameters
alpha = 0.1   # learning rate
beta = 0.5 # momentum rate
epochs = 500   # number of training iterations

# Initialize previous weight updates (momentum)
delta_w1_prev = 0
delta_w2_prev = 0
delta_w3_prev = 0

# Activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

# Store errors and weights for plotting
errors = []
w1_list = []
w2_list = []
w3_list = []

print("-" * 50)
print("Multilayer Perceptron with Backpropagation")
print("-" * 50)

# -----------------------------
# Step 2: Training Loop
# -----------------------------
for epoch in range(1, epochs + 1):

    # ---- Forward Pass ----
    net_h = X1 * w1 + X2 * w2
    h = sigmoid(net_h)

    net_o = h * w3
    y = sigmoid(net_o)

    # ---- Error ----
    error = target - y
    errors.append(abs(error))

    # ---- Backpropagation ----
    delta_o = error * sigmoid_derivative(y)
    delta_h = delta_o * w3 * sigmoid_derivative(h)

    # ---- Weight Updates (with Momentum) ----
    delta_w3 = alpha * delta_o * h + beta * delta_w3_prev
    delta_w1 = alpha * delta_h * X1 + beta * delta_w1_prev
    delta_w2 = alpha * delta_h * X2 + beta * delta_w2_prev

    # Update weights
    w3 += delta_w3
    w1 += delta_w1
    w2 += delta_w2

    # Store current deltas for next iteration
    delta_w3_prev = delta_w3
    delta_w1_prev = delta_w1
    delta_w2_prev = delta_w2

    # Save weights for plotting
    w1_list.append(w1)
    w2_list.append(w2)
    w3_list.append(w3)

    # ---- Display progress ----
    print(f"Epoch {epoch:2d} | Output: {y:.4f} | Error: {error:.4f} | w1: {w1:.4f} | w2: {w2:.4f} | w3: {w3:.4f}")

# -----------------------------
# Step 3: Plot Error Convergence
# -----------------------------
plt.figure(figsize=(10,4))
plt.plot(range(1, epochs + 1), errors, marker='o', color='red')
plt.title("Error Convergence over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Absolute Error")
plt.grid(True)
plt.show()

# -----------------------------
# Step 4: Plot Weights Over Time
# -----------------------------
plt.figure(figsize=(10,4))
plt.plot(range(1, epochs + 1), w1_list, label='w1')
plt.plot(range(1, epochs + 1), w2_list, label='w2')
plt.plot(range(1, epochs + 1), w3_list, label='w3')
plt.title("Weight Changes Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Weight Value")
plt.legend()
plt.grid(True)
plt.show()
