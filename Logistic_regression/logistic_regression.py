import math
import matplotlib.pyplot as plt

# Sample binary classification dataset (you can replace with your own)
# Let's classify numbers >= 10 as class 1, others as class 0
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
     11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

y = [0 if xi < 10 else 1 for xi in x]

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# Binary cross entropy loss
def compute_loss(m, b, x, y):
    total_loss = 0
    n = len(x)
    for i in range(n):
        z = m * x[i] + b
        y_pred = sigmoid(z)
        total_loss += -(y[i] * math.log(y_pred + 1e-10) + (1 - y[i]) * math.log(1 - y_pred + 1e-10))
    return total_loss / n

# Gradient Descent
def gradient_descent(m_now, b_now, x, y, L):
    m_gradient = 0
    b_gradient = 0
    n = len(x)

    for i in range(n):
        z = m_now * x[i] + b_now
        y_pred = sigmoid(z)

        # Gradients for logistic regression
        m_gradient += -(1/n) * x[i] * (y[i] - y_pred)
        b_gradient += -(1/n) * (y[i] - y_pred)

    m = m_now - L * m_gradient
    b = b_now - L * b_gradient

    return m, b

# Train the model
m = 0
b = 0
L = 0.1
epochs = 2000

for epoch in range(epochs):
    m, b = gradient_descent(m, b, x, y, L)
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss = {compute_loss(m, b, x, y):.4f}")

print("\nFinal Model:")
print("m =", m)
print("b =", b)

# Predictions
predicted_prob = [sigmoid(m * xi + b) for xi in x]
predicted_class = [1 if p >= 0.5 else 0 for p in predicted_prob]

print("\nPredicted classes:")
print(predicted_class)

# Visualization
plt.scatter(x, y, color="black", label="Actual classes")
plt.plot(x, predicted_prob, color="red", label="Predicted probability (sigmoid)")
plt.axhline(0.5, color="gray", linestyle="--")
plt.xlabel("x")
plt.ylabel("Predicted Probability")
plt.title("Logistic Regression from Scratch")
plt.grid()
plt.legend()
plt.show()
