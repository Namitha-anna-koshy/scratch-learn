## Theory Behind the Code (aka what I told myself I understood)

This project builds a **Linear Regression model** using **Gradient Descent**, from scratch.  

The idea? Fit a line to data so it *looks like* you know what you’re doing:

#### y = mx + b

Where:
- `m` is the slope (the “how steep am I spiraling” factor)
- `b` is the y-intercept (aka where the line slaps the y-axis)

---

### The Loss Function: 

We measure how tragically off our predictions are using the **Mean Squared Error** (MSE):

#### MSE = (1/n) * Σ(yi - (mxi + b))²

Where:
- `n` = number of data points
- `yi` = actual value (the truth)
- `mxi + b` = predicted value (our best lie)

Lower MSE = better fit.  
Higher MSE = your model's crying in a corner.

---

### Derivatives: 

Gradient descent needs us to take partial derivatives of the MSE.

#### With respect to `m`:
∂/∂m = -(2/n) * Σ xi * (yi - (mxi + b))

#### With respect to `b`:
∂/∂b = -(2/n) * Σ (yi - (mxi + b))

These tell us:  
> "Hey, your line is off — here's how to un-wreck it."

---

### 🔁 Gradient Descent Update Rule (A.K.A. How We Learn™)

We use those derivatives to slowly nudge our `m` and `b` in the right direction:

m = m - L * ∂/∂m
b = b - L * ∂/∂b

Where `L` is the **learning rate** — how fast we panic-adjust.  
Too high? We overshoot. Too low? We never arrive. 

Do this for a bunch of iterations, and the line becomes *slightly less garbage*.

---

## 📜 What the Code’s Doing (Yes, It's Alive)

```python
x = [...]  # The x-values (inputs, or "what we control")
y = [...]  # The y-values (outputs, or "what life throws back")

# Make it look official with a DataFrame
data = pd.DataFrame({'x': x, 'y': y})

# Starting from rock bottom
m = 0
b = 0
L = 0.0001      # Learning rate: tiny steps to avoid chaos
epochs = 1000   # Aka 1000 chances to get it less wrong

# Begin the loop of emotional growth (also known as "training")
for i in range(epochs):
    m, b = gradient_descent(m, b, data, L)
Inside gradient_descent, we calculate our slopes of despair (gradients),
update m and b, and pray we’re closer to the correct line.
```
### Final Output (Proof of Pain)
After training, the script:

Prints the final values of m and b (slope and intercept)

Plots the original data as black dots of judgment

Draws a red regression line that hopefully makes you feel smart

plt.scatter(x, y)  # Real-life chaos
plt.plot(x, predicted_y)  # Our mathematical best guess
If it looks like it fits? Congrats u just leanrt the concept behind linear regression.

