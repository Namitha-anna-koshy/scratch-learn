from pathlib import Path

content = """# Logistic Regression from Scratch using Gradient Descent

## Overview

This project implements **Logistic Regression from scratch using Gradient Descent**, without relying on machine learning libraries such as Scikit-learn.

The goal is to solve a **binary classification problem**: predicting whether an input belongs to **Class 0** or **Class 1**.

In this example:

- Numbers **less than 10** → Class **0**
- Numbers **greater than or equal to 10** → Class **1**

The model learns a decision boundary that separates the two classes.

---

# Theory Behind Logistic Regression

Logistic Regression is a **supervised learning classification algorithm** used when the output is categorical.

Unlike Linear Regression, which predicts continuous values, Logistic Regression predicts probabilities.

---

## The Logistic Function (Sigmoid Function)

The core of Logistic Regression is the **sigmoid function**:

**σ(z) = 1 / (1 + e^(-z))**

Where:

**z = mx + b**

The sigmoid transforms any real number into a value between **0 and 1**.

This output is interpreted as probability.

### Example:

If:

**σ(z) = 0.85**

Then:

The model predicts an **85% probability** that the input belongs to Class 1.

---

## Decision Boundary

To convert probabilities into class labels:

- If probability ≥ 0.5 → Class **1**
- If probability < 0.5 → Class **0**

This threshold creates the classification boundary.

---

## Loss Function: Binary Cross-Entropy

Since this is classification, Mean Squared Error is not ideal.

Instead, Logistic Regression uses **Binary Cross-Entropy Loss**:

**Loss = -(1/n) Σ [ y log(ŷ) + (1-y) log(1-ŷ) ]**

Where:

- **y** → Actual label
- **ŷ** → Predicted probability
- **n** → Number of samples

This loss heavily penalizes confident incorrect predictions.

Lower loss means better classification performance.

---

## Gradient Descent Optimization

To minimize loss, we compute gradients.

### Gradient with respect to slope (m)

**∂L/∂m = -(1/n) Σ x(y - ŷ)**

### Gradient with respect to intercept (b)

**∂L/∂b = -(1/n) Σ (y - ŷ)**

The update rules are:

**m = m - L × ∂L/∂m**

**b = b - L × ∂L/∂b**

Where:

- **L** → Learning Rate

Repeated updates allow the model to converge to optimal parameters.

---
