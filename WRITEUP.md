# Detailed Technical writeup on a simple MLP

## Overview

This repository is an introductory project to learning the basics of machine learning by writing a custom small feedforward neural network (MLP) with one hidden layer and 4 neurons.


### Technical Decisions

1. Only 1 hidden layer with 4 neurons was kept to maintain simplicity while focusing entirely on implementing gradient descents on 2 numpy arrays of weights and biases respectively.

2. Gradients were derived by hand using basics of chain rule.

3. The loss function was taken as MSE (Mean Squared Error) to ensure large deviations are penalized much more heavily than tiny errors.

4. Data pre-processing strategy was Z-Score normalization considering the nature of 'bill' being much larger than the 'size' of the table in the dataset to make the mean 0 and standard deviation 1.

### Math

1. Gradients of the loss function $L$ is taken against variables like prediction , weight and bias to slowly shift these variables so as to minimize the loss function.

* **Loss Function ($L$):** $L = \frac{1}{N} \sum (\text{pred} - y)^2$
* **Hidden Forward Pass:** $Z_1 = X W_1 + b_1$, and $A_1 = \text{ReLU}(Z_1)$
* **Output Forward Pass:** $\text{pred} = A_1 W_2 + b_2$

#### Shapes of Initial Data & Parameters
$$X \in \mathbb{R}^{N \times 2}$$
$$Y \in \mathbb{R}^{N \times 1}$$
$$W_1 \in \mathbb{R}^{2 \times 4}, \quad b_1 \in \mathbb{R}^{1 \times 4}$$
$$W_2 \in \mathbb{R}^{4 \times 1}, \quad b_2 \in \mathbb{R}^{1 \times 1}$$

#### Forward Pass
$$Z_1 = X W_1 + b_1 \quad \rightarrow \quad (N, 4)$$
$$A_1 = \text{ReLU}(Z_1) \quad \rightarrow \quad (N, 4)$$
$$\text{pred} = A_1 W_2 + b_2 \quad \rightarrow \quad (N, 1)$$

#### Backward Pass (Gradients & Shapes)
$$\frac{\partial L}{\partial \text{pred}} = \frac{2}{N}(\text{pred} - Y) \quad \rightarrow \quad (N, 1)$$

$$\frac{\partial L}{\partial W_2} = A_1^T \cdot \frac{\partial L}{\partial \text{pred}} \quad \rightarrow \quad (4, N) \times (N, 1) = (4, 1)$$

$$\frac{\partial L}{\partial b_2} = \sum_{\text{rows}} \frac{\partial L}{\partial \text{pred}} \quad \rightarrow \quad (1, 1)$$

$$\frac{\partial L}{\partial A_1} = \frac{\partial L}{\partial \text{pred}} \cdot W_2^T \quad \rightarrow \quad (N, 1) \times (1, 4) = (N, 4)$$

$$\frac{\partial L}{\partial Z_1} = \frac{\partial L}{\partial A_1} \odot (Z_1 > 0) \quad \rightarrow \quad (N, 4)$$

$$\frac{\partial L}{\partial W_1} = X^T \cdot \frac{\partial L}{\partial Z_1} \quad \rightarrow \quad (2, N) \times (N, 4) = (2, 4)$$

$$\frac{\partial L}{\partial b_1} = \sum_{\text{rows}} \frac{\partial L}{\partial Z_1} \quad \rightarrow \quad (1, 4)$$

