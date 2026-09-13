Project on a basic MLP built from scratch using only Numpy.

## Features
- The project is about a small feedforward neural network with 2 layers built purely from scratch in Numpy.
- Seaborn dataset 'tips' was used to train and test the model.
- Custom implemented forward pass , backpropagation and manually calculated gradients using the chain rule.
- Z-Score normalization was required considering the parameters of the dataset.


### Prerequisites

1. Ensure you have Python 3 installed.
2. Install the required libraries:
   ```bash
   pip install numpy torch scikit-learn seaborn matplotlib
   ```

###  Running the Code
1. For exploring the various functions and training loops in the jupyter notebook
   ```bash
   jupyter notebook task1.ipynb
   ```
2. To run the correctness harness to compare my model vs Scikit's MLP with a direct comparison of both models' loss curves.   
   ```bash
   py script.py
   ```

### Results 
Seaborn `tips` dataset (80/20 train-test split, 2,000 steps, learning rate = $0.01$):
- **Custom NumPy MLP Test MSE:** `0.6792` ($R^2 = 0.4566$)
- **Scikit-Learn MLPRegressor Test MSE:** `0.6547` ($R^2 = 0.4762$)

## Writeup
Details regarding technical decisions / mistakes and further improvements in this project may be found in writeup.md 
