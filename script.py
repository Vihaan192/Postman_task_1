import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

def relu(z):
    return np.maximum(0, z)

def mse(pred, y):
    return np.mean((pred - y) ** 2)

def train(X, y, w1, b1, w2, b2, alpha=0.01, steps=2000):
    samples = X.shape[0]
    history = []
    for i in range(steps):
        z1 = (X @ w1) + b1
        a1 = relu(z1)
        pred = (a1 @ w2) + b2
        
        loss = mse(pred, y)
        history.append(loss)
        
        dpred = (2 / samples) * (pred - y)
        dw2 = (a1.T @ dpred)
        db2 = np.sum(dpred, axis=0, keepdims=True)
        da1 = (dpred @ w2.T)
        dz1 = da1 * (z1 > 0)
        dw1 = (X.T @ dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        w1 -= alpha * dw1
        w2 -= alpha * dw2
        b1 -= alpha * db1
        b2 -= alpha * db2
        
    return history, w1, b1, w2, b2

def predict(X, w1, b1, w2, b2):
    z1 = (X @ w1) + b1
    a1 = relu(z1)
    return (a1 @ w2) + b2

df = sns.load_dataset('tips')

X = df[['total_bill', 'size']].values
y = df[['tip']].values 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = MLPRegressor(
    hidden_layer_sizes=(4,), 
    activation='relu', 
    solver='sgd', 
    learning_rate_init=0.01, 
    max_iter=2000, 
    random_state=42
)
model.fit(X_train_s, y_train.ravel())
sk_preds = model.predict(X_test_s)
sk_mse = mean_squared_error(y_test, sk_preds)
print(f"-> Skikit-Learn Test MSE: {sk_mse:.4f}\n")

np.random.seed(42)
w1 = np.random.randn(2,4) * 0.01
b1 = np.zeros((1, 4))
w2 = np.random.randn(4, 1) * 0.01
b2 = zeros_b2 = np.zeros((1, 1))

history, w1_fin, b1_fin, w2_fin, b2_fin = train(X_train_s, y_train, w1, b1, w2, b2, alpha=0.01, steps=2000)

preds = predict(X_test_s, w1_fin, b1_fin, w2_fin, b2_fin)
c_mse = mean_squared_error(y_test, preds)
print(f"-> Custom NumPy MLP Test MSE: {c_mse:.4f}\n")

my_r2 = r2_score(y_test, preds)
sk_r2 = r2_score(y_test, sk_preds)

print(f"R^2 Scores Comparison -> My Model : {my_r2} vs Scikit Learn : {sk_r2}")

plt.figure(figsize=(9, 5))
plt.plot(history, label='Custom NumPy MLP', color='blue')
plt.plot(model.loss_curve_, label='Scikit-Learn MLPRegressor', color='orange', linestyle='--')
plt.xlabel("Steps")
plt.ylabel("Loss (MSE)")
plt.title("Me vs Scikit ")
plt.legend()
plt.grid(True)
plt.show()