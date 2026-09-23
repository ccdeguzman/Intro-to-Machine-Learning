"""
Name: Christian de Guzman
HW3: perceptron_scratch.py
"""
# Loading creditcard.csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import time

df = pd.read_csv("creditcard.csv")
X = df.drop(columns =["Class", "Time"]).values # 284807 x 30
y = df["Class"].values # 284807

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# print(X_train, X_test, y_train, y_test)

# Scratch Perceptron Class
class ScratchPerceptron:
    def __init__ (self , lr =0.01 , n_epochs =50) :
        self.lr = lr
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
        self.errors_ = [] # misclassifications per epoch
        
    def net_input(self, X):
        return X @ self.weights + self.bias

    def predict(self, X):
        """ Return predicted labels for input array X."""
        linear_output = X@self.weights+self.bias
        return(linear_output >= 0).astype(int)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        t0 = time.time()
        for epoch in range(self.n_epochs):
            # Shuffle training indices at the start of each epoch
            indices = np.random.permutation(n_samples)
            epoch_errors = 0
            
            for i in indices:
                xi = X[i]
                yi = y[i]
                y_hat = int ((np.dot(self.weights, xi) + self.bias) >= 0)
                
                if y_hat != yi:         # misclassified
                    delta = yi - y_hat  # + 1 or -1
                    # weight update rule
                    self.weights += self.lr * delta * xi 
                    self.bias += self.lr * delta
                    epoch_errors += 1
            self.errors_.append(epoch_errors) # Store the number of misclassifications per epoch
            print(f"Epoch time: {time.time()-t0:.2f}s")

# Training the model
model = ScratchPerceptron(lr=0.01, n_epochs=50)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

y_test_scores = model.net_input(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
test_recall = recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)
test_auprc = average_precision_score(y_test, y_test_scores)

print("Training Accuracy: ", train_accuracy)
print("Test Accuracy: ", test_accuracy)
print("Test Precision: ", test_precision)
print("Test Recall: ", test_recall)
print("Test F1-score: ", test_f1)
print("Test AUPRC: ", test_auprc)

plt.figure()
plt.plot(range(1, len(model.errors_) + 1), model.errors_, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Misclassifications")
plt.title("Scratch Perceptron: Misclassifications per Epoch")
plt.grid(True)
plt.savefig("convergence.png", dpi=150)
plt.show()

precision, recall, _ = precision_recall_curve(
    y_test, X_test @ model.weights + model.bias
)

"""
plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision Recall Curve (Scratch Perceptron)")
plt.grid(True)
plt.savefig("pr_curve.png", dpi=150)
plt.show()
"""
feature_names = df.drop(columns=["Class"]).columns.tolist()  # V1-V28 + Amount
top5_scratch = np.argsort(np.abs(model.weights))[-5:][::-1]
print("Scratch top 5:", [feature_names[i] for i in top5_scratch])