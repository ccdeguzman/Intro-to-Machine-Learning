"""
Name: Christian de Guzman
HW3: perceptron_sklearn.py
"""
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import numpy as np
import time 

df = pd.read_csv("creditcard.csv")
X = df.drop(columns =["Class", "Time"]).values # 284807 x 30
y = df["Class"].values # 284807

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = Perceptron (
    eta0=0.01, # learning rate ( match Part A)
    max_iter=50, # epochs ( match Part A)
    tol=None, # disable early stopping
    random_state=42,
    shuffle=True
)
clf.fit(X_train, y_train)

#Predictions
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

y_test_scores = X_test@clf.coef_[0] + clf.intercept_[0]

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

# Re-train the sklearn Perceptron with class weight={0:1, 1:578} to compensate for the imbalance.
clf_w = Perceptron(
    eta0=0.01, 
    max_iter=50, 
    tol=None, 
    random_state=42,
    shuffle=True,
    class_weight={0:1, 1:578}
)
clf_w.fit(X_train, y_train)

# Predictions
y_train_pred_w = clf_w.predict(X_train)
y_test_pred_w = clf_w.predict(X_test)

# Scores for AUPRC
y_test_scores_w = X_test@clf_w.coef_[0] + clf_w.intercept_[0]

print("     Perceptron with class weight")
train_accuracy_w = accuracy_score(y_train, y_train_pred_w)
test_accuracy_w = accuracy_score(y_test, y_test_pred_w)
test_precision_w = precision_score(y_test, y_test_pred_w, pos_label=1, zero_division=0)
test_recall_w = recall_score(y_test, y_test_pred_w, pos_label=1, zero_division=0)
test_f1_w = f1_score(y_test, y_test_pred_w, pos_label=1, zero_division=0)
test_auprc_w = average_precision_score(y_test, y_test_scores_w)

print("Training Accuracy: ", train_accuracy_w)
print("Test Accuracy: ", test_accuracy_w)
print("Test Precision: ", test_precision_w)
print("Test Recall: ", test_recall_w)
print("Test F1-score: ", test_f1_w)
print("Test AUPRC: ", test_auprc_w)

t0 = time.time()
clf.fit(X_train, y_train)
print(f"Sklearn total training time: {time.time()-t0:.2f}s")

feature_names = df.drop(columns=["Class"]).columns.tolist() 
top5_sk = np.argsort(np.abs(clf.coef_[0]))[-5:][::-1]
print("Sklearn top 5:", [feature_names[i] for i in top5_sk])