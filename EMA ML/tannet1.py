import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from pytorch_tabnet.tab_model import TabNetClassifier
import torch

# 1. Load Data
train_df = pd.read_csv(r"D:\AlgoT\trades.csv")
test_df = pd.read_csv(r"D:\AlgoT\tradesN50OUTSAMPLE.csv")

# Drop NaNs
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# 2. Separate features and target
X_train = train_df.drop(columns=["Y"])
y_train = train_df["Y"]

X_test = test_df.drop(columns=["Y"])
y_test = test_df["Y"]

# 3. Encode labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# 4. Normalize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Convert to numpy
X_train_np = X_train_scaled.astype(np.float32)
X_test_np = X_test_scaled.astype(np.float32)

y_train_np = y_train_enc.astype(np.int64)
y_test_np = y_test_enc.astype(np.int64)

# 6. Train TabNet
clf = TabNetClassifier(
    n_d=32,  # Smaller to prevent overfitting
    n_a=32,
    n_steps=5,  # Reduce steps
    gamma=1.3,  # Less aggressive feature selection
    lambda_sparse=1e-3,  # Stronger sparsity
    optimizer_params=dict(lr=0.02),  # Higher LR for faster convergence
    scheduler_params={"step_size": 5, "gamma": 0.95},  # More frequent LR decay

)

clf.fit(
    X_train=X_train_np, y_train=y_train_np,
    eval_set=[(X_test_np, y_test_np)],
    eval_name=["test"],
    eval_metric=["accuracy"],
    max_epochs=100,
    patience=10,
    batch_size=512,  # Smaller batches for stability
    virtual_batch_size=64,
)

# 7. Predict & Evaluate
preds = clf.predict(X_test_np)
print("\n🔍 Model: TabNet")
print(f"Accuracy: {accuracy_score(y_test_np, preds):.4f}")
print("\nClassification Report:")

print(classification_report(y_test_np, preds, target_names=[str(cls) for cls in le.classes_]))

print("Confusion Matrix:")
print(confusion_matrix(y_test_np, preds))
