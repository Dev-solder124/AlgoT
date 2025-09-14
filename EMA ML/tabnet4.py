import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tabnet.tab_model import TabNetClassifier

# Load data
df_train = pd.read_csv(r'D:\AlgoT\trades.csv')
df_test = pd.read_csv(r'D:\AlgoT\tradesN50OUTSAMPLE.csv')

# Feature engineering: bin hours into categories
def categorize_hour(hour):
    if hour < 12:
        return 'Morning'
    elif hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'

for df in (df_train, df_test):
    df['Time_of_Day'] = df['HOUR'].apply(categorize_hour)

# One-hot encode Time_of_Day and drop HOUR
X_train = pd.get_dummies(df_train.drop(columns=['HOUR', 'target']))
X_test = pd.get_dummies(df_test.drop(columns=['HOUR', 'target']))

# Align columns
X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)

y_train = df_train['target'].values
y_test = df_test['target'].values

# Scale numeric features
one_hot_cols = [col for col in X_train.columns if col.startswith('Time_of_Day_')]
num_cols = [col for col in X_train.select_dtypes(include=[np.number]).columns if col not in one_hot_cols]

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Oversample with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train.values, y_train)

# Compute class weights
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Define Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss
        return focal.mean()

criterion = FocalLoss(weight=class_weights, gamma=2.0)

# Prepare data for TabNet
X_train_np = X_train_res.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)
y_train_np = y_train_res.astype(np.int64)
y_test_np = y_test.astype(np.int64)

# Initialize and train TabNet
clf = TabNetClassifier(
    n_d=32, n_a=32, n_steps=5, gamma=1.5, lambda_sparse=1e-4,
    optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2),
    mask_type='sparsemax',
    loss_fn=criterion,
    verbose=1
)

clf.fit(
    X_train=X_train_np, y_train=y_train_np,
    eval_set=[(X_train_np, y_train_np), (X_test_np, y_test_np)],
    eval_name=['train', 'valid'],
    eval_metric=['accuracy'],
    max_epochs=500, patience=20,
    batch_size=256, virtual_batch_size=64
)

# Evaluation
y_pred = clf.predict(X_test_np)
acc = accuracy_score(y_test_np, y_pred)
print(f"Accuracy: {acc:.4f}\n")
print("Classification report:\n", classification_report(y_test_np, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test_np, y_pred, labels=classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted class")
plt.ylabel("Actual class")
plt.title("Confusion Matrix")
plt.show()
