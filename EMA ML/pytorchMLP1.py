import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import time

# =====================
# 1. Data Preprocessing
# =====================
df = pd.read_csv("D:/AlgoT/trades.csv")
df.dropna(inplace=True)

X = df.drop(columns=['Y']).values
y = df['Y'].astype(int).values

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Dataloaders
train_ds = TensorDataset(X_train_tensor, y_train_tensor)
test_ds = TensorDataset(X_test_tensor, y_test_tensor)
train_dl = DataLoader(train_ds, batch_size=512, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=1024)

# =====================
# 2. Model Definition
# =====================
class DeepTabularModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# =====================
# 3. Training Setup
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepTabularModel(input_dim=X.shape[1], output_dim=len(np.unique(y))).to(device)

# Handle class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# =====================
# 4. Training Loop
# =====================
num_epochs = 800  # Approx. 1.5–2 hours
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_dl)
    scheduler.step(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Early stopping condition (optional)
    if time.time() - start_time > 7100:  # ~1hr 58 mins
        print("⏳ Stopping due to time limit.")
        break

# =====================
# 5. Evaluation
# =====================
model.eval()
with torch.no_grad():
    preds = model(X_test_tensor.to(device))
    pred_labels = torch.argmax(preds, dim=1).cpu().numpy()

print("\nClassification Report:")
print(classification_report(y_test, pred_labels, digits=4))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Compute and plot confusion matrix
cm = confusion_matrix(y_test, pred_labels)
labels = sorted(np.unique(y_test))
print(cm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
