import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load training data
df = pd.read_csv(r"D:\AlgoT\trades.csv")
df.dropna(inplace=True)

X_train = df.drop(columns=["Y"])
y_train = df["Y"]

# Load test data
dft = pd.read_csv(r"D:\AlgoT\tradesN50OUTSAMPLE.csv")
dft.dropna(inplace=True)

X_test = dft.drop(columns=["Y"])
y_test = dft["Y"]

# Encode target labels if they are strings (optional step)
if y_train.dtype == 'object':
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

# Initialize and train HistGradientBoostingClassifier
model = HistGradientBoostingClassifier(
    max_iter=300,        # try 300-1000
    learning_rate=0.03,  # lower = more stable
    max_depth=12,         # try 6–12
    l2_regularization=2,  # try 0.1 to 10
    early_stopping=True,
    random_state=42
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("🔍 Model: HistGradientBoostingClassifier")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importances
try:
    importances = model.feature_importances_
    print("\nFeature Importances:")
    for name, score in zip(X_train.columns, importances):
        print(f"{name:20s}: {score:.4f}")
except AttributeError:
    print("\nFeature importances are not directly available in HistGradientBoostingClassifier.")


plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), 
            annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()