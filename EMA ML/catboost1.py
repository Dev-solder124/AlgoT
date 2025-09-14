import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier

# Load CSV
df = pd.read_csv(r"D:\AlgoT\trades.csv")  # Replace with your actual file name
df.dropna(inplace=True)

# Split features and target
X_train = df.drop(columns=["Y"])
y_train= df["Y"]


dft = pd.read_csv(r"D:\AlgoT\tradesN50OUTSAMPLE.csv")  # Replace with your actual file name
dft.dropna(inplace=True)

# Split features and target
X_test = dft.drop(columns=["Y"])
y_test = dft["Y"]



# Initialize and train CatBoost classifier
model = CatBoostClassifier(verbose=0)  # Silent mode (no training logs)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model
print("🔍 Model: CatBoost")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature Importances
feature_importances = model.feature_importances_
print("\nFeature Importances:")
print(feature_importances)