from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb  # If installed
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

# Load your dataset

# Load CSV
df = pd.read_csv(r"D:\AlgoT\trades.csv")  # Replace with your actual file name
df.dropna(inplace=True)

# Split features and target
X_train = df.drop(columns=["Y"])
y_train= df["Y"]


dft = pd.read_csv(r"D:\AlgoT\N50trades.csv")  # Replace with your actual file name
dft.dropna(inplace=True)

# Split features and target
X_test = dft.drop(columns=["Y"])
y_test = dft["Y"]


# Dictionary of models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    # "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

for name, model in models.items():
    print(f"\n🔍 Model: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
