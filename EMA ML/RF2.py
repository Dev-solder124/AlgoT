import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import cross_val_score

# Load training data
print("reading data")
df = pd.read_csv(r"D:\AlgoT\trades.csv")
df.dropna(inplace=True)
X_train = df.drop(columns=["Y"])
y_train = df["Y"]

# Load testing data
dft = pd.read_csv(r"D:\AlgoT\tradesN50OUTSAMPLE.csv")
dft.dropna(inplace=True)
X_test = dft.drop(columns=["Y"])
y_test = dft["Y"]

# Train default Random Forest
rf = RandomForestClassifier(
    n_estimators=500,        # Increased from default 100
    min_samples_split=5,     # Slightly more conservative than default 2
    max_features='sqrt',      # Default for classification
    random_state=42,
    max_depth=25,
    n_jobs=-1,
    verbose=1                # Show training progress
)
  # Default params
print("training model")
rf.fit(X_train, y_train)

# Predict and evaluate
print("testing")
y_pred = rf.predict(X_test)
print("\n✅ Evaluation on Test Set")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature Importances
importances = rf.feature_importances_
feature_names = X_train.columns

# Create DataFrame for easy sorting and plotting
feat_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Print top 10 features
print("\n🔍 Top 10 Important Features:")
print(feat_df.head(10))

# # Compare train vs test performance
# train_score = rf.score(X_train, y_train)
# test_score = rf.score(X_test, y_test)
# print(f"Train Accuracy: {train_score:.4f}\nTest Accuracy: {test_score:.4f}")

# Check cross-validation consistency
# cv_scores = cross_val_score(rf, X_train, y_train, cv=100)
# print(f"CV Scores: {cv_scores}\nMean CV: {np.mean(cv_scores):.4f}")

# Plot
# plt.figure(figsize=(10, 6))
feat_df.head(30).plot(kind='barh', x='Feature', y='Importance', legend=False)
plt.gca().invert_yaxis()
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.tight_layout()


plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), 
            annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
