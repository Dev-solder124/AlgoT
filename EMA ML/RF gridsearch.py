# ================ COMPLETE RANDOM FOREST OPTIMIZATION PIPELINE ================
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ====================== 1. DATA LOADING & PREPROCESSING ======================
print("🔄 Loading and preprocessing data...")

# Load your datasets (replace with your actual paths)
train_df = pd.read_csv(r"D:\AlgoT\trades.csv")
test_df = pd.read_csv(r"D:\AlgoT\N50trades.csv")

# Preprocessing function
def preprocess_data(df):
    df = df.copy()
    # Handle missing/infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

# Apply preprocessing
train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)

# Split features and target
X_train = train_df.drop(columns=["Y"])
y_train = train_df["Y"]
X_test = test_df.drop(columns=["Y"])
y_test = test_df["Y"]

# ====================== 2. FEATURE SELECTION ======================
print("\n🔍 Performing feature selection...")

# Initial model for feature importance
base_rf = RandomForestClassifier(n_estimators=100, random_state=42)
base_rf.fit(X_train, y_train)

# Select important features (top 50%)
selector = SelectFromModel(base_rf, threshold="median")
X_train_sel = selector.transform(X_train)
X_test_sel = selector.transform(X_test)

print(f"Reduced features from {X_train.shape[1]} to {X_train_sel.shape[1]}")

# ====================== 3. MODEL TRAINING WITH BEST PARAMS ======================
print("\n🚀 Training model with optimized parameters...")

# Best parameters from your grid search analysis
best_params = {
    'n_estimators': 200,
    'max_features': 'sqrt',
    'min_samples_split': 10,
    'max_depth': None,
    'min_samples_leaf': 1,
    'class_weight': None
}

# Initialize and train model
final_rf = RandomForestClassifier(
    **best_params,
    random_state=42,
    n_jobs=-1  # Use all cores
)
final_rf.fit(X_train_sel, y_train)

# ====================== 4. EVALUATION ======================
print("\n📊 Evaluating model performance...")

# Predictions
y_pred = final_rf.predict(X_test_sel)

# Metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), 
            annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ====================== 5. FEATURE IMPORTANCE ANALYSIS ======================
print("\n📈 Analyzing feature importance...")

# Get feature names after selection
selected_features = X_train.columns[selector.get_support()]

# Plot importance
importances = final_rf.feature_importances_
importance_df = pd.DataFrame({
    'Feature': selected_features,
    'Importance': importances
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
importance_df.head(20).plot.barh(x='Feature', y='Importance')
plt.gca().invert_yaxis()
plt.title('Top 20 Important Features')
plt.tight_layout()
plt.show()

# ====================== 6. (OPTIONAL) FINAL TUNING ======================
print("\n🎛️ Performing final parameter refinement...")

# Narrowed parameter space around best values
param_dist = {
    'n_estimators': [150, 200, 250],
    'max_features': ['sqrt', 0.7],
    'min_samples_split': [8, 10, 12],
    'max_depth': [None, 30, 50]
}

# Quick randomized search
quick_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=15,  # Test only 15 combinations
    cv=3,       # Faster 3-fold CV
    n_jobs=-1,
    random_state=42
)
quick_search.fit(X_train_sel, y_train)

print("🔍 Best refined parameters:", quick_search.best_params_)

# ====================== 7. SAVE FINAL MODEL ======================
print("\n💾 Saving final model...")
joblib.dump(final_rf, 'optimized_random_forest.pkl')
print("✅ Pipeline completed! Model saved as 'optimized_random_forest.pkl'")

# ====================== USAGE EXAMPLE ======================
# To load and use the model later:
# loaded_model = joblib.load('optimized_random_forest.pkl')
# predictions = loaded_model.predict(new_data)