import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import PowerTransformer

# 1. Data Loading and Cleaning
df = pd.read_csv("D:/AlgoT/trades.csv")
df = df.dropna(subset=['Y']).copy()

# 2. Enhanced Feature Engineering
main_features = ['CDL_BODY', 'HOUR', 'ATR', 'dBBB', 'POSITION']
for col in main_features:
    pt = PowerTransformer()
    df[f'{col}_power'] = pt.fit_transform(df[[col]])

# Class-specific features
df['ATR_to_HOUR'] = df['ATR'] / (df['HOUR'] + 1)
df['CDL_BODY_squared'] = df['CDL_BODY'] ** 2

# 3. Class Weighting
class_weights = {0: 1.2, 1: 0.8, 2: 2.5}  # Boost Class 2

# 4. Data Preparation
X = df[main_features + ['ATR_to_HOUR', 'CDL_BODY_squared']]
y = df['Y'].astype(int)

# 5. Optimized Model Configuration
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.025,
    depth=5,
    l2_leaf_reg=10,
    class_weights=class_weights,
    loss_function='MultiClassOneVsAll',
    eval_metric='Accuracy',  # Changed from Recall
    custom_metric=['Recall', 'Precision'],  # Track class-wise metrics
    early_stopping_rounds=20,
    leaf_estimation_iterations=10,
    bootstrap_type='Bayesian',
    random_strength=0.5,
    verbose=100
)

# 6. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 7. Training
train_pool = Pool(X_train, y_train)
test_pool = Pool(X_test, y_test)

model.fit(
    train_pool,
    eval_set=test_pool,
    plot=True
)

# 8. Evaluation
preds = model.predict(X_test)
print(classification_report(y_test, preds, digits=4))

# 9. Class-Specific Analysis
print("\nClass 2 Metrics:")
class_report = classification_report(y_test, preds, output_dict=True)
print(f"Recall: {class_report['2']['recall']:.4f}")
print(f"Precision: {class_report['2']['precision']:.4f}")
print(f"F1: {class_report['2']['f1-score']:.4f}")

# 10. Save Model
model.save_model('optimized_model.cbm')