import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report
from xgboost import XGBClassifier, callback
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.impute import SimpleImputer

# 1. Load and preprocess data
print("1. Loading and preprocessing data...")
df = pd.read_csv(r"D:\AlgoT\trades.csv")

# Fill NaN in target column
if df['Y'].isna().sum() > 0:
    median_target = df['Y'].median()
    df['Y'] = df['Y'].fillna(median_target)
    print(f"Warning: Filled {df['Y'].isna().sum()} NaN values in target with median")
else:
    print(f"Warning: Filled 0 NaN values in target with median")

X = df.drop(columns=['Y'])
y = df['Y']

# Fill NaNs in features
if X.isna().sum().sum() > 0:
    print(f"Imputing {X.isna().sum().sum()} missing values in features with median...")
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
else:
    print("No missing values in features.")

# 2. Scale features
print("2. Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split data
print("3. Splitting data...")
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.37, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.55, random_state=42, stratify=y_temp)

print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

# 4. Feature Selection
print("4. Selecting top features...")
k = min(40, X_train.shape[1])
selector = SelectKBest(score_func=f_classif, k=k)
X_train_selected = selector.fit_transform(X_train, y_train)
X_val_selected = selector.transform(X_val)
X_test_selected = selector.transform(X_test)

print(f"Selected {X_train_selected.shape[1]} features out of {X.shape[1]}")

# 5. Train Models
print("5. Training model...")

# Compute class weights
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# XGBoost
print("\nTraining XGBoost with class weights...")
xgb = XGBClassifier(
    objective='multi:softprob',
    num_class=len(np.unique(y_train)),
    learning_rate=0.05,
    n_estimators=200,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    verbosity=1,
    random_state=42
)

xgb.fit(
    X_train_selected,
    y_train,
    sample_weight=sample_weights,
    eval_set=[(X_val_selected, y_val)],
    callbacks=[callback.EarlyStopping(rounds=30)],
    verbose=10
)

# LightGBM
print("\nTraining LightGBM with default objective...")
lgbm = LGBMClassifier(
    objective='multiclass',
    num_class=len(np.unique(y_train)),
    learning_rate=0.05,
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)

lgbm.fit(
    X_train_selected,
    y_train,
    eval_set=[(X_val_selected, y_val)],
    eval_metric='multi_logloss',
    callbacks=[lambda env: callback.EarlyStopping(rounds=30)(env)]
)

# Logistic Regression (baseline)
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_selected, y_train)

# Ensemble
print("\nCreating optimized ensemble...")
ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb),
        ('lgbm', lgbm),
        ('lr', lr)
    ],
    voting='soft'
)

ensemble.fit(X_train_selected, y_train)

# Evaluation
print("\nEvaluating model on test set...")
y_pred = ensemble.predict(X_test_selected)
print(classification_report(y_test, y_pred))
