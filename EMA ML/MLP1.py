import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import time

# Step 1: Load and clean data
df = pd.read_csv(r"D:\AlgoT\trades.csv")
df_cleaned = df.dropna()

# Step 2: Separate features and target
X = df_cleaned.drop(columns=['Y'])
y = df_cleaned['Y']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Neural Network Model
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=300, random_state=42)

# Step 6: Training
start_time = time.time()
mlp.fit(X_train_scaled, y_train)
training_time = time.time() - start_time

# Step 7: Evaluation
y_pred = mlp.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Output results
print(f"Training time: {training_time:.2f} seconds")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)
