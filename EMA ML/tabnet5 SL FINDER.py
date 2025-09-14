import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import torch
from torch.nn import Module
import torch.nn.functional as F
from pytorch_tabnet.tab_model import TabNetClassifier
import warnings

# -------------------------------
# Custom Precision-Optimized Loss
# -------------------------------
class PrecisionFocalLoss(Module):
    def __init__(self, class_weights, gamma=2.0, alpha=0.8, reduction="mean"):
        super(PrecisionFocalLoss, self).__init__()
        self.class_weights = torch.tensor(class_weights)
        self.gamma = gamma
        self.alpha = alpha  # Higher alpha favors precision
        self.reduction = reduction

    def forward(self, inputs, targets):
        if inputs.device != self.class_weights.device:
            self.class_weights = self.class_weights.to(inputs.device)
            
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        ce_loss = F.nll_loss(log_probs, targets, reduction='none')
        
        # Precision-focused adjustment
        precision_factor = torch.where(
            targets == 2,  # For class 2
            self.alpha * (1 - probs[:, 2]) ** self.gamma,
            (1 - self.alpha) * (1 - probs.gather(1, targets.unsqueeze(1)).squeeze() ** self.gamma)
        )
        
        focal_loss = self.class_weights[targets] * precision_factor * ce_loss
        
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()

# -------------------------------
# TabNet Trainer with Precision Optimization
# -------------------------------
class TabNetPrecisionTrainer:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self._prepare_data()
        self._initialize_model()

    def _prepare_data(self):
        # Load and clean data
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        # Drop rows with NaN
        train_df = train_df.dropna()
        test_df = test_df.dropna()

        # Check target column
        if "Y" not in train_df.columns or "Y" not in test_df.columns:
            raise ValueError("Target column 'Y' missing in data")

        # Separate features and target
        X_train = train_df.drop(columns=["Y"])
        y_train = train_df["Y"]
        X_test = test_df.drop(columns=["Y"])
        y_test = test_df["Y"]

        # Normalize label types
        y_train = y_train.astype(float).astype(int).astype(str)
        y_test = y_test.astype(float).astype(int).astype(str)

        # Get common labels
        common_labels = sorted(list(set(y_train.unique()) & set(y_test.unique())))
        if not common_labels:
            raise ValueError("No common labels between train and test sets")

        # Filter to common labels
        train_mask = y_train.isin(common_labels)
        test_mask = y_test.isin(common_labels)

        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]

        # Encode labels
        self.le = LabelEncoder()
        y_train_enc = self.le.fit_transform(y_train)
        y_test_enc = self.le.transform(y_test)

        # Analyze original class distribution
        original_counts = np.bincount(y_train_enc)
        print("\nOriginal Class Distribution:", original_counts)

        # Determine sampling strategy based on actual counts
        min_count = min(original_counts)
        max_count = max(original_counts)
        
        # Use SMOTE for minority class (class 2) and undersample majority classes
        # We'll use 90% of the smallest class as our target
        target_size = int(min_count * 0.9)
        
        # Create sampling pipeline
        over = SMOTE(sampling_strategy={2: target_size}, random_state=42)
        under = RandomUnderSampler(
            sampling_strategy={
                0: target_size,
                1: target_size
            },
            random_state=42
        )
        
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        
        X_res, y_res = pipeline.fit_resample(X_train, y_train_enc)

        # Standardize
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_res)
        self.X_test_scaled = self.scaler.transform(X_test)
        self.y_train_res = y_res
        self.y_test_enc = y_test_enc

        print("\n✅ Data Prepared:")
        print(f"Training samples: {len(self.X_train_scaled)}")
        print(f"Test samples: {len(self.X_test_scaled)}")
        print(f"Balanced class distribution: {np.bincount(self.y_train_res)}")
        print(f"Classes: {self.le.classes_}")

    def _initialize_model(self):
        # Calculate class weights with precision focus
        class_counts = np.bincount(self.y_train_res)
        class_weights = 1. / (class_counts ** 0.75)  # Softer weighting
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        
        # Moderate weighting for Class 2 to boost precision
        class_weights[2] *= 1.8  # Reduced from 2.0 for better balance

        # Define TabNet with precision-optimized parameters
        self.clf = TabNetClassifier(
            n_d=24,  # Smaller network
            n_a=24,
            n_steps=3,  # Fewer steps
            gamma=1.2,  # Lower gamma
            lambda_sparse=1e-3,
            optimizer_fn=torch.optim.AdamW,
            optimizer_params=dict(lr=4e-4, weight_decay=1e-4),  # Conservative learning
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_params={"mode": "max", "patience": 8, "factor": 0.5},
            mask_type='entmax',
            device_name='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Apply custom precision-focused loss
        self.clf._loss_fn = PrecisionFocalLoss(
            class_weights=class_weights,
            gamma=2.0,
            alpha=0.8  # Slightly reduced from 0.85 for better balance
        )

    def train(self):
        self.clf.fit(
            X_train=self.X_train_scaled,
            y_train=self.y_train_res,
            eval_set=[(self.X_test_scaled, self.y_test_enc)],
            eval_name=["test"],
            eval_metric=["accuracy"],
            max_epochs=100,
            patience=25,  # More patience
            batch_size=512,
            virtual_batch_size=128,
            drop_last=False
        )

    def evaluate(self):
        # Get probabilities and predictions
        probs = self.clf.predict_proba(self.X_test_scaled)
        final_preds = np.argmax(probs, axis=1)
        
        # Calculate metrics
        precision = precision_score(self.y_test_enc, final_preds, average=None)
        recall = recall_score(self.y_test_enc, final_preds, average=None)
        f1 = f1_score(self.y_test_enc, final_preds, average=None)

        print("\n📊 Evaluation Results:")
        print(f"Accuracy: {accuracy_score(self.y_test_enc, final_preds):.4f}")
        print("\nClass-wise Metrics:")
        for i, class_name in enumerate(self.le.classes_):
            print(f"Class {class_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}")

        print("\nClassification Report:")
        print(classification_report(
            self.y_test_enc, final_preds,
            target_names=self.le.classes_,
            zero_division=0
        ))

        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test_enc, final_preds))
        
        # Focused analysis
        print("\n🎯 Class 2 Performance:")
        print(f"Precision: {precision[2]:.4f} (Higher is better - fewer false positives)")
        print(f"Recall:    {recall[2]:.4f}")
        print(f"F1 Score:  {f1[2]:.4f}")
        print(f"Tradeoff:  Precision is {precision[2]/recall[2]:.1f}x higher than recall")


# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    print("🚀 Starting TabNet Precision Optimization Training...")
    try:
        trainer = TabNetPrecisionTrainer(
            train_path=r"D:\AlgoT\trades.csv",
            test_path=r"D:\AlgoT\tradesN50OUTSAMPLE.csv"
        )
        trainer.train()
        trainer.evaluate()
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("Please check your data and try again.")