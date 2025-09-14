import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import torch
from torch.nn import Module, Softmax
import torch.nn.functional as F
from pytorch_tabnet.tab_model import TabNetClassifier
import warnings

# -------------------------------
# Focal Loss Implementation
# -------------------------------
class FocalLoss(Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        ce_loss = F.nll_loss(log_probs, targets, reduction='none')
        focal_loss = self.alpha * (1 - probs.gather(1, targets.unsqueeze(1)).squeeze()) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()


# -------------------------------
# TabNet Trainer
# -------------------------------
class TabNetTrainer:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self._prepare_data()
        self._initialize_model()

    def _prepare_data(self):
        # Load and clean
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        # Drop rows with NaN in features or target
        train_df = train_df.dropna()
        test_df = test_df.dropna()

        # Ensure target column exists
        if "Y" not in train_df.columns or "Y" not in test_df.columns:
            raise ValueError("Target column 'Y' missing in data")

        # Separate features and target
        self.X_train = train_df.drop(columns=["Y"])
        self.y_train = train_df["Y"]
        self.X_test = test_df.drop(columns=["Y"])
        self.y_test = test_df["Y"]

        # Normalize label types: float → int → str
        self.y_train = self.y_train.astype(float).astype(int).astype(str)
        self.y_test = self.y_test.astype(float).astype(int).astype(str)

        # Get common labels
        common_labels = sorted(list(set(self.y_train.unique()) & set(self.y_test.unique())))
        if not common_labels:
            raise ValueError("No common labels between train and test sets")

        # Filter both sets to only include common labels
        train_mask = self.y_train.isin(common_labels)
        test_mask = self.y_test.isin(common_labels)

        self.X_train = self.X_train[train_mask]
        self.y_train = self.y_train[train_mask]
        self.X_test = self.X_test[test_mask]
        self.y_test = self.y_test[test_mask]

        # Encode labels
        self.le = LabelEncoder()
        self.y_train_enc = self.le.fit_transform(self.y_train)
        self.y_test_enc = self.le.transform(self.y_test)


        # SMOTE Oversampling
        sm = SMOTE(random_state=42)
        self.X_train_res, self.y_train_res = sm.fit_resample(self.X_train, self.y_train_enc)

        # Standardize
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train_res)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print("\n✅ Data Prepared:")
        print(f"Training samples: {len(self.X_train_scaled)}")
        print(f"Test samples: {len(self.X_test_scaled)}")
        print(f"Classes: {self.le.classes_}")

    def _initialize_model(self):
        # Define TabNet
        self.clf = TabNetClassifier(
            n_d=24,
            n_a=24,
            n_steps=3,
            gamma=1.3,
            lambda_sparse=1e-4,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-3),
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_params={"mode": "max", "patience": 3},
            mask_type='entmax',
            device_name='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Focal Loss
        self.clf._loss_fn = FocalLoss(alpha=1.0, gamma=2.0)

    def train(self):
        self.clf.fit(
            X_train=self.X_train_scaled,
            y_train=self.y_train_res,
            eval_set=[(self.X_test_scaled, self.y_test_enc)],
            eval_name=["test"],
            eval_metric=["accuracy"],
            max_epochs=50,
            patience=10,
            batch_size=512,
            virtual_batch_size=64,
            drop_last=False
        )

    def evaluate(self):
        preds = self.clf.predict(self.X_test_scaled)
        probs = self.clf.predict_proba(self.X_test_scaled)
        final_preds = np.argmax(probs, axis=1)

        print("\n📊 Evaluation Results:")
        print(f"Accuracy: {accuracy_score(self.y_test_enc, final_preds):.4f}")
        print("\nClassification Report:")
        print(classification_report(
            self.y_test_enc, final_preds,
            target_names=self.le.classes_,
            zero_division=0
        ))

        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test_enc, final_preds))

        # Focused metrics
        # class_0_precision = precision_score(self.y_test_enc, final_preds, average=None)[0]
        # class_2_recall = recall_score(self.y_test_enc, final_preds, average=None)[2]
        # print(f"\n🎯 Precision (Class 0): {class_0_precision:.4f}")
        # print(f"🎯 Recall (Class 2):    {class_2_recall:.4f}")


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    print("🚀 Starting TabNet training...")
    try:
        trainer = TabNetTrainer(
            train_path=r"D:\AlgoT\trades.csv",
            test_path=r"D:\AlgoT\tradesN50OUTSAMPLE.csv"
        )
        trainer.train()
        trainer.evaluate()
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("Please check your data and try again.")
