import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from torch.nn import CrossEntropyLoss
import warnings


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class TabNetTrainer:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self._prepare_data()
        self._initialize_model()

    def _prepare_data(self):
        # Load and clean
        train_df = pd.read_csv(self.train_path)
        train_df=train_df.drop(columns=['POSITION'])
        test_df = pd.read_csv(self.test_path)
        test_df=test_df.drop(columns=['POSITION'])
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

        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"\n✅ Data Prepared:")
        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")
        print(f"Classes: {list(self.le.classes_)}")

    def _initialize_model(self):
        # Compute class weights
        class_counts = np.bincount(self.y_train_enc)
        class_counts = np.maximum(class_counts, 1)
        self.class_weights = torch.tensor(
            np.sqrt(np.median(class_counts) / class_counts),
            dtype=torch.float32
        )

        # Initialize TabNetClassifier
        self.clf = TabNetClassifier(
            n_d=24,
            n_a=24,
            n_steps=3  ,
            gamma=1.1,
            lambda_sparse=1e-3,
            mask_type="entmax",
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-3),
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_params={"mode": "max", "patience": 3},
            device_name="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.clf._loss_fn = CrossEntropyLoss(weight=self.class_weights)

    def train(self):
        self.clf.fit(
            X_train=self.X_train_scaled,
            y_train=self.y_train_enc,
            eval_set=[(self.X_test_scaled, self.y_test_enc)],
            eval_name=["test"],
            eval_metric=["accuracy"],
            max_epochs=50,
            patience=10,
            batch_size=512,
            virtual_batch_size=64
        )

    def evaluate(self):
        probas = self.clf.predict_proba(self.X_test_scaled)
        final_preds = np.argmax(probas, axis=1)

        class_names = list(self.le.classes_)

        print(f"\n📊 Evaluation Results:")
        print(f"Accuracy: {accuracy_score(self.y_test_enc, final_preds):.4f}")
        print("\nClassification Report:")
        print(classification_report(
            self.y_test_enc, final_preds,
            target_names=class_names,
            zero_division=0
        ))

        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test_enc, final_preds))
        

        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(self.y_test_enc, final_preds), 
                    annot=True, fmt='d', cmap='Blues',
                    xticklabels=np.unique(self.y_test_enc),
                    yticklabels=np.unique(final_preds))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        # Save model parameters
        self.clf.save_model("tabnet2_model")

        # 🎯 Focused Metrics
        prec_class_0 = precision_score(self.y_test_enc, final_preds, labels=[0], average='macro')
        rec_class_2 = recall_score(self.y_test_enc, final_preds, labels=[2], average='macro')
        print(f"\n🎯 Precision (Class 0): {prec_class_0:.4f}")
        print(f"🎯 Recall (Class 2):    {rec_class_2:.4f}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    try:
        print("🚀 Starting TabNet training...")
        trainer = TabNetTrainer(
            train_path=r"D:\AlgoT\trades2.csv",

            test_path=r"D:\AlgoT\tradesN50OUTSAMPLE.csv"
        )
        trainer.train()
        trainer.evaluate()
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("Please check your data and try again.")

