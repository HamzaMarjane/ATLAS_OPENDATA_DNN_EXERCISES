import h5py
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# ---------- Load HDF5 data ----------
def load_h5(path="data/atlas_processed.h5"):
    with h5py.File(path, "r") as f:
        X = torch.tensor(f["X"][:], dtype=torch.float32)
        y = torch.tensor(f["y"][:], dtype=torch.float32)
    return X, y

# ---------- Model definition (must match train.py) ----------
def build_model(dropout=0.2):
    return nn.Sequential(
        nn.Linear(18, 32),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load processed data
    X, y = load_h5()

    # Same test split as during training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y.numpy()
    )

    # ---------- Load trained model ----------
    checkpoint = torch.load("models/dnn_best.pt", map_location=device)

    mean = checkpoint["mean"]
    std  = checkpoint["std"]

    # Apply same scaling as training
    X_test = (X_test - mean) / std
    X_test = X_test.to(device)

    model = build_model(dropout=0.2).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # ---------- Evaluation ----------
    with torch.no_grad():
        scores = model(X_test).squeeze().cpu()

    predictions = (scores >= 0.5).float()

    accuracy = accuracy_score(y_test.numpy(), predictions.numpy())
    auc = roc_auc_score(y_test.numpy(), scores.numpy())

    print("\n--- Evaluation results ---")
    print("Accuracy :", accuracy)
    print("ROC AUC  :", auc)

if __name__ == "__main__":
    main()
