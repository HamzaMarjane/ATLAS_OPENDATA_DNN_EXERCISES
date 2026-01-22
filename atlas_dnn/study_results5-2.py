import os
import h5py
import torch
import matplotlib.pyplot as plt
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

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
    os.makedirs("plots", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load processed data
    X, y = load_h5()

    # Same test split as before
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y.numpy()
    )

    # Load checkpoint from training (5-2)
    checkpoint_path = "models/dnn_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    mean = checkpoint["mean"]
    std = checkpoint["std"]

    # Apply scaling
    X_test = (X_test - mean) / std
    X_test = X_test.to(device)

    model = build_model(dropout=0.2).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Get scores
    with torch.no_grad():
        scores = model(X_test).squeeze().cpu()

    # Split scores by class
    signal_scores = scores[y_test == 1].numpy()
    background_scores = scores[y_test == 0].numpy()

    # ---------- Plot 1: Score distributions ----------
    plt.figure()
    plt.hist(signal_scores, bins=40, histtype="step", label="Signal")
    plt.hist(background_scores, bins=40, histtype="step", label="Background")
    plt.xlabel("Model output score")
    plt.ylabel("Events")
    plt.legend()
    scores_path = "plots/scores5-2.png"
    plt.savefig(scores_path, dpi=200)
    print("Saved:", scores_path)

    # ---------- Plot 2: ROC curve ----------
    fpr, tpr, _ = roc_curve(y_test.numpy(), scores.numpy())
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    roc_path = "plots/roc5-2.png"
    plt.savefig(roc_path, dpi=200)
    print("Saved:", roc_path)

    print("\nDone. Plots are in the plots/ folder.")

if __name__ == "__main__":
    main()
