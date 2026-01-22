import os
import h5py
import torch
import matplotlib.pyplot as plt
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

os.makedirs("plots", exist_ok=True)

def load_h5(path="data/atlas_processed.h5"):
    with h5py.File(path, "r") as f:
        X = torch.tensor(f["X"][:], dtype=torch.float32)
        y = torch.tensor(f["y"][:], dtype=torch.float32)
    return X, y

def build_model():
    return nn.Sequential(
        nn.Linear(18, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )

def main():
    X, y = load_h5()
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y.numpy()
    )

    ckpt = torch.load("models/dnn.pt", map_location="cpu")
    mean, std = ckpt["mean"], ckpt["std"]
    X_test = (X_test - mean) / std

    model = build_model()
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        scores = model(X_test).squeeze()

    sig = scores[y_test == 1].numpy()
    bkg = scores[y_test == 0].numpy()

    # Score histogram
    plt.figure()
    plt.hist(sig, bins=40, histtype="step", label="Signal")
    plt.hist(bkg, bins=40, histtype="step", label="Background")
    plt.xlabel("Model output score")
    plt.ylabel("Events")
    plt.legend()
    plt.savefig("plots/scores.png", dpi=200)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test.numpy(), scores.numpy())
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig("plots/roc.png", dpi=200)

    print("Saved plots: plots/scores.png and plots/roc.png")

if __name__ == "__main__":
    main()
