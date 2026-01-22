import h5py
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

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

    preds = (scores >= 0.5).float()

    acc = accuracy_score(y_test.numpy(), preds.numpy())
    auc = roc_auc_score(y_test.numpy(), scores.numpy())

    print("Accuracy:", acc)
    print("ROC AUC:", auc)

if __name__ == "__main__":
    main()

