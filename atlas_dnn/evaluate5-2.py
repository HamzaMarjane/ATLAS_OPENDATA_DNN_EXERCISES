import os
import glob
import h5py
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# ---- TAB completion for file paths ----
try:
    import readline  # built-in on most Linux
except ImportError:
    readline = None

def _path_completer(text, state):
    # expand ~ and environment vars for matching, but keep user-visible text
    expanded = os.path.expanduser(os.path.expandvars(text))
    matches = glob.glob(expanded + "*")
    # add trailing slash for dirs to continue completion nicely
    matches = [m + ("/" if os.path.isdir(m) and not m.endswith("/") else "") for m in matches]
    # return matches in original form if possible
    return matches[state] if state < len(matches) else None

def input_path(prompt, default):
    if readline:
        readline.set_completer_delims(" \t\n;")
        readline.parse_and_bind("tab: complete")
        readline.set_completer(_path_completer)

    s = input(f"{prompt} [default: {default}]: ").strip()
    if not s:
        s = default
    return s

# ---- data/model ----
def load_h5(path):
    with h5py.File(path, "r") as f:
        X = torch.tensor(f["X"][:], dtype=torch.float32)
        y = torch.tensor(f["y"][:], dtype=torch.float32)
    return X, y

def build_model(dropout=0.2):
    # must match train.py architecture you used (with Dropout)
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

    h5_path = input_path("HDF5 data file", "data/atlas_processed.h5")
    ckpt_path = input_path("Model checkpoint", "models/dnn_best.pt")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Data file not found: {h5_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    X, y = load_h5(h5_path)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y.numpy()
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    if not all(k in ckpt for k in ("model_state", "mean", "std")):
        raise KeyError("Checkpoint must contain keys: model_state, mean, std")

    mean, std = ckpt["mean"], ckpt["std"]
    X_test = (X_test - mean) / std
    X_test = X_test.to(device)

    model = build_model(dropout=0.2).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        scores = model(X_test).squeeze().cpu()

    preds = (scores >= 0.5).float()

    acc = accuracy_score(y_test.numpy(), preds.numpy())
    auc = roc_auc_score(y_test.numpy(), scores.numpy())

    print("\n--- Evaluation ---")
    print("Data file  :", h5_path)
    print("Checkpoint :", ckpt_path)
    print("Accuracy   :", acc)
    print("ROC AUC    :", auc)

if __name__ == "__main__":
    main()

