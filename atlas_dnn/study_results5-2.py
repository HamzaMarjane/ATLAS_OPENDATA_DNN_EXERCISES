import os
import glob
import h5py
import torch
import matplotlib.pyplot as plt
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

os.makedirs("plots", exist_ok=True)

# ---- TAB completion for file paths ----
try:
    import readline
except ImportError:
    readline = None

def _path_completer(text, state):
    expanded = os.path.expanduser(os.path.expandvars(text))
    matches = glob.glob(expanded + "*")
    matches = [m + ("/" if os.path.isdir(m) and not m.endswith("/") else "") for m in matches]
    return matches[state] if state < len(matches) else None

def input_path(prompt, default):
    if readline:
        readline.set_completer_delims(" \t\n;")
        readline.parse_and_bind("tab: complete")
        readline.set_completer(_path_completer)

    s = input(f"{prompt} [default: {default}]: ").strip()
    return s if s else default

def load_h5(path):
    with h5py.File(path, "r") as f:
        X = torch.tensor(f["X"][:], dtype=torch.float32)
        y = torch.tensor(f["y"][:], dtype=torch.float32)
    return X, y

def build_model(dropout=0.2):
    return nn.Sequential(
        nn.Linear(18, 32),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )

def safe_tag(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

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

    sig = scores[y_test == 1].numpy()
    bkg = scores[y_test == 0].numpy()

    tag = safe_tag(ckpt_path)

    # Score histogram
    plt.figure()
    plt.hist(sig, bins=40, histtype="step", label="Signal")
    plt.hist(bkg, bins=40, histtype="step", label="Background")
    plt.xlabel("Model output score")
    plt.ylabel("Events")
    plt.legend()
    out_scores = f"plots/scores_{tag}.png"
    plt.savefig(out_scores, dpi=200)

    # ROC
    fpr, tpr, _ = roc_curve(y_test.numpy(), scores.numpy())
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    out_roc = f"plots/roc_{tag}.png"
    plt.savefig(out_roc, dpi=200)

    print("\n--- Study plots saved ---")
    print("Scores:", out_scores)
    print("ROC   :", out_roc)

if __name__ == "__main__":
    main()
