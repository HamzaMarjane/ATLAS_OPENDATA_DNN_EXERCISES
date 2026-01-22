import os
import h5py
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split

os.makedirs("models", exist_ok=True)

def load_h5(path="data/atlas_processed.h5"):
    with h5py.File(path, "r") as f:
        X = torch.tensor(f["X"][:], dtype=torch.float32)
        y = torch.tensor(f["y"][:], dtype=torch.float32)
    return X, y

def build_model(dropout=0.2):
    # Same idea as your notebook, but with Dropout added
    return nn.Sequential(
        nn.Linear(18, 32),
        nn.ReLU(),
        nn.Dropout(dropout),        # <-- regularisation
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load processed data
    X, y = load_h5()

    # Train/validation split (needed for early stopping + scheduler)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y.numpy()
    )

    # Scale using training stats only
    mean = X_train.mean(dim=0, keepdim=True)
    std  = X_train.std(dim=0, keepdim=True).clamp_min(1e-8)

    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std

    # ---- BATCHING via DataLoader ----
    batch_size = 1024
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False
    )

    model = build_model(dropout=0.2).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # ---- LR SCHEDULER (dynamic LR) ----
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # ---- EARLY STOPPING ----
    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = 8  # stop if no improvement for 8 epochs

    max_epochs = 100

    for epoch in range(1, max_epochs + 1):
        # ---- TRAIN ----
        model.train()
        train_loss_sum = 0.0
        n_train = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * xb.size(0)
            n_train += xb.size(0)

        train_loss = train_loss_sum / n_train

        # ---- VALIDATION LOSS ----
        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                preds = model(xb).squeeze()
                loss = criterion(preds, yb)

                val_loss_sum += loss.item() * xb.size(0)
                n_val += xb.size(0)

        val_loss = val_loss_sum / n_val

        # Scheduler reacts to validation loss
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d} | lr={lr:.5f} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_improve = 0

            torch.save(
                {"model_state": model.state_dict(), "mean": mean, "std": std},
                "models/dnn_best.pt"
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping: val_loss not improving. Best val_loss={best_val_loss:.4f}")
                break

    print("Saved best model to models/dnn_best.pt")

if __name__ == "__main__":
    main()
