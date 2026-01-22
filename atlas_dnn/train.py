import os
import h5py
import torch
from torch import nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

os.makedirs("models", exist_ok=True)

def load_h5(path="data/atlas_processed.h5"):
    with h5py.File(path, "r") as f:
        X = torch.tensor(f["X"][:], dtype=torch.float32)
        y = torch.tensor(f["y"][:], dtype=torch.float32)
    return X, y

def main():
    X, y = load_h5()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y.numpy()
    )

    mean = X_train.mean(dim=0, keepdim=True)
    std  = X_train.std(dim=0, keepdim=True).clamp_min(1e-8)

    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std

    model = nn.Sequential(
        nn.Linear(18, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    N_epochs = 50
    for epoch in range(N_epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train).squeeze()
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{N_epochs}  Loss={loss.item():.4f}")

    torch.save(
        {"model_state": model.state_dict(), "mean": mean, "std": std},
        "models/dnn.pt"
    )
    print("Saved models/dnn.pt")

if __name__ == "__main__":
    main()
