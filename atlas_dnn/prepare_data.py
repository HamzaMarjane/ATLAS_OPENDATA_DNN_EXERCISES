import os
import h5py
import torch
import awkward as ak
import uproot
import atlasopenmagic as atom

os.makedirs("data", exist_ok=True)

atom.set_release("2025e-13tev-beta")

BRANCHES = [
    "lep_pt", "lep_eta", "lep_phi", "lep_e",
    "jet_pt", "jet_eta", "jet_phi", "jet_e",
    "met", "met_phi"
]

def prepare_data(file_path: str) -> torch.Tensor:
    # Colab-style: open remote URL directly (https) + let fsspec cache handle it
    print(f"Opening: {file_path}", flush=True)
    file = uproot.open(file_path)
    tree = file["analysis"]

    print("Reading branches...", flush=True)
    array = tree.arrays(BRANCHES)

    # >= 2 jets
    mask_2plus_jets = ak.num(array["jet_pt"]) >= 2
    filtered = array[mask_2plus_jets]

    # lepton pt >= 10 GeV (object-level)
    lep_10gev_mask = filtered["lep_pt"] >= 10
    filtered["lep_pt"]  = filtered["lep_pt"][lep_10gev_mask]
    filtered["lep_eta"] = filtered["lep_eta"][lep_10gev_mask]
    filtered["lep_phi"] = filtered["lep_phi"][lep_10gev_mask]
    filtered["lep_e"]   = filtered["lep_e"][lep_10gev_mask]

    # exactly 2 leptons after cut (event-level)
    mask_exact2 = ak.num(filtered["lep_pt"]) == 2
    final = filtered[mask_exact2]

    X = torch.zeros(len(final), 18, dtype=torch.float32)

    for i, b in enumerate(["lep_pt", "lep_eta", "lep_phi", "lep_e"]):
        X[:, i]   = torch.tensor(final[b][:, 0], dtype=torch.float32)
        X[:, i+4] = torch.tensor(final[b][:, 1], dtype=torch.float32)

    for i, b in enumerate(["jet_pt", "jet_eta", "jet_phi", "jet_e"]):
        X[:, i+8]  = torch.tensor(final[b][:, 0], dtype=torch.float32)
        X[:, i+12] = torch.tensor(final[b][:, 1], dtype=torch.float32)

    X[:, 16] = torch.tensor(final["met"], dtype=torch.float32)
    X[:, 17] = torch.tensor(final["met_phi"], dtype=torch.float32)

    return X

def main():
    signal_urls = atom.get_urls("346802", skim="2to4lep", protocol="https", cache=True)
    bkg_urls    = atom.get_urls("411234", skim="2to4lep", protocol="https", cache=True)

    # Prepare (limit for speed)
    X_sig = prepare_data(signal_urls[0])[:100000]
    X_bkg = prepare_data(bkg_urls[0])[:100000]

    y_sig = torch.ones(len(X_sig), dtype=torch.float32)
    y_bkg = torch.zeros(len(X_bkg), dtype=torch.float32)

    X = torch.cat([X_sig, X_bkg], dim=0).numpy()
    y = torch.cat([y_sig, y_bkg], dim=0).numpy()

    out = "data/atlas_processed.h5"
    with h5py.File(out, "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("y", data=y)

    print(f"Saved {out}  X={X.shape}  y={y.shape}", flush=True)

if __name__ == "__main__":
    main()
