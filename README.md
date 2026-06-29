# FedTether

**Cross-Attack Robust Aggregation for Federated Learning on IoT Edge Networks**

---

## Overview

Federated learning lets resource-constrained IoT and edge devices train a shared model without exchanging raw data, but compromised or faulty devices can poison the global model. Existing robust aggregators each fail in some regime: naive averaging (FedAvg) collapses under high-magnitude attacks, while distance- and statistics-based defenses (Krum, median, trimmed mean) lose accuracy on non-IID data and stay vulnerable to stealthy attacks.

**FedTether** reweights client updates by their deviation from a coordinate-wise median through a robust, *dead-zoned* score, applied as a server-side optimizer step. The dead-zone is the key mechanism: when no update is anomalous, all weights are unchanged and FedTether reduces **exactly** to FedAvg — so it pays **no robustness tax** when there is nothing to defend against.

Across three datasets, three attack types, and Byzantine fractions up to 40%, FedTether is the only method that stays top-tier in every tested regime, requiring no shared data, no validation set, and no known attacker count.

## Method at a glance

For each round, given client updates δ_i = c_i − θ:

1. **Reference direction** — coordinate-wise median `m = median({δ_i})`.
2. **Energy** — `e_i = ||δ_i − m||²`.
3. **Robust score** — standardize `e_i` with a MAD-based location/scale and a floor:
   - `MAD = 1.4826 · median(|e_i − median(e)|)`
   - `s = max(MAD, ρ · median(e)) + ε`
   - `z_i = (e_i − median(e)) / s`
4. **Dead-zone weight** — `γ_i = exp(−β · max(0, z_i − τ))`; aggregation weights `w_i = γ_i n_i / Σ_j γ_j n_j`.
5. **Server step** — `θ ← θ + η · Σ_i w_i δ_i`.

When no update exceeds the dead-zone (`z_i ≤ τ` for all i), `γ_i = 1` and the rule is identical to FedAvg.

## Key results

CIFAR-10 final accuracy at 30% Byzantine, under three attacks (best per column in **bold**):

| Method        | ALIE (stealthy) | Sign-flip   | Gaussian    |
|---------------|:---------------:|:-----------:|:-----------:|
| FedAvg        | 0.681           | 0.551       | 0.094       |
| Median        | 0.492           | 0.469       | 0.713       |
| Trimmed mean  | 0.590           | 0.472       | 0.096       |
| Multi-Krum    | 0.387           | 0.552       | **0.726**   |
| RFA           | 0.609           | **0.602**   | **0.726**   |
| FedTether-S   | **0.685**       | 0.600       | 0.724       |
| FedTether-A   | 0.678           | 0.600       | 0.728       |

Every baseline collapses under at least one attack; FedTether stays top-tier under all three. See the paper for cross-dataset (MNIST / CIFAR-10 / CIFAR-100), robustness-tax (0–40% Byzantine), and 50-client partial-participation results.

## Repository structure

<!-- TODO: update to match the released layout -->

```
FedTether/
├── notebooks/
│   └── FedTether.ipynb   # end-to-end experiments
├── src/                              # (planned) standalone implementation
│   ├── aggregators/                  # FedTether, FedAvg, Median, Trimmed, Multi-Krum, RFA
│   ├── attacks/                      # ALIE, sign-flip, Gaussian
│   └── data/                         # Dirichlet non-IID partitioning
├── configs/                          # hyperparameters per experiment
├── requirements.txt
└── README.md
```

## Setup

<!-- TODO: confirm versions against your environment and add a requirements.txt -->

```bash
git clone https://github.com/nmaislab/FedTether.git
cd FedTether
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # torch, flwr (Flower), numpy, ...
```

Experiments are implemented in [Flower](https://flower.ai/). The current entry point is the notebook under `notebooks/`.

## Experimental configuration

| Setting | Value |
|---|---|
| Framework | Flower |
| Clients | 10 (full participation); 50 (partial-participation study) |
| Data split | Dirichlet non-IID, α = 0.3 |
| Model | Compact CNN, GroupNorm (8 groups), ~0.62–0.82M params |
| Local training | 2 epochs, SGD, lr 0.01, momentum 0.9, batch 32 |
| Rounds | MNIST 15 / CIFAR-10 25 / CIFAR-100 30 |
| Seeds | 5 (identical per-seed init) |
| Baselines | FedAvg, Median, Trimmed Mean, Multi-Krum, RFA |
| Attacks | ALIE (stealthy), sign-flipping, Gaussian |

**FedTether hyperparameters (frozen study-wide):** τ = 3.0, ρ = 0.5, ε = 1e-12, η = 1.0 (no momentum); static β = 2.0; adaptive β ∈ [0.2, 4.0] with κ = 1.5, z₀ = 3.0.

## Citation

```bibtex
@inproceedings{almasri_fedtether,
  title     = {FedTether: Cross-Attack Robust Aggregation for Federated Learning on IoT Edge Networks},
  author    = {Al-Masri, Eyhab},
  booktitle = {TODO},
  year      = {TODO}
}
```

## License
MIT or Apache-2.0


## Contact
TBD
