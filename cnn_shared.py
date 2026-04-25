
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

DNA_VOCAB = {c: i+1 for i, c in enumerate("ACGT")}
PRO_VOCAB = {c: i+1 for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY")}

def encode(seq, vocab, max_len, augment=False):
    seq  = seq.upper()
    ids  = [vocab.get(c, 0) for c in seq]
    if augment and len(ids) > max_len:
        start = np.random.randint(0, len(ids) - max_len + 1)
        ids   = ids[start: start + max_len]
    if augment:
        ids = [0 if np.random.rand() < 0.05 else v for v in ids]
    ids = ids[:max_len]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

class PairDataset(Dataset):
    def __init__(self, parquet_path, vocab, max_len, augment=False):
        df = pd.read_parquet(parquet_path)
        self.vocab   = vocab
        self.max_len = max_len
        self.augment = augment
        self.seq1    = df["seq1"].tolist()
        self.seq2    = df["seq2"].tolist()
        self.dist    = df["distance"].astype(float).tolist()

    def __len__(self): return len(self.dist)

    def __getitem__(self, idx):
        return (encode(self.seq1[idx], self.vocab, self.max_len, self.augment),
                encode(self.seq2[idx], self.vocab, self.max_len, self.augment),
                torch.tensor(self.dist[idx], dtype=torch.float32))

class CNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_filters=128, dropout=0.4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.convs = nn.Sequential(
            nn.Conv1d(embed_dim,     num_filters,   kernel_size=7, padding=3),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(4),
            nn.Conv1d(num_filters,   num_filters*2, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_filters*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(4),
            nn.Conv1d(num_filters*2, num_filters*4, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_filters*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(4),
            nn.Conv1d(num_filters*4, num_filters*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters*4),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.out_dim = num_filters * 4

    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        return self.convs(x).squeeze(-1)

class SiameseCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_filters=128, dropout=0.4):
        super().__init__()
        self.encoder = CNNEncoder(vocab_size, embed_dim, num_filters, dropout)
        d = self.encoder.out_dim
        self.head = nn.Sequential(
            nn.Linear(d * 3, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x1, x2):
        e1, e2 = self.encoder(x1), self.encoder(x2)
        return self.head(torch.cat([e1, e2, torch.abs(e1 - e2)], dim=1)).squeeze(-1)

def train(model, train_loader, val_loader,
          epochs=30, lr=1e-3, device="cpu",
          save_path="checkpoints/best_model.pt"):

    model.to(device)

    all_targets = torch.tensor(train_loader.dataset.dist)
    dist_mean   = all_targets.mean().item()
    dist_std    = all_targets.std().item()
    print(f"Distance stats — mean: {dist_mean:.4f}  std: {dist_std:.4f}")

    opt       = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5)
    best_val  = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for x1, x2, y in train_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            y_norm = (y - dist_mean) / dist_std
            loss   = F.mse_loss(model(x1, x2), y_norm)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item()

        model.eval()
        val_loss, preds, targets = 0, [], []
        with torch.no_grad():
            for x1, x2, y in val_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                y_norm = (y - dist_mean) / dist_std
                p      = model(x1, x2)
                val_loss += F.mse_loss(p, y_norm).item()
                preds.extend((p * dist_std + dist_mean).cpu().tolist())
                targets.extend(y.cpu().tolist())

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        pearson_r   = np.corrcoef(preds, targets)[0, 1]
        scheduler.step(val_loss)
        print(f"Epoch {epoch:02d} | train MSE {train_loss:.4f} | "
              f"val MSE {val_loss:.4f} | val r {pearson_r:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(),
                        "dist_mean": dist_mean,
                        "dist_std":  dist_std}, save_path)
            print(f"           ↑ saved ({save_path})")
