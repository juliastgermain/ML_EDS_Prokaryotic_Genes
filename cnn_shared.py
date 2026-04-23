# cnn_shared.py  — updated for actual sequence lengths

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

DNA_VOCAB = {c: i+1 for i, c in enumerate('ACGT')}
PRO_VOCAB = {c: i+1 for i, c in enumerate('ACDEFGHIKLMNPQRSTVWY')}

def encode(seq, vocab, max_len):
    ids = [vocab.get(c, 0) for c in seq.upper()[:max_len]]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

class PairDataset(Dataset):
    def __init__(self, parquet_path, vocab, max_len):
        df = pd.read_parquet(parquet_path)
        self.vocab   = vocab
        self.max_len = max_len
        self.seq1    = df['seq1'].tolist()
        self.seq2    = df['seq2'].tolist()
        self.dist    = df['distance'].astype(float).tolist()

    def __len__(self): return len(self.dist)

    def __getitem__(self, idx):
        return (encode(self.seq1[idx], self.vocab, self.max_len),
                encode(self.seq2[idx], self.vocab, self.max_len),
                torch.tensor(self.dist[idx], dtype=torch.float32))

class CNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_filters=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.convs = nn.Sequential(
            # block 1 — local motifs
            nn.Conv1d(embed_dim,      num_filters,   kernel_size=7, padding=3),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.MaxPool1d(4),           # L/4

            # block 2 — medium-range patterns
            nn.Conv1d(num_filters,    num_filters*2, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_filters*2),
            nn.ReLU(),
            nn.MaxPool1d(4),           # L/16

            # block 3 — added to absorb the longer sequences
            nn.Conv1d(num_filters*2,  num_filters*4, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_filters*4),
            nn.ReLU(),
            nn.MaxPool1d(4),           # L/64

            # block 4 — high-level
            nn.Conv1d(num_filters*4,  num_filters*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters*4),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),   # → fixed vector regardless of length
        )
        self.out_dim = num_filters * 4

    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        return self.convs(x).squeeze(-1)

class SiameseCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_filters=128):
        super().__init__()
        self.encoder = CNNEncoder(vocab_size, embed_dim, num_filters)
        d = self.encoder.out_dim
        self.head = nn.Sequential(
            nn.Linear(d * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, x1, x2):
        e1, e2 = self.encoder(x1), self.encoder(x2)
        combined = torch.cat([e1, e2, torch.abs(e1 - e2)], dim=1)
        return self.head(combined).squeeze(-1)

def train(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cpu'):
    model.to(device)
    opt       = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best_val  = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for x1, x2, y in train_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            loss = F.mse_loss(model(x1, x2), y)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x1, x2, y in val_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                val_loss += F.mse_loss(model(x1, x2), y).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        scheduler.step(val_loss)
        print(f"Epoch {epoch:02d} | train MSE {train_loss:.4f} | val MSE {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f'checkpoints/best_{type(model).__name__}_{model.encoder.embed.weight.shape[0]-1}vocab.pt')
            print(f"           ↑ saved")