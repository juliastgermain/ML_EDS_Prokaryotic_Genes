# train_protein.py
from cnn_shared import PairDataset, SiameseCNN, PRO_VOCAB, train
from torch.utils.data import DataLoader
import torch, os

# 1500 covers ~90% of sequences
MAX_LEN    = 1500
BATCH_SIZE = 32
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs('checkpoints', exist_ok=True)

train_ds = PairDataset('data/train_protein.parquet', PRO_VOCAB, MAX_LEN)
test_ds  = PairDataset('data/test_protein.parquet',  PRO_VOCAB, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

model = SiameseCNN(vocab_size=len(PRO_VOCAB), embed_dim=64, num_filters=128)
print(f"Protein model params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Device: {DEVICE}")

train(model, train_loader, val_loader, epochs=20, lr=1e-3, device=DEVICE)