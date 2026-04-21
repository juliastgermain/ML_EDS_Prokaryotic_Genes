from datasets import load_dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
rpob_bac_seq_dna = load_dataset("tattabio/rpob_bac_dna_phylogeny_sequences")
rpob_arch_seq_dna = load_dataset("tattabio/rpob_arch_dna_phylogeny_sequences")

rpob_bac_seq_pro = load_dataset("tattabio/rpob_bac_phylogeny_sequences")
rpob_arch_seq_pro = load_dataset("tattabio/rpob_arch_phylogeny_sequences")

dist = load_dataset("tattabio/rpob_phylogeny_distances")

# Create lookups for DNA and Protein
dna_lookup = {
    **{row['Entry']: row['Sequence'] for row in rpob_bac_seq_dna['train']},
    **{row['Entry']: row['Sequence'] for row in rpob_arch_seq_dna['train']}
}

pro_lookup = {
    **{row['Entry']: row['Sequence'] for row in rpob_bac_seq_pro['train']},
    **{row['Entry']: row['Sequence'] for row in rpob_arch_seq_pro['train']}
}

valid_ids = set(dna_lookup.keys())


pairs_list = []

# Note: Access the ['train'] split specifically
for row in dist['train']:
    id1, id2 = row['ID1'], row['ID2']
    
    if id1 in valid_ids and id2 in valid_ids:
        pairs_list.append({
            'ID1': id1,
            'ID2': id2,
            'distance': row['distance'],
            'dna_seq1': dna_lookup[id1],
            'dna_seq2': dna_lookup[id2],
            'pro_seq1': pro_lookup[id1],
            'pro_seq2': pro_lookup[id2]
        })

# Create the real Pandas DataFrame
df_master = pd.DataFrame(pairs_list)

# Now line 33 will work perfectly!
df_dna = df_master[['dna_seq1', 'dna_seq2', 'distance']].copy()
df_pro = df_master[['pro_seq1', 'pro_seq2', 'distance']].copy()


print(f"Total valid pairs found: {len(df_master)}")

# Split indices to keep the pairs identical across modalities
train_idx, test_idx = train_test_split(
    df_master.index, 
    test_size=0.2, 
    random_state=42
)

# Apply the same split to both dataframes
train_dna = df_dna.loc[train_idx]
test_dna = df_dna.loc[test_idx]

train_pro = df_pro.loc[train_idx]
test_pro = df_pro.loc[test_idx]

print(f"Final training pairs: {len(train_dna)}")
print(f"Final testing pairs: {len(test_dna)}")