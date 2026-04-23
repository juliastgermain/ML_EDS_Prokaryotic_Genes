import os, re
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def clean_id(full_id):
    if not full_id: return ""
    match = re.search(r'([A-Z]{2}_GC[AF]_\d{9}\.\d+)', str(full_id))
    return match.group(1) if match else str(full_id).split('_')[0]

def get_pair_type(id1, id2):
    if id1 in arc_ids and id2 in arc_ids: return 'arc-arc'
    if id1 in bac_ids and id2 in bac_ids: return 'bac-bac'
    return 'arc-bac'

# 1. Load
dna_bac = load_dataset("tattabio/rpob_bac_dna_phylogeny_sequences", split='train')
dna_arc = load_dataset("tattabio/rpob_arch_dna_phylogeny_sequences", split='train')
pro_bac = load_dataset("tattabio/rpob_bac_phylogeny_sequences",      split='train')
pro_arc = load_dataset("tattabio/rpob_arch_phylogeny_sequences",     split='train')
dist_bac = load_dataset("tattabio/rpob_bac_dna_phylogeny_distances", split='train')
dist_arc = load_dataset("tattabio/rpob_arch_dna_phylogeny_distances", split='train')

# 2. Maps
dna_map = {clean_id(r['Entry']): r['Sequence'] for r in dna_bac}
dna_map.update({clean_id(r['Entry']): r['Sequence'] for r in dna_arc})
pro_map = {clean_id(r['Entry']): r['Sequence'] for r in pro_bac}
pro_map.update({clean_id(r['Entry']): r['Sequence'] for r in pro_arc})

bac_ids = {clean_id(r['Entry']) for r in dna_bac}
arc_ids = {clean_id(r['Entry']) for r in dna_arc}

# 3. Split on organism IDs (same split for both so organisms never leak)
common_ids = sorted(set(dna_map) & set(pro_map))
train_ids, test_ids = train_test_split(common_ids, test_size=0.2, random_state=42)
train_set, test_set = set(train_ids), set(test_ids)
print(f"Organisms — train: {len(train_set)} | test: {len(test_set)}")

# 4. Build SEPARATE dna and protein pair lists
train_dna, test_dna = [], []
train_pro, test_pro = [], []

for row in list(dist_bac) + list(dist_arc):
    id1, id2 = clean_id(row['ID1']), clean_id(row['ID2'])
    ptype = get_pair_type(id1, id2)
    dist  = float(row['distance'])

    in_train = id1 in train_set and id2 in train_set
    in_test  = id1 in test_set  and id2 in test_set

    if in_train or in_test:
        target = (train_dna, train_pro) if in_train else (test_dna, test_pro)

        if id1 in dna_map and id2 in dna_map:
            target[0].append({'distance': dist, 'pair_type': ptype,
                               'seq1': dna_map[id1], 'seq2': dna_map[id2]})
        if id1 in pro_map and id2 in pro_map:
            target[1].append({'distance': dist, 'pair_type': ptype,
                               'seq1': pro_map[id1], 'seq2': pro_map[id2]})

# 5. Report
for name, data in [('train_dna', train_dna), ('test_dna',  test_dna),
                   ('train_pro', train_pro), ('test_pro',  test_pro)]:
    df = pd.DataFrame(data)
    print(f"\n{name}: {len(df):,} pairs")
    print(df['pair_type'].value_counts().to_string())

# 6. Save
os.makedirs('data', exist_ok=True)
pd.DataFrame(train_dna).to_parquet('data/train_dna.parquet')
pd.DataFrame(test_dna) .to_parquet('data/test_dna.parquet')
pd.DataFrame(train_pro).to_parquet('data/train_protein.parquet')
pd.DataFrame(test_pro) .to_parquet('data/test_protein.parquet')
print("\nSaved all four files.")