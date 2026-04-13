import torch
from data.dataset import BiosignalDataset
from data.dataloader import create_dataloader
from train.train_utils import load_manifest_from_processed, split_manifest_by_subject

manifest = load_manifest_from_processed('/tmp/test_parse_100', signal_types=[0,1,2,3,4,5])
train_m, _ = split_manifest_by_subject(manifest, val_ratio=0.05, seed=42)
ds = BiosignalDataset(train_m, window_seconds=600.0, cache_size=4, patch_size=200)
dl = create_dataloader(ds, max_length=120000, batch_size=32, shuffle=False, num_workers=0, collate_mode='any_variate', patch_size=200)

batch = next(iter(dl))
p = 200
b, l = batch.values.shape
n = l // p
sid = batch.sample_id[:, ::p]
vid = batch.variate_id[:, ::p]
mask = sid != 0

# start_samples -> abs_time_id
per_row_max = vid.max(dim=-1).values
offsets = torch.zeros(b, dtype=torch.long)
if b > 1:
    offsets[1:] = per_row_max[:-1].cumsum(0)
gvi = (offsets.unsqueeze(-1) + vid - 1).clamp(min=0)
ps = batch.start_samples[gvi]
rel_time = torch.arange(n).unsqueeze(0).expand(b, -1)
abs_time = ps + rel_time * p
abs_time_id = abs_time // p

# positive pair count
for bi in range(min(b, 5)):
    pairs = 0
    for s in sid[bi].unique():
        if s == 0:
            continue
        m = (sid[bi] == s) & mask[bi]
        vids_here = vid[bi][m].unique()
        if len(vids_here) < 2:
            continue
        tids = abs_time_id[bi][m]
        vid_at = vid[bi][m]
        for t in tids.unique():
            t_vids = vid_at[tids == t].unique()
            if len(t_vids) >= 2:
                pairs += 1
    n_vids = vid[bi][mask[bi]].unique().numel()
    print(f'row {bi}: variates={n_vids}, matching_time_bins={pairs}')
