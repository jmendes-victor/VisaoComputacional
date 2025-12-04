import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .config import DIM_FEATURES, CSV_DIR, EMB_DIR, TASK_FILES


class WSIEmbeddingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, embeddings_dir: str, preload: bool = True):
        self.pids = df["Patient"].values
        self.labels = pd.to_numeric(df["GT"]).values.astype(int)
        self.emb_dir = embeddings_dir
        self.data_cache = []
        self.use_cache = preload

        if self.use_cache:
            for i, pid in enumerate(self.pids):
                path = os.path.join(self.emb_dir, f"{pid}.npy")
                try:
                    feat = torch.from_numpy(np.load(path)).float()
                    if feat.shape[0] > 0:
                        feat = F.normalize(feat, p=2, dim=1)
                    else:
                        feat = torch.zeros((1, DIM_FEATURES))
                except Exception:
                    feat = torch.zeros((1, DIM_FEATURES))
                self.data_cache.append((feat, torch.tensor(self.labels[i]).long()))

    def __len__(self) -> int:
        return len(self.pids)

    def __getitem__(self, idx: int):
        if self.use_cache:
            return self.data_cache[idx]
        return torch.zeros((1, DIM_FEATURES)), torch.tensor(0).long()


def mil_collate(batch):
    features, labels = zip(*batch)
    return list(features), torch.stack(labels)


def get_tasks_scenario_3(csv_dir: str = CSV_DIR, emb_dir: str = EMB_DIR) -> list:
    tasks = []
    print("\n>>> Carregando Datasets...", flush=True)
    
    for i, (tr, val, te) in enumerate(TASK_FILES):
        d_tr = pd.read_csv(os.path.join(csv_dir, tr))
        d_te = pd.read_csv(os.path.join(csv_dir, te))
        classes = sorted(d_tr["GT"].unique().astype(int).tolist())
        
        train_ds = WSIEmbeddingDataset(d_tr, emb_dir, preload=True)
        test_ds = WSIEmbeddingDataset(d_te, emb_dir, preload=True)
        
        print(f"   Task {i+1}: {len(train_ds)} treino, {len(test_ds)} teste [OK]", flush=True)
        
        tasks.append({
            "id": i,
            "classes": classes,
            "train_loader": DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=mil_collate),
            "test_loader": DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=mil_collate)
        })
    
    return tasks
