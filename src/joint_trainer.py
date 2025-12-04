import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score

from .config import DIM_FEATURES, NUM_CLASSES, CSV_DIR, EMB_DIR
from .model import CLAM_SB
from .dataset import WSIEmbeddingDataset, mil_collate


def train_joint_learning(csv_dir: str = CSV_DIR, emb_dir: str = EMB_DIR) -> tuple:
    print(f"\n{'='*60}")
    print(f" EXPERIMENTO: JOINT TRAINING (Upper Bound)")
    print(f"{'='*60}")
    
    task_files = ["2_4_train.csv", "0_5_train.csv", "3_1_train.csv"]
    dfs = []
    
    print(">>> [JT] Mesclando datasets...", flush=True)
    for f in task_files:
        dfs.append(pd.read_csv(os.path.join(csv_dir, f)))
    
    merged_df = pd.concat(dfs, ignore_index=True)
    joint_ds = WSIEmbeddingDataset(merged_df, emb_dir, preload=True)
    joint_loader = DataLoader(joint_ds, batch_size=1, shuffle=True, collate_fn=mil_collate)

    model = CLAM_SB(input_dim=DIM_FEATURES, n_classes=NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    epochs = 40
    print(f">>> [JT] Treinando por {epochs} épocas...", flush=True)
    
    for epoch in range(epochs):
        model.train()
        for batch_feat, batch_lbl in joint_loader:
            feat = batch_feat[0]
            label = batch_lbl.long()
            
            optimizer.zero_grad()
            logits, attn = model(feat)
            loss = F.cross_entropy(logits, label)
            loss.backward()
            optimizer.step()
            
    # Avaliação por tarefa
    print(">>> [JT] Calculando referência por tarefa...", flush=True)
    task_test_files = ["2_4_test.csv", "0_5_test.csv", "3_1_test.csv"]
    acc_list = []
    
    for i, t_file in enumerate(task_test_files):
        d_te = pd.read_csv(os.path.join(csv_dir, t_file))
        t_ds = WSIEmbeddingDataset(d_te, emb_dir, preload=True)
        t_loader = DataLoader(t_ds, batch_size=1, shuffle=False, collate_fn=mil_collate)
        
        trues, preds = [], []
        model.eval()
        
        with torch.no_grad():
            for bf, bl in t_loader:
                feat = bf[0]
                lg, _ = model(feat)
                pred = torch.argmax(lg, dim=1).item()
                trues.append(bl[0].item())
                preds.append(pred)
        
        acc = balanced_accuracy_score(trues, preds)
        acc_list.append(acc)
        print(f"    -> JT Task {i+1} Acc: {acc:.4f}", flush=True)

    return np.mean(acc_list), np.array(acc_list)
