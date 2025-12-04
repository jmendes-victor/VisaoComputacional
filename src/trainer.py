import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from sklearn.metrics import balanced_accuracy_score

from .config import DIM_FEATURES, NUM_CLASSES, BUFFER_SIZE, AKD_LAMBDA, KD_LAMBDA
from .model import CLAM_SB
from .memory import BalancedReplayBuffer
from .losses import calc_kd_loss, calc_akd_loss


def train_engine(method_name: str, tasks: list) -> tuple:
    print(f"\n{'='*60}")
    print(f" EXPERIMENTO: {method_name} | MODELO: CLAM-SB")
    print(f"{'='*60}")

    model = CLAM_SB(input_dim=DIM_FEATURES, n_classes=NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    memory = BalancedReplayBuffer(capacity=(BUFFER_SIZE if method_name != "FT" else 0))
    
    teacher_model = None
    seen_classes = []
    num_tasks = len(tasks)
    acc_matrix = np.zeros((num_tasks, num_tasks))

    for t_current in range(num_tasks):
        task = tasks[t_current]
        task_id = task['id']
        seen_classes = sorted(list(set(seen_classes + task['classes'])))
        
        print(f"> Task {task_id+1}/{num_tasks} | Classes: {task['classes']}", flush=True)

        if task_id > 0 and method_name == "OURS":
            teacher_model = deepcopy(model)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False

        epochs = 20 if task_id == 0 else 30

        for epoch in range(epochs):
            model.train()
            
            for batch_feat, batch_lbl in task['train_loader']:
                feat = batch_feat[0]
                label = batch_lbl.long()

                optimizer.zero_grad()
                logits, attn = model(feat)
                
                mask = torch.full_like(logits, -float('inf'))
                mask[:, seen_classes] = logits[:, seen_classes]
                loss = F.cross_entropy(mask, label)

                if task_id > 0 and method_name in ["ER", "OURS"]:
                    buf_feat, buf_lbl = memory.sample()
                    if buf_feat is not None:
                        r_logits, r_attn = model(buf_feat)
                        r_mask = torch.full_like(r_logits, -float('inf'))
                        r_mask[:, seen_classes] = r_logits[:, seen_classes]
                        loss += F.cross_entropy(r_mask, buf_lbl)

                        if method_name == "OURS" and teacher_model is not None:
                            with torch.no_grad():
                                t_logits, t_attn = teacher_model(buf_feat)
                            loss += KD_LAMBDA * calc_kd_loss(
                                r_logits[:, seen_classes], 
                                t_logits[:, seen_classes]
                            )
                            loss += AKD_LAMBDA * calc_akd_loss(r_attn, t_attn)

                loss.backward()
                optimizer.step()

        if method_name != "FT":
            model.eval()
            with torch.no_grad():
                memory.add_batch(task['train_loader'])

        print(f"  [Eval] Resultados Parciais:", flush=True)
        for t_eval_idx in range(t_current + 1):
            t_eval = tasks[t_eval_idx]
            trues, preds = [], []
            model.eval()
            
            with torch.no_grad():
                for bf, bl in t_eval['test_loader']:
                    feat = bf[0]
                    lg, _ = model(feat)
                    mask = torch.full_like(lg, -float('inf'))
                    mask[:, seen_classes] = lg[:, seen_classes]
                    pred = torch.argmax(mask, dim=1).item()
                    trues.append(bl[0].item())
                    preds.append(pred)
            
            acc = balanced_accuracy_score(trues, preds)
            acc_matrix[t_current, t_eval_idx] = acc
            print(f"    -> Task {t_eval_idx+1} Acc: {acc:.4f}", flush=True)

    final_accs = acc_matrix[num_tasks-1, :]
    avg_acc = np.mean(final_accs)
    
    bwt_sum = 0
    for i in range(num_tasks - 1):
        bwt_sum += (acc_matrix[num_tasks-1, i] - acc_matrix[i, i])
    bwt = bwt_sum / (num_tasks - 1) if num_tasks > 1 else 0.0
    
    return avg_acc, bwt, np.diag(acc_matrix)
