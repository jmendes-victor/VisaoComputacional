import random
import torch


class BalancedReplayBuffer:
    def __init__(self, capacity: int = 42):
        self.capacity = capacity
        self.memory = {}

    def add_batch(self, loader) -> None:
        for bf, bl in loader:
            feat = bf[0].detach().cpu()
            lbl = bl[0].item()
            if lbl not in self.memory:
                self.memory[lbl] = []
            self.memory[lbl].append({'feat': feat, 'label': lbl})
        self._prune()

    def _prune(self) -> None:
        if len(self.memory) == 0:
            return
            
        classes = list(self.memory.keys())
        limit_per_class = self.capacity // len(classes)
        if limit_per_class < 1:
            limit_per_class = 1
        
        for cls in classes:
            items = self.memory[cls]
            if len(items) > limit_per_class:
                random.shuffle(items)
                self.memory[cls] = items[:limit_per_class]

    def sample(self):
        all_classes = list(self.memory.keys())
        if not all_classes:
            return None, None
            
        chosen_cls = random.choice(all_classes)
        if not self.memory[chosen_cls]:
            return None, None
            
        s = random.choice(self.memory[chosen_cls])
        return s['feat'], torch.tensor([s['label']]).long()
