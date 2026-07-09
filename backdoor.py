import random
import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset

# === BackdoorTransform (不变) ===
# 这个类保持不变，它负责在给定的Tensor上应用触发器
class BackdoorTransform:
    def __init__(self, trigger_pattern='square', trigger_size=5, dataset_name: str = None, trigger_color: str = 'white'):
        self.trigger_pattern = trigger_pattern
        self.trigger_size = trigger_size
        self.dataset_name = (dataset_name or '').lower() if isinstance(dataset_name, str) else None
        self.trigger_color = trigger_color

    def _get_norm_stats(self, c: int):
        # Return mean/std tensors on CPU for given dataset (RGB only).
        if c != 3:
            return None, None
        if self.dataset_name in ['cifar10', 'cifar-10', 'cifar100', 'cifar-100']:
            mean = torch.tensor([125.3, 123.0, 113.9]) / 255.0
            std  = torch.tensor([63.0, 62.1, 66.7]) / 255.0
            return mean, std
        if self.dataset_name == 'svhn':
            mean = torch.tensor([0.4377, 0.4438, 0.4728])
            std  = torch.tensor([0.1980, 0.2010, 0.1970])
            return mean, std
        # Unknown dataset: no stats
        return None, None

    def _desired_pixel_color(self, c: int):
        # Define desired pixel-space color in [0,1]
        if c == 1:
            return torch.ones(1)  # white for grayscale
        # For RGB, use bright white to maximize visibility across backgrounds
        if isinstance(self.trigger_color, str) and self.trigger_color.lower() == 'red':
            return torch.tensor([1.0, 0.0, 0.0])
        # default to white
        return torch.tensor([1.0, 1.0, 1.0])

    def __call__(self, img):
        # 期望 img 是 torch.Tensor，形状为 (C, H, W)
        img = img.clone()
        c, h, w = img.shape
        # Compute color in normalized space to make it visually obvious after Normalize
        mean, std = self._get_norm_stats(c)
        if mean is not None and std is not None:
            pixel_color = self._desired_pixel_color(c).to(img.dtype)
            color = ((pixel_color - mean) / std).to(img.device)
        else:
            # Fallback: use strong positive value in current (normalized) space
            color = torch.ones(c, device=img.device)
            if c == 3:
                # push to bright across channels
                color = torch.full((3,), 3.0, device=img.device, dtype=img.dtype)

        if self.trigger_pattern == 'cross':
            cx, cy = w // 2, h // 2
            size = self.trigger_size // 2
            img[:, cy - size:cy + size + 1, cx] = color.view(c, 1)
            img[:, cy, cx - size:cx + size + 1] = color.view(c, 1)

        elif self.trigger_pattern == 'square':
            # 注意：按你的要求，square 分支保持不变
            sx = w - self.trigger_size - 2
            sy = h - self.trigger_size - 2
            img[:, sy:sy + self.trigger_size, sx:sx + self.trigger_size] = color.view(c, 1, 1)

        else:
            # 对 DBA 样式，使用 spacing 替代原来的固定 "+ 2"
            spacing = 1  # 把 gap 从原先的 2 改为 1；设为 0 表示无 gap

            if self.trigger_pattern == 'dba_dice6_bar1':
                bar_w = int(self.trigger_size); bar_h = 1
                sx = 1
                sy = 1
                sx0 = max(0, sx); sx1 = min(w, sx + bar_w)
                sy0 = max(0, sy); sy1 = min(h, sy + bar_h)
                if sx1 > sx0 and sy1 > sy0:
                    img[:, sy0:sy1, sx0:sx1] = color.view(c, 1, 1)

            elif self.trigger_pattern == 'dba_dice6_bar2':
                bar_w = int(self.trigger_size); bar_h = 1
                sx = 1 + bar_w + spacing
                sy = 1
                sx0 = max(0, sx); sx1 = min(w, sx + bar_w)
                sy0 = max(0, sy); sy1 = min(h, sy + bar_h)
                if sx1 > sx0 and sy1 > sy0:
                    img[:, sy0:sy1, sx0:sx1] = color.view(c, 1, 1)

            elif self.trigger_pattern == 'dba_dice6_bar3':
                bar_w = int(self.trigger_size); bar_h = 1
                sx = 1
                sy = 1 + bar_h + spacing
                sx0 = max(0, sx); sx1 = min(w, sx + bar_w)
                sy0 = max(0, sy); sy1 = min(h, sy + bar_h)
                if sx1 > sx0 and sy1 > sy0:
                    img[:, sy0:sy1, sx0:sx1] = color.view(c, 1, 1)

            elif self.trigger_pattern == 'dba_dice6_bar4':
                bar_w = int(self.trigger_size); bar_h = 1
                sx = 1 + bar_w + spacing
                sy = 1 + bar_h + spacing
                sx0 = max(0, sx); sx1 = min(w, sx + bar_w)
                sy0 = max(0, sy); sy1 = min(h, sy + bar_h)
                if sx1 > sx0 and sy1 > sy0:
                    img[:, sy0:sy1, sx0:sx1] = color.view(c, 1, 1)

            elif self.trigger_pattern == 'dba_dice6_bar5':
                bar_w = int(self.trigger_size); bar_h = 1
                sx = 1
                sy = 1 + 2 * (bar_h + spacing)
                sx0 = max(0, sx); sx1 = min(w, sx + bar_w)
                sy0 = max(0, sy); sy1 = min(h, sy + bar_h)
                if sx1 > sx0 and sy1 > sy0:
                    img[:, sy0:sy1, sx0:sx1] = color.view(c, 1, 1)

            elif self.trigger_pattern == 'dba_dice6_bar6':
                bar_w = int(self.trigger_size); bar_h = 1
                sx = 1 + bar_w + spacing
                sy = 1 + 2 * (bar_h + spacing)
                sx0 = max(0, sx); sx1 = min(w, sx + bar_w)
                sy0 = max(0, sy); sy1 = min(h, sy + bar_h)
                if sx1 > sx0 and sy1 > sy0:
                    img[:, sy0:sy1, sx0:sx1] = color.view(c, 1, 1)

            elif self.trigger_pattern == 'dba_dice6_all':
                bar_w = int(self.trigger_size); bar_h = 1
                positions = []
                for row in range(3):      # 3 rows
                    for col in range(2):  # 2 cols
                        sx = 1 + col * (bar_w + spacing)
                        sy = 1 + row * (bar_h + spacing)
                        positions.append((sx, sy))
                for sx, sy in positions:
                    sx0 = max(0, sx); sx1 = min(w, sx + bar_w)
                    sy0 = max(0, sy); sy1 = min(h, sy + bar_h)
                    if sx1 > sx0 and sy1 > sy0:
                        img[:, sy0:sy1, sx0:sx1] = color.view(c, 1, 1)

            elif self.trigger_pattern == 'dba_dice4_all':
                bar_w = int(self.trigger_size)
                bar_h = 1
                positions = []
                for row in range(2):
                    for col in range(2):
                        sx = 1 + col * (bar_w + spacing)
                        sy = 1 + row * (bar_h + spacing)
                        positions.append((sx, sy))
                for sx, sy in positions:
                    sx0 = max(0, sx); sx1 = min(w, sx + bar_w)
                    sy0 = max(0, sy); sy1 = min(h, sy + bar_h)
                    if sx1 > sx0 and sy1 > sy0:
                        img[:, sy0:sy1, sx0:sx1] = color.view(c, 1, 1)

        return img



# === (已修改) PoisonedDataset: 动态（On-the-Fly）注入 ===
class PoisonedDataset(Dataset):
    """
    Wrapper dataset，用于动态（on-the-fly）应用后门攻击。
    这确保了来自 base_dataset 的数据增强（如翻转、裁剪）
    在触发器被插入 *之前* 已经应用。
    
    它还在每次调用 __getitem__ 时根据 poison_ratio 动态选择要毒化的样本。
    """
    def __init__(self, base_dataset: Dataset, indices: list, 
                 poison_ratio: float, transform: BackdoorTransform, target_label: int):
        """
        base_dataset: 原始数据集 (e.g., train_dataset)
        indices: 这个子集覆盖的原始索引列表 (同 Subset.indices)
        poison_ratio: 毒化比例 (e.g., 0.3)
        transform: BackdoorTransform 实例
        target_label: 后门目标标签
        """
        self.dataset = base_dataset
        self.indices = list(indices) # 这个子集指向的原始索引
        self.poison_ratio = poison_ratio
        self.transform = transform
        self.target_label = target_label
        
        # 预先筛选出非目标标签的样本 *位置* (在子集中的位置)
        # 这有助于在 __getitem__ 中快速决策，仅在非目标样本上尝试毒化
        self.non_target_positions = {
            pos for pos in range(len(self.indices)) 
            # 检查 base_dataset 中的原始标签
            if self.dataset[self.indices[pos]][1] != self.target_label
        }
        
        if len(self.non_target_positions) == 0 and self.poison_ratio > 0:
             print(f"[Warning] PoisonedDataset: 在 client {indices[:5]}... 中没有找到非目标样本进行毒化。")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # idx 是这个子集中的位置 (0 .. len(self.indices)-1)
        orig_idx = self.indices[idx]
        
        # 1. 从基础数据集中获取样本（这将首先应用所有数据增强）
        img, label = self.dataset[orig_idx]

        # 2. 动态决定是否毒化
        # 仅当这个 *位置* 属于非目标标签，并且随机roll命中了poison_ratio时
        if idx in self.non_target_positions:
            if random.random() < self.poison_ratio:
                # 3. 动态应用后门变换 (在数据增强之后)
                img = self.transform(img)
                label = self.target_label

        # 4. 返回（可能被毒化的）图像和标签
        return img, label


# === (已修改) BackdoorAttack: 使用动态 PoisonedDataset ===
class BackdoorAttack:
    def __init__(self, target_label=7, poison_ratio=0.3, trigger_pattern='square', trigger_size=5,
                 dataset_name: str = None, trigger_color: str = 'white'):
        self.target_label = target_label
        self.poison_ratio = float(poison_ratio)
        self.trigger_pattern = trigger_pattern
        self.trigger_size = trigger_size
        # 创建一个 transform 实例，以传递给动态数据集
        self.transform = BackdoorTransform(trigger_pattern, trigger_size, dataset_name=dataset_name, trigger_color=trigger_color)

    def poison_dataset(self, dataset: Dataset) -> Dataset:
        """
        使用我们的动态 PoisonedDataset 包装器来包装所提供的dataset（可以是 Dataset 或 Subset）。
        """
        # 确定基础数据集和索引 (如果已经是Subset，则保留原始映射)
        if hasattr(dataset, 'indices') and hasattr(dataset, 'dataset'):
            base_dataset = dataset.dataset
            original_indices = list(dataset.indices)
        else:
            base_dataset = dataset
            original_indices = list(range(len(dataset)))

        n = len(original_indices)
        if n == 0 or self.poison_ratio == 0:
            return dataset  # 无需毒化

        # 注意：我们不再预先计算 poison_map 或静态的 poison_positions。
        # 新的 PoisonedDataset 将在其 __getitem__ 方法中处理动态随机采样。

        # 返回新的动态 PoisonedDataset
        return PoisonedDataset(
            base_dataset=base_dataset,
            indices=original_indices,
            poison_ratio=self.poison_ratio,
            transform=self.transform,
            target_label=self.target_label
        )


# === 评估函数 (不变) ===
# evaluate_backdoor 函数已经实现了动态评估，所以它不需要更改。
def evaluate_backdoor(model, test_loader, device, trigger_pattern='square', trigger_size=5, target_label=7,
                      dataset_name: str = None, trigger_color: str = 'white'):
    """
    在 test_loader 上评估后门成功率。
    - For non-target true samples, apply trigger and see % predicted as target_label (recall).
    - Also compute precision-ish metric by applying trigger to true target samples and counting FP.
    """
    # 注意：这个 attack 实例仅用于 *评估*，与训练时的注入是分开的
    attack = BackdoorAttack(
        target_label=target_label,
        poison_ratio=1.0, # 评估时总是应用
        trigger_pattern=trigger_pattern,
        trigger_size=trigger_size,
        dataset_name=dataset_name,
        trigger_color=trigger_color
    )

    model.eval()
    TP = 0  # true positives: 非目标样本 -> 被预测为目标
    FP = 0  # false positives: 目标样本 -> (应用触发器后) 被预测为目标
    total_non_target = 0
    total_target = 0 # 用于计算 FP

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # 1. 非目标样本 (用于计算 BDSR / Recall)
            non_target_mask = (target != target_label)
            if non_target_mask.sum() > 0:
                non_target_data = data[non_target_mask]
                total_non_target += non_target_data.shape[0]
                
                # 动态构造带触发器的对抗性样本
                # (注意: attack.transform 是 BackdoorTransform 实例)
                adv_data = torch.stack([attack.transform(img.cpu()).to(device) for img in non_target_data])
                pred = model(adv_data).argmax(dim=1)
                TP += (pred == target_label).sum().item()

            # 2. 目标样本 (用于计算 Precision 时的 FP)
            target_mask = (target == target_label)
            if target_mask.sum() > 0:
                target_data = data[target_mask]
                total_target += target_data.shape[0]
                
                # 看看触发器是否会导致目标样本被错误分类（虽然它们已经是目标了）
                adv_data_t = torch.stack([attack.transform(img.cpu()).to(device) for img in target_data])
                pred_t = model(adv_data_t).argmax(dim=1)
                # FP 在这里定义为：被预测为目标的（应用触发器后）目标样本
                FP += (pred_t == target_label).sum().item() 

    # BDSR (Recall): 在所有非目标样本上，攻击的成功率
    recall = 100. * TP / total_non_target if total_non_target > 0 else 0.0
    
    # Precision: 在所有被预测为目标的样本中，有多少是来自非目标样本的
    precision = 100. * TP / (TP + FP) if (TP + FP) > 0 else 0.0

    # 保持与您原始输出一致的打印
    print(f'Backdoor Recall (non-{target_label} samples): {recall:.2f}%')
    print(f'Backdoor Precision (predicted {target_label}): {precision:.2f}%')

    return recall