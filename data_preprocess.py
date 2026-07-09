import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split, Subset, ConcatDataset
from typing import Tuple, List, Dict, Optional
from backdoor import BackdoorAttack


class Cutout(object):
    # 在图像上随机遮挡 n_holes 个长度为 length 的方块
    def __init__(self, n_holes: int, length: int):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img


def load_dataset(dataset_name: str, data_dir: str = './data') -> Tuple[Dataset, Dataset]:
    # 加载指定数据集（MNIST/FashionMNIST/CIFAR10/...），返回训练集和测试集
    name = dataset_name.lower()

    if name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    elif name == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    elif name == 'svhn':
        # SVHN: 32x32 RGB, 常用均值/方差
        svhn_mean = (0.4377, 0.4438, 0.4728)
        svhn_std  = (0.1980, 0.2010, 0.1970)
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(svhn_mean, svhn_std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(svhn_mean, svhn_std),
        ])
        train_dataset = datasets.SVHN(root=data_dir, split='train', download=True, transform=transform_train)
        test_dataset  = datasets.SVHN(root=data_dir, split='test',  download=True, transform=transform_test)
        # 兼容后续基于 targets 的划分逻辑
        if not hasattr(train_dataset, 'targets') and hasattr(train_dataset, 'labels'):
            train_dataset.targets = train_dataset.labels
        if not hasattr(test_dataset, 'targets') and hasattr(test_dataset, 'labels'):
            test_dataset.targets = test_dataset.labels

    elif name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([125.3, 123.0, 113.9]) / 255.0,
                np.array([63.0, 62.1, 66.7]) / 255.0),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([125.3, 123.0, 113.9]) / 255.0,
                np.array([63.0, 62.1, 66.7]) / 255.0),
        ])
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    elif name == 'cifar100':
        # CIFAR-100 REFERENCE: https://zhuanlan.zhihu.com/p/144665196
        transform_train = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([125.3, 123.0, 113.9]) / 255.0,
                np.array([63.0, 62.1, 66.7]) / 255.0),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([125.3, 123.0, 113.9]) / 255.0,
                np.array([63.0, 62.1, 66.7]) / 255.0),
        ])
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)

    # ---------------- EMNIST-Balanced ----------------
    elif name in ['emnist_balanced', 'emnist-balanced']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.EMNIST(root=data_dir, split='balanced', train=True,  download=True, transform=transform)
        test_dataset  = datasets.EMNIST(root=data_dir, split='balanced', train=False, download=True, transform=transform)

    # ---------------- EMNIST-ByClass -----------------
    elif name in ['emnist_byclass', 'emnist-byclass', 'emnist-by-class']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.EMNIST(root=data_dir, split='byclass', train=True,  download=True, transform=transform)
        test_dataset  = datasets.EMNIST(root=data_dir, split='byclass', train=False, download=True, transform=transform)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_dataset, test_dataset


def create_iid_splits(dataset: Dataset, num_clients: int) -> List[Dataset]:
    # 将数据集随机均分为 IID 分区
    total_size = len(dataset)
    partition_size = total_size // num_clients
    lengths = [partition_size] * (num_clients - 1)
    lengths.append(total_size - sum(lengths))
    partitions = random_split(dataset, lengths)
    return partitions


def create_dirichlet_splits(dataset: Dataset, num_clients: int, alpha: float) -> List[Dataset]:
    # 使用 Dirichlet 分布 (参数 alpha) 将数据划分给多个客户端 (非IID)
    # 兼容 .targets / .labels
    if hasattr(dataset, 'targets'):
        raw_labels = dataset.targets
    elif hasattr(dataset, 'labels'):
        raw_labels = dataset.labels
    else:
        raise ValueError("Dataset does not have a 'targets' or 'labels' attribute")
    # 提取标签数组
    labels = raw_labels.numpy() if isinstance(raw_labels, torch.Tensor) else np.array(raw_labels)
    num_classes = len(np.unique(labels))
    # 每个类别按 Dirichlet(alpha) 采样出在各客户端上的比例分布
    label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)
    # 获取每个类别对应的数据索引列表，并打乱顺序
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    for idx_list in class_indices:
        np.random.shuffle(idx_list)
    # 按照每个类别的比例，将索引切分给各客户端
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for c_idx, fracs in zip(class_indices, label_distribution):
        # 计算当前类别在各客户端的分割点索引
        split_points = (np.cumsum(fracs)[:-1] * len(c_idx)).astype(int)
        splits = np.split(c_idx, split_points)  # 切分得到每个客户端该类别的索引片段
        for i, part in enumerate(splits):
            client_indices[i].extend(part.tolist())
    # 构建每个客户端对应的数据子集
    client_datasets = [Subset(dataset, indices) for indices in client_indices]
    return client_datasets


def create_data_loaders(partitions, batch_size=32, test_dataset=None):
    import os
    cpu_cnt = max(os.cpu_count() or 2, 2)
    # 在 Windows/Notebook 环境下将 num_workers 设为 0，避免子进程异常退出
    use_workers = 0 if os.name == 'nt' else min(cpu_cnt, 8)

    # 基础参数（num_workers=0 时，不能传 prefetch_factor/persistent_workers）
    base_kwargs = dict(
        batch_size=batch_size,
        num_workers=use_workers,
    )

    # 仅在有 worker 时开启 pin_memory / persistent_workers / prefetch_factor
    if use_workers > 0:
        base_kwargs.update(
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

    train_loaders = [
        DataLoader(partition, shuffle=True, **base_kwargs)
        for partition in partitions
    ]

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, shuffle=False, **base_kwargs)

    return train_loaders, test_loader


def get_partitioned_data(dataset_name: str,
                         num_clients: int,
                         batch_size: int = 32,
                         partition_type: str = 'iid',
                         shards_per_client: int = 2,
                         data_dir: str = './data',
                         backdoor_config: Dict[int, Dict] = None,
                         cut_ratio: float = 1.0,
                         dirichlet_alpha: float = None,
                         build_kd_from_clean: bool = False,
                         kd_clients: Optional[List[int]] = None,
                         kd_batch_size: Optional[int] = None) -> Tuple[List[DataLoader], DataLoader, Optional[DataLoader]]:
    # 获取划分后的训练集 DataLoader 列表和测试集 DataLoader
    train_dataset, test_dataset = load_dataset(dataset_name, data_dir)

    # 根据参数选择划分方式：优先使用 Dirichlet 非IID 划分，否则按指定类型划分
    if dirichlet_alpha is not None:
        partitions = create_dirichlet_splits(train_dataset, num_clients, dirichlet_alpha)
    elif partition_type == 'iid':
        partitions = create_iid_splits(train_dataset, num_clients)
    elif partition_type == 'non-iid':
        # 未指定 alpha 时，默认退回 IID 划分
        partitions = create_iid_splits(train_dataset, num_clients)
    else:
        raise ValueError(f"Unknown partition type: {partition_type}")

    # 可选：按比例裁剪每个客户端的数据集大小
    if cut_ratio < 1.0:
        for i in range(len(partitions)):
            full_dataset = partitions[i]
            indices = list(range(len(full_dataset)))
            np.random.shuffle(indices)
            cut_len = int(len(indices) * cut_ratio)
            partitions[i] = Subset(full_dataset, indices[:cut_len])

    # 如需：在注入后门之前构建“干净 KD 数据集”
    kd_loader = None
    if build_kd_from_clean:
        poisoned_clients = set((backdoor_config or {}).keys())
        if kd_clients is None:
            kd_clients = [cid for cid in range(num_clients) if cid not in poisoned_clients]
        clean_parts = [partitions[cid] for cid in kd_clients if 0 <= cid < len(partitions)]
        if clean_parts:
            kd_dataset = ConcatDataset(clean_parts)
            cpu_cnt = max(os.cpu_count() or 2, 2)
            use_workers = 0 if os.name == 'nt' else min(cpu_cnt, 8)
            base_kwargs = dict(
                batch_size=kd_batch_size or batch_size,
                num_workers=use_workers,
                shuffle=True
            )
            if use_workers > 0:
                base_kwargs.update(
                    pin_memory=True,
                    persistent_workers=True,
                    prefetch_factor=2,
                )
            kd_loader = DataLoader(kd_dataset, **base_kwargs)

    # 如提供了 backdoor_config 字典，则对指定的每个客户端注入后门样本
    if backdoor_config:
        for client_id, params in backdoor_config.items():
            if 0 <= client_id < num_clients:
                # 提取当前客户端的后门参数，如未提供某项则使用默认值
                target_label = params.get('target_label', 7)
                poison_ratio = params.get('poison_ratio', 0.3)
                trigger_pattern = params.get('trigger_pattern')  # 'cross'
                trigger_size = params.get('trigger_size', 5)
                print(f"Implementing backdoor attack on client {client_id}")
                print(f"Backdoor parameters for client {client_id}: "
                      f"target_label={target_label}, poison_ratio={poison_ratio}, "
                      f"trigger_pattern={trigger_pattern}, trigger_size={trigger_size}")
                # 利用指定参数创建后门攻击对象并对该客户端的数据集进行投毒
                attack = BackdoorAttack(target_label=target_label,
                                        poison_ratio=poison_ratio,
                                        trigger_pattern=trigger_pattern,
                                        trigger_size=trigger_size,
                                        dataset_name=dataset_name,
                                        trigger_color='white')
                partitions[client_id] = attack.poison_dataset(partitions[client_id])
            else:
                print(f"Warning: Invalid backdoor client id {client_id} (ignored).")

    train_loaders, test_loader = create_data_loaders(partitions, batch_size, test_dataset)
    return train_loaders, test_loader, kd_loader
