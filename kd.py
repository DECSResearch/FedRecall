import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, Dataset
import copy
from sklearn.metrics import accuracy_score
import numpy as np
import time
import os
import pickle
from typing import Dict, List, Tuple, Any, Optional

# 项目结构导入
from fedavg import fedavg, get_model_weights, set_model_weights
from model import model_init
from data_preprocess import get_partitioned_data
from backdoor import BackdoorAttack, evaluate_backdoor

def subtract_model(model_path: str,
                   clients_to_remove: List[int],
                   history_path: str,
                   unlearn_rounds: Optional[List[int]],
                   device: Optional[torch.device],
                   config: Optional[dict]):
    """
    通过减去指定客户端在选定轮次中累积的加权更新，从全局模型中移除其贡献（仅支持列表）

    参数:
        model_path: 保存的全局模型路径 (.pth)
        clients_to_remove: 要移除的客户端ID列表（仅列表；无int回退）
        history_path: 完整联邦学习历史路径（pickle格式，包含client_updates_history, aggregation_history）
        unlearn_rounds: 应用遗忘的轮次；如果为None，使用历史中所有可用轮次
        device: 使用的torch设备；如果为None，自动选择
        config: 模型配置字典；如果为None，将尝试从model_path推断数据集
    """
    subtract_start_time = time.time()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Removing contribution from clients {sorted(list(clients_to_remove or []))}...")

    # === Load model ===
    model_load_start = time.time()
    if config is not None:
        model = model_init(config).to(device)
    else:
        # 回退：通过路径关键字推断数据集
        lower = (model_path or "").lower()
        if 'fashion' in lower:
            data_name = 'fashion_mnist'
        elif 'cifar100' in lower:
            data_name = 'cifar100'
        elif 'cifar' in lower:
            data_name = 'cifar10'
        else:
            data_name = 'mnist'
        model = model_init({'dataset': data_name}).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    print(f"Model loading time: {time.time() - model_load_start:.2f} seconds")

    # === Load history ===
    history_load_start = time.time()
    with open(history_path, 'rb') as f:
        complete_history = pickle.load(f)
    print(f"History loading time: {time.time() - history_load_start:.2f} seconds")

    client_updates_history = complete_history['client_updates_history']
    aggregation_history = complete_history.get('aggregation_history', {})
    all_rounds = sorted(client_updates_history.keys())
    rounds = all_rounds if unlearn_rounds is None else [r for r in unlearn_rounds if r in client_updates_history]

    # === Compute contribution across rounds & clients ===
    contribution_start = time.time()
    client_contribution: Dict[str, np.ndarray] = {}

    for round_num in rounds:
        round_updates = client_updates_history.get(round_num, {})
        if not round_updates:
            continue

        total_samples = sum(c['sample_size'] for c in round_updates.values()) or 0
        if total_samples <= 0:
            # 此轮无需处理
            continue

        for client_id in clients_to_remove or []:
            if client_id not in round_updates:
                continue
            cdat = round_updates[client_id]
            c_weights = cdat['weights']
            c_size = cdat['sample_size']
            ratio = float(c_size) / float(total_samples)

            for key, val in c_weights.items():
                if key not in client_contribution:
                    client_contribution[key] = np.zeros_like(val, dtype=np.float32)
                # 累积加权贡献
                client_contribution[key] += ratio * val.astype(np.float32)

    print(f"Client contribution calculation time: {time.time() - contribution_start:.2f} seconds")

    # === Subtract weights only ===
    remove_start = time.time()
    # 快照当前权重（仅参数）
    current_weights = {name: param.detach().cpu().numpy().astype(np.float32)
                       for name, param in model.named_parameters()}

    for name, param in model.named_parameters():
        if name in client_contribution:
            adjusted = current_weights[name] - client_contribution[name]
            param.data = torch.from_numpy(adjusted).to(device)

    print(f"Client contribution removal time: {time.time() - remove_start:.2f} seconds")
    print("[BN] BatchNorm layers were ignored during subtraction. Original stats retained.")
    print(f"Total model subtraction time: {time.time() - subtract_start_time:.2f} seconds")
    print("Client contributions have been removed from the model")
    return model


def knowledge_distillation(teacher_model, student_model, train_loader, test_loader, device=None, config=None):
    kd_start_time = time.time()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    temperature = config.get("kd_temperature")
    alpha = config.get("kd_alpha")
    epochs = config.get("kd_epochs")
    lr = config.get("kd_lr", config["lr"])

    optimizer = optim.SGD(student_model.parameters(), lr=lr, momentum=0.9)

    history = {
        'epoch': [],
        'train_loss': [],
        'test_loss': [],
        'test_accuracy': [],
        'epoch_time': [],
        'backdoor_success_rate': []
    }

    teacher_model.eval()

    if config is not None and config.get("debug_print_baseline"):
        # 打印学生模型的初始化校验和与baseline指标（不影响后续训练）
        init_weights = get_model_weights(student_model)
        checksum = 0.0
        for v in init_weights.values():
            if isinstance(v, np.ndarray):
                checksum += float(np.sum(v.astype(np.float64)))
        print(f"[Debug][KD] Student init checksum: {checksum:.6e}")
        base_loss, base_acc = evaluate_model(student_model, test_loader, device)
        base_bsr = evaluate_backdoor(
            student_model, test_loader, device,
            trigger_pattern=config.get("backdoor_pattern"),
            trigger_size=config.get("backdoor_size"),
            target_label=config.get("backdoor_target")
        )
        print(f"[KD Baseline Preview] Test Loss: {base_loss:.4f}, Accuracy: {base_acc:.2f}%, Backdoor SR: {base_bsr:.2f}%")

    # Debug-only: if enabled, print baseline once and continue training
    if config is not None and config.get("debug_baseline_only"):
        print("[Debug][KD] Baseline printed (debug_baseline_only is ON), proceeding with KD training...")

    print(f"Starting knowledge distillation training, total {epochs} epochs...")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        student_model.train()
        epoch_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            student_output = student_model(data)
            with torch.no_grad():
                teacher_output = teacher_model(data)

            soft_target_loss = F.kl_div(
                F.log_softmax(student_output / temperature, dim=1),
                F.softmax(teacher_output / temperature, dim=1),
                reduction='batchmean'
            ) * (temperature ** 2)

            hard_target_loss = F.cross_entropy(student_output, target)
            loss = alpha * soft_target_loss + (1 - alpha) * hard_target_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        test_loss, test_accuracy = evaluate_model(student_model, test_loader, device)
        epoch_time = time.time() - epoch_start_time

        trigger_pattern = config.get("backdoor_pattern")
        trigger_size = config.get("backdoor_size")
        target_label = config.get("backdoor_target")

        backdoor_sr = evaluate_backdoor(
            student_model,
            test_loader,
            device,
            trigger_pattern=trigger_pattern,
            trigger_size=trigger_size,
            target_label=target_label
        )

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_accuracy)
        history['epoch_time'].append(epoch_time)
        history['backdoor_success_rate'].append(backdoor_sr)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    kd_total_time = time.time() - kd_start_time
    print(f"Total knowledge distillation time: {kd_total_time:.2f} seconds")

    history['total_time'] = kd_total_time
    return student_model, history


def evaluate_model(model, test_loader, device):
    eval_start = time.time()
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    eval_time = time.time() - eval_start
    print(f"Model evaluation time: {eval_time:.2f} seconds")
    return test_loss, accuracy
