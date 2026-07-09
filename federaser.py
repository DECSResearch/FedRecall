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

from fedavg import fedavg, get_model_weights, set_model_weights
from model import model_init
from data_preprocess import get_partitioned_data
from server import Server
from backdoor import BackdoorAttack, evaluate_backdoor
from path_utils import resolve_output_path

class FedEraser:
    def __init__(self, 
                 model_path: str,
                 history_path: str,
                 config: dict,
                 history_dir: str = './fl_history',
                 save_dir: str = './unlearned_models',
                 device: Optional[torch.device] = None):
        """
        初始化FedEraser

        参数:
            model_path: 联邦学习模型路径
            history_path: 完整联邦学习历史路径
            history_dir: 联邦学习历史目录
            save_dir: 移除客户端贡献后保存模型的目录
            device: 计算设备
        """
        # 初始化计时器
        self.timers = {
            'init': 0.0,
            'data_loading': 0.0,
            'aggregation': 0.0,
            'client_training': 0.0,
            'calibration': 0.0,
            'evaluation': 0.0,
            'rounds': {}
        }
        
        # 记录初始化开始时间
        init_start_time = time.time()
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.model_path = model_path
        self.history_path = history_path
        self.config = config
        self.history_dir = history_dir
        # 保存目录解析到 MyDrive（Colab）
        self.save_dir = resolve_output_path(save_dir)
        
        # 创建保存目录
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # Load global model
        model_load_start = time.time()
        self.global_model = model_init(self.config).to(self.device)
        self.global_model.load_state_dict(torch.load(model_path, map_location=self.device))
        model_load_time = time.time() - model_load_start
        print(f"Model loading time: {model_load_time:.2f} seconds")
        
        # 加载历史记录
        history_load_start = time.time()
        with open(history_path, 'rb') as f:
            complete_data = pickle.load(f)
            
        self.client_updates_history = complete_data['client_updates_history']
        self.aggregation_history = complete_data['aggregation_history']
        self.current_round = complete_data['current_round']
        history_load_time = time.time() - history_load_start
        print(f"History loading time: {history_load_time:.2f} seconds")
        
        # 获取客户端ID列表
        self.client_ids = set()
        for round_num in self.client_updates_history:
            self.client_ids.update(self.client_updates_history[round_num].keys())
        self.client_ids = sorted(list(self.client_ids))
        
        # 记录总初始化时间
        self.timers['init'] = time.time() - init_start_time
        
        print(f"Loaded federated learning history, total rounds: {self.current_round}, number of clients: {len(self.client_ids)}")
        print(f"FedEraser initialization time: {self.timers['init']:.2f} seconds")

    # ------------------------ 已修改：仅支持列表 ------------------------
    def select_remaining_clients(self, clients_to_remove: List[int]) -> List[int]:
        """
        选择除要移除的客户端外的所有客户端

        参数:
            clients_to_remove: 要移除的客户端ID（仅列表）

        返回:
            剩余客户端ID列表
        """
        remove_set = set(clients_to_remove or [])
        return [client_id for client_id in self.client_ids if client_id not in remove_set]
    # -------------------------------------------------------------------

    def get_client_model(self, client_id: int, round_num: int) -> Dict[str, np.ndarray]:
        """
        获取特定轮次的客户端模型权重
        
        参数:
            client_id: 客户端ID
            round_num: 轮次编号
            
        返回:
            客户端模型权重
        """
        if round_num in self.client_updates_history and client_id in self.client_updates_history[round_num]:
            return self.client_updates_history[round_num][client_id]['weights']
        else:
            raise ValueError(f"Client {client_id} does not exist in round {round_num}")
    
    def get_global_model_at_round(self, round_num: int) -> Dict[str, np.ndarray]:
        """
        获取特定轮次的全局模型权重
        
        参数:
            round_num: 轮次编号
            
        返回:
            全局模型权重
        """
        if round_num in self.aggregation_history:
            return self.aggregation_history[round_num]
        else:
            raise ValueError(f"Global model for round {round_num} does not exist")
    
    def evaluate_model(self, model: torch.nn.Module, test_loader: DataLoader) -> Tuple[float, float]:
        """
        在测试集上评估模型性能
        
        参数:
            model: 要评估的模型
            test_loader: 测试数据加载器
            
        返回:
            (test_loss, test_accuracy)
        """
        # 记录评估开始时间
        eval_start = time.time()
        
        model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        
        # 记录评估时间
        eval_time = time.time() - eval_start
        self.timers['evaluation'] += eval_time
        
        print(f"Model evaluation time: {eval_time:.2f} seconds")
        
        return test_loss, accuracy
    
    def aggregate_remaining_clients(self, round_num: int, 
                                   remaining_clients: List[int]) -> Dict[str, np.ndarray]:
        """
        聚合剩余客户端的权重
        
        参数:
            round_num: 轮次编号
            remaining_clients: 剩余客户端ID列表
            
        返回:
            聚合后的全局模型权重
        """
        # 记录聚合开始时间
        agg_start = time.time()
        
        weights_list = []
        sample_sizes = []
        
        for client_id in remaining_clients:
            if client_id in self.client_updates_history[round_num]:
                client_data = self.client_updates_history[round_num][client_id]
                weights_list.append(client_data['weights'])
                sample_sizes.append(client_data['sample_size'])
        
        # 使用FedAvg进行聚合
        result = fedavg(weights_list, sample_sizes)
        
        # 记录聚合时间
        agg_time = time.time() - agg_start
        self.timers['aggregation'] += agg_time
        
        print(f"Remaining clients aggregation time (Round {round_num}): {agg_time:.2f} seconds")
        
        return result
    
    def train_clients_one_step(self, 
                              client_models: List[Dict[str, np.ndarray]], 
                              global_model: Dict[str, np.ndarray], 
                              train_loaders: List[DataLoader],
                              lr: float = 0.01,
                              epochs: int = 1) -> List[Dict[str, np.ndarray]]:
        """
        对客户端模型进行一步训练
        
        参数:
            client_models: 客户端模型权重列表
            global_model: 全局模型权重
            train_loaders: 客户端训练数据加载器列表
            lr: 学习率
            epochs: 训练轮数
            
        返回:
            训练后的客户端模型权重列表
        """
        # 记录训练开始时间
        train_start = time.time()
        
        new_client_models = []
        client_times = []
        
        for i, (client_weights, train_loader) in enumerate(zip(client_models, train_loaders)):
            # 记录客户端训练开始时间
            client_start = time.time()
            
            # 初始化模型
            model = model_init(self.config).to(self.device)
            
            # 加载全局模型权重（鲁棒类型与dtype转换）
            set_model_weights(model, global_model, self.device)
            
            # 设置优化器
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)
            
            # 训练模型
            model.train()
            for epoch in range(epochs):
                # epoch_start = time.time()  # (未使用的变量；保持代码简洁)
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()
            
            # 提取权重
            new_weights = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
            
            new_client_models.append(new_weights)
            
            # 记录客户端训练时间
            client_time = time.time() - client_start
            client_times.append(client_time)
            print(f"Client {i} total training time: {client_time:.2f} seconds")
        
        # 记录客户端训练统计信息
        if client_times:
            print(f"Client training time statistics - Min: {min(client_times):.2f}s, Max: {max(client_times):.2f}s, Average: {sum(client_times)/len(client_times):.2f}s")
        
        # 记录总训练时间
        train_time = time.time() - train_start
        self.timers['client_training'] += train_time
        
        print(f"Total training time for all clients: {train_time:.2f} seconds")
        
        return new_client_models
    
    def unlearning_step_once(self,
                             old_client_models: List[Dict[str, np.ndarray]],
                             new_client_models: List[Dict[str, np.ndarray]],
                             global_model_before_forget: Dict[str, np.ndarray],
                             global_model_after_forget: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        使用校准执行一步遗忘，调整模型权重
        
        参数:
            old_client_models: 旧客户端模型权重列表
            new_client_models: 新客户端模型权重列表
            global_model_before_forget: 遗忘前的全局模型权重
            global_model_after_forget: 遗忘后的全局模型权重
            
        返回:
            校准后的全局模型权重
        """
        # 记录校准开始时间
        calib_start = time.time()
        
        old_param_update = {}  # oldCM - oldGM_t
        new_param_update = {}  # newCM - newGM_t
        return_model_state = {}  # newGM_t + ||oldCM - oldGM_t|| * (newCM - newGM_t) / ||newCM - newGM_t||
        
        assert len(old_client_models) == len(new_client_models)
        
        for layer in global_model_before_forget.keys():
            # 跳过BatchNorm缓冲区（running_mean, running_var, num_batches_tracked）以避免不匹配
            if any(bn_key in layer for bn_key in ['running_mean', 'running_var', 'num_batches_tracked']):
                return_model_state[layer] = global_model_after_forget[layer]
                continue
            # 初始化参数更新
            old_param_update[layer] = np.zeros_like(global_model_before_forget[layer])
            new_param_update[layer] = np.zeros_like(global_model_before_forget[layer])
            
            # 计算客户端模型平均值
            for client_idx in range(len(old_client_models)):
                old_param_update[layer] += old_client_models[client_idx][layer]
                new_param_update[layer] += new_client_models[client_idx][layer]
            
            old_param_update[layer] /= len(old_client_models)  # oldCM
            new_param_update[layer] /= len(new_client_models)  # newCM
            
            # 计算更新方向
            old_param_update[layer] = old_param_update[layer] - global_model_before_forget[layer]  # oldCM - oldGM_t
            new_param_update[layer] = new_param_update[layer] - global_model_after_forget[layer]   # newCM - newGM_t
            
            # 计算步长和方向
            old_norm = np.linalg.norm(old_param_update[layer])  # ||oldCM - oldGM_t||
            new_norm = np.linalg.norm(new_param_update[layer])  # ||newCM - newGM_t||
            
            # 避免除零
            if new_norm > 1e-10:
                step_direction = new_param_update[layer] / new_norm  # (newCM - newGM_t) / ||newCM - newGM_t||
                return_model_state[layer] = global_model_after_forget[layer] + old_norm * step_direction
            else:
                # 如果新更新几乎为零，则不校准
                return_model_state[layer] = global_model_after_forget[layer]
        
        # 记录校准时间
        calib_time = time.time() - calib_start
        self.timers['calibration'] += calib_time
        
        print(f"FedEraser calibration time: {calib_time:.2f} seconds")
        
        return return_model_state

    # ------------------------ 已修改：仅支持列表 ------------------------
    def run(self, 
            clients_to_remove: List[int],
            batch_size: int,
            lr: float,
            epochs: int,
            train_loaders: List[DataLoader],
            test_loader: DataLoader,
            save_model_path: Optional[str],
            unlearn_rounds: List[int],
            max_rounds: int,
            init_state_dict: Optional[Dict[str, torch.Tensor]] = None
        ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        使用全新模型运行FedEraser算法（串行推进）
        接口现在要求传入要移除的客户端ID列表

        参数:
            clients_to_remove: List[int]，要移除其贡献的客户端
            batch_size, lr, epochs, train_loaders, test_loader, save_model_path, unlearn_rounds, max_rounds:
                语义与之前相同

        返回:
            (final_weights, history)
        """
        total_start_time = time.time()
        print(f"Starting FedEraser, removing client IDs: {sorted(list(clients_to_remove or []))}")

        # ---------- 解析轮次 ----------
        if unlearn_rounds is None:
            if max_rounds is not None and 0 < max_rounds < self.current_round:
                unlearn_rounds = list(range(max_rounds))
            else:
                unlearn_rounds = list(range(self.current_round))
        else:
            unlearn_rounds = [r for r in unlearn_rounds if 0 <= r < self.current_round]
        unlearn_rounds = sorted(unlearn_rounds)

        # ---------- 剩余客户端和加载器映射（鲁棒） ----------
        remaining_clients = self.select_remaining_clients(clients_to_remove)

        # 安全地构建cid->loader映射，不改变外部契约
        cid_to_loader: Dict[int, DataLoader] = {}
        if len(train_loaders) == len(self.client_ids):
            # 常见情况：train_loaders与self.client_ids顺序对齐
            for cid, loader in zip(self.client_ids, train_loaders):
                cid_to_loader[cid] = loader
        else:
            # 回退：在可能的情况下假设index==cid
            for cid in self.client_ids:
                if 0 <= cid < len(train_loaders):
                    cid_to_loader[cid] = train_loaders[cid]

        selected_train_loaders = [cid_to_loader[cid] for cid in remaining_clients if cid in cid_to_loader]
        if not selected_train_loaders:
            raise RuntimeError("No matching train loaders for remaining clients; please ensure client-loader alignment.")

        # ---------- 历史字典 ----------
        history = {
            'round': [],
            'test_loss': [],
            'test_accuracy': [],
            'round_time': [],
            'backdoor_success_rate': []
        }

        # ---------- 初始权重和基线评估（使用全新随机初始化，或共享初始化） ----------
        if init_state_dict is not None:
            initial_model = model_init(self.config).to(self.device)
            initial_model.load_state_dict(init_state_dict)
            current_weights = {k: v.clone().detach().cpu().numpy() for k, v in initial_model.state_dict().items()}
        else:
            initial_model = model_init(self.config).to(self.device)
            current_weights = {k: v.clone().detach().cpu().numpy() for k, v in initial_model.state_dict().items()}

        # 使用current_weights在round = -1处评估基线（随机初始化基线）
        baseline_model = model_init(self.config).to(self.device)
        set_model_weights(baseline_model, current_weights, self.device)
        if self.config.get("debug_print_baseline"):
            # 计算一个简洁的初始化校验和，便于与其他分支对比
            checksum = 0.0
            for v in current_weights.values():
                if isinstance(v, np.ndarray):
                    checksum += float(np.sum(v.astype(np.float64)))
            print(f"[Debug][FedEraser] Init checksum: {checksum:.6e}")
        test_loss, test_accuracy = self.evaluate_model(baseline_model, test_loader)
        backdoor_sr = evaluate_backdoor(
            baseline_model,
            test_loader,
            self.device,
            trigger_pattern=self.config.get("backdoor_pattern"),
            trigger_size=self.config.get("backdoor_size"),
            target_label=self.config.get("backdoor_target")
        )
        print(f"Round -1 (baseline): Test Loss = {test_loss:.4f}, Accuracy = {test_accuracy:.2f}%, Backdoor SR = {backdoor_sr:.2f}%")
        history['round'].append(-1)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_accuracy)
        history['round_time'].append(0.0)
        history['backdoor_success_rate'].append(backdoor_sr)
        
        # ---------- Debug: baseline-only flag now only prints; no early return ----------
        if self.config.get("debug_baseline_only"):
            print("[Debug] Baseline printed (debug_baseline_only is ON), proceeding with training...")

        # ---------- 轮次串行推进 ----------
        for round_idx in unlearn_rounds:
            if round_idx not in self.client_updates_history or round_idx not in self.aggregation_history:
                print(f"[warn] Missing history for round {round_idx}; skipping this round.")
                continue

            round_start_time = time.time()
            try:
                print(f"--- FedEraser processing round {round_idx} (serial) ---")

                old_global_weights = self.get_global_model_at_round(round_idx)

                # 收集此轮次剩余客户端的旧客户端模型
                old_client_models: List[Dict[str, np.ndarray]] = []
                for cid in remaining_clients:
                    if cid in self.client_updates_history[round_idx]:
                        old_client_models.append(self.get_client_model(cid, round_idx))
                if not old_client_models:
                    print(f"[warn] Round {round_idx} has no valid client updates; skipping this round.")
                    continue

                # 从current_weights开始对客户端进行一步训练（串行）
                client_models_init = [copy.deepcopy(current_weights) for _ in selected_train_loaders]
                client_trainers = self.train_clients_one_step(
                    client_models=client_models_init,
                    global_model=current_weights,   # 串行推进关键
                    train_loaders=selected_train_loaders,
                    lr=lr,
                    epochs=epochs
                )

                sample_sizes = [len(t.dataset) for t in selected_train_loaders]
                new_global_weights = fedavg(client_trainers, sample_sizes)

                # 一步校准以移除已遗忘客户端的影响
                calibrated_weights = self.unlearning_step_once(
                    old_client_models=old_client_models,
                    new_client_models=client_trainers,
                    global_model_before_forget=old_global_weights,
                    global_model_after_forget=new_global_weights
                )

                # 更新串行状态
                current_weights = calibrated_weights

                # 评估当前串行模型
                new_model = model_init(self.config).to(self.device)
                set_model_weights(new_model, current_weights, self.device)
                test_loss, test_accuracy = self.evaluate_model(new_model, test_loader)
                backdoor_sr = evaluate_backdoor(
                    new_model,
                    test_loader,
                    self.device,
                    trigger_pattern=self.config.get("backdoor_pattern"),
                    trigger_size=self.config.get("backdoor_size"),
                    target_label=self.config.get("backdoor_target")
                )

            except Exception as e:
                print(f"Error during round {round_idx}: {e}")
                test_loss, test_accuracy, backdoor_sr = 0.0, 0.0, 0.0

            print(f"Round {round_idx}: Test Loss = {test_loss:.4f}, Accuracy = {test_accuracy:.2f}%, Backdoor SR = {backdoor_sr:.2f}%")
            round_time = time.time() - round_start_time
            history['round'].append(round_idx)
            history['test_loss'].append(test_loss)
            history['test_accuracy'].append(test_accuracy)
            history['round_time'].append(round_time)
            history['backdoor_success_rate'].append(backdoor_sr)

        # ---------- 完成 ----------
        final_weights = current_weights if unlearn_rounds else None
        if final_weights is not None:
            final_model = model_init(self.config).to(self.device)
            set_model_weights(final_model, final_weights, self.device)
            test_loss, test_accuracy = self.evaluate_model(final_model, test_loader)
            if save_model_path:
                save_model_path = resolve_output_path(save_model_path)
                torch.save(final_model.state_dict(), save_model_path)

        total_time = time.time() - total_start_time
        history['total_time'] = total_time
        return final_weights, history
    # -------------------------------------------------------------------
