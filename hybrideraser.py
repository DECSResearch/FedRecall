import torch 
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
# (已添加) 导入 DataLoader 和 random_split。
from torch.utils.data import DataLoader, random_split
import copy
import numpy as np
import time
import os
import pickle
from typing import Dict, List, Tuple, Optional

from fedavg import fedavg, get_model_weights, set_model_weights
from model import model_init
from data_preprocess import get_partitioned_data
from server import Server
from federaser import FedEraser
from kd import subtract_model, knowledge_distillation
from backdoor import evaluate_backdoor
from path_utils import resolve_output_path
 


class HybridEraser:
    def __init__(self,
                 model_path: str,
                 history_path: str,
                 config: dict,
                 device: Optional[torch.device] = None,
                 kd_epochs_per_round: int = 2,
                 aggregation_interval: int = 5,
                 aggregation_alpha: float = 0.5,
                 use_dynamic_alpha: bool = True,
                 save_dir: str = './hybrid_results'):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # === 新增：混合分支开关 ===
        # 仅保留 'federaser+kd'
        self.hybrid_branch = 'federaser+kd'

        # 兼容键名：优先使用 aggregation_interval；否则回退 hybrid_aggregation_interval；再否则使用默认值 5
        if 'aggregation_interval' in config:
            self.aggregation_interval = config['aggregation_interval']
        elif 'hybrid_aggregation_interval' in config:
            self.aggregation_interval = config['hybrid_aggregation_interval']
        else:
            self.aggregation_interval = 5

        self.model_path = model_path
        self.history_path = history_path
        self.kd_epochs_per_round = kd_epochs_per_round
        self.aggregation_alpha = aggregation_alpha
        self.use_dynamic_alpha = use_dynamic_alpha
        # 将保存目录解析到 MyDrive（在 Colab 环境）
        self.save_dir = resolve_output_path(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        # 是否在聚合后重估 BN 统计（默认关闭，按需开启）
        self.reestimate_bn: bool = bool(config.get("reestimate_bn", False))
        # 仅在首次聚合后执行一次 BN 重估
        self._bn_reestimated_once: bool = False
        # 仅按验证集准确率进行聚合（开关 + 模式）
        # accuracy_only_mode: 'ratio' -> alpha = fed_acc / (fed_acc + kd_acc)
        #                      'argmax' -> alpha = 1.0 if fed_acc >= kd_acc else 0.0
        self.accuracy_only_agg: bool = bool(config.get("accuracy_only_agg", False))
        # 强制仅使用比例分配（不支持胜者全拿）
        self.accuracy_only_mode: str = "ratio"

        # KD 教师采用原始全局模型
        self.teacher_model = model_init(config).to(self.device)
        self.teacher_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.teacher_model.eval()
        # 共享随机初始化权重：用于让 Fed 分支与 KD 学生在首次聚合前从同一起点出发
        _shared_init_model = model_init(config).to(self.device)
        self.shared_init_weights = get_model_weights(_shared_init_model)

        with open(history_path, 'rb') as f:
            complete_history = pickle.load(f)
        self.client_updates_history = complete_history['client_updates_history']
        self.aggregation_history = complete_history['aggregation_history']

        self.dataset = config['dataset']

        # --- Dynamic alpha internal state (EMA, history, smoothing) ---
        self._dyn_prev_alpha: float = float(self.aggregation_alpha)
        self._dyn_fed_ema: Optional[float] = None
        self._dyn_kd_ema: Optional[float] = None
        self._dyn_fed_hist: List[float] = []
        self._dyn_kd_hist: List[float] = []
        self._dyn_total_rounds: int = 0

        # === 客户端侧分支：按 hybrid_branch 实例化 ===
        self.branch = FedEraser(model_path=model_path,
                                history_path=history_path,
                                config=config,
                                history_dir=os.path.join(self.save_dir, 'history'),
                                save_dir=os.path.join(self.save_dir, 'models'),
                                device=self.device)

        self.remaining_clients: Optional[List[int]] = None
        self.student_model: Optional[nn.Module] = None
        self.train_loaders: List[DataLoader] = []

        # (已修改) 这些将在 .run() 中被设置为 85% 测试集、10% KD训练集 和 5% 验证集。
        self.test_loader: DataLoader = None
        self.val_loader: DataLoader = None
        self.kd_loader: DataLoader = None

        self.history = {
            'round': [],
            'test_loss': [],
            'test_accuracy': [],
            'backdoor_success_rate': [],
            # 追加：分支前评估与KD前评估（每轮）
            'fed_round': [],
            'fed_loss': [],
            'fed_accuracy': [],
            'fed_backdoor_success_rate': [],
            'kd_round': [],
            'kd_loss': [],
            'kd_accuracy': [],
            'kd_backdoor_success_rate': [],
            # 追加：聚合后指标（与上面的 test_* 等价，单独留存便于导出）
            'agg_round': [],
            'agg_loss': [],
            'agg_accuracy': [],
            'agg_backdoor_success_rate': []
        }

    def _aggregate_weights(self, weights_a: Dict[str, np.ndarray],
                           weights_b: Dict[str, np.ndarray], alpha: float) -> Dict[str, np.ndarray]:
        aggregated = {}
        bn_keys = ("running_mean", "running_var", "num_batches_tracked")
        for key in weights_a:
            if key in weights_b:
                wa = weights_a[key]
                wb = weights_b[key]

                if not isinstance(wa, np.ndarray):
                    wa = np.array(wa, dtype=np.float32)
                if not isinstance(wb, np.ndarray):
                    wb = np.array(wb, dtype=np.float32)

                if wa.shape != wb.shape:
                    # 尝试广播标量权重（例如 batchnorm 的 num_batches_tracked）。
                    if wa.ndim == 0 and wb.ndim > 0:
                        wa = np.broadcast_to(wa, wb.shape)
                    elif wb.ndim == 0 and wa.ndim > 0:
                        wb = np.broadcast_to(wb, wa.shape)
                    elif wa.shape != wb.shape:
                        # 如果形状仍然不匹配，发出警告并跳过（保留 a）
                        print(f"Warning: Skipping aggregation for key {key} due to shape mismatch: {wa.shape} vs {wb.shape}")
                        aggregated[key] = np.array(weights_a[key], dtype=np.float32)
                        continue

                # BN 运行统计与普通参数统一按 alpha 加权聚合；对计数进行取整
                out = alpha * wa + (1.0 - alpha) * wb
                if "num_batches_tracked" in key:
                    out = np.round(out).astype(np.int64).astype(np.float32)
                aggregated[key] = np.array(out, dtype=np.float32)
            else:
                val = weights_a[key]
                if not isinstance(val, np.ndarray):
                    val = np.array(val, dtype=np.float32)
                aggregated[key] = val
        return aggregated
    
    def _reestimate_bn_stats(self, model: nn.Module, loader: Optional[DataLoader], max_batches: int = 50) -> None:
        if loader is None:
            return
        was_training = model.training
        model.train()
        seen = 0
        with torch.no_grad():
            for data, _ in loader:
                data = data.to(self.device)
                _ = model(data)
                seen += 1
                if seen >= max_batches:
                    break
        if not was_training:
            model.eval()

    def _evaluate_performance(self, model: nn.Module, loader: Optional[DataLoader] = None) -> Tuple[float, float, float]:
        """ (已修改) 使用指定的 loader (默认为 self.test_loader) 评估模型 """
        # 如果未指定 loader，则使用 85% 的测试集
        eval_loader = loader if loader is not None else self.test_loader
        
        if eval_loader is None or len(eval_loader.dataset) == 0:
            print("Warning: _evaluate_performance called with no data.")
            return 0.0, 0.0, 0.0

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in eval_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)
                
        test_loss /= total if total > 0 else 1
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        # 从 config 中获取目标标签/触发器参数（尽量兼容你的 backdoor.py）
        target_label = self.config.get("backdoor_target", 7)
        backdoor_sr = evaluate_backdoor(
            model, eval_loader, self.device,
            target_label=target_label,
            trigger_pattern=self.config.get("backdoor_pattern", 'square'),
            trigger_size=self.config.get("backdoor_size", 5),
            dataset_name=self.dataset,
            trigger_color='white'
        )
        
        return test_loss, accuracy, backdoor_sr

    def run(self,
            clients_to_remove: List[int],
            train_loaders: List[DataLoader],
            test_loader: DataLoader,  # 测试集（由外部传入）
            val_loader: DataLoader,   # 验证集（由外部传入，用于动态 alpha）
            kd_loader: DataLoader,    # KD 训练集（来自 unlearn_clients 原始分片）
            max_rounds: Optional[int],
            batch_size: int,
            lr: float,
            local_epochs: int,
            kd_temperature: float,
            kd_alpha: float,
            output_excel_path: Optional[str] = None) -> Tuple[nn.Module, Dict]:
        
        total_rounds = len(self.aggregation_history)
        rounds_to_process = list(range(min(max_rounds, total_rounds))) if max_rounds else list(range(total_rounds))
        
        # 统一客户端选择与 loader 对齐
        self.remaining_clients = self.branch.select_remaining_clients(clients_to_remove)
        # train_loaders 为全体客户端的 loader 列表，按 remaining_clients 筛
        self.train_loaders = [train_loaders[cid] for cid in self.remaining_clients]
        
        # (已修改) 设置验证集、测试集和KD训练集
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.kd_loader = kd_loader

        # 学生初始化（KD 分支）
        self.student_model = model_init(self.config).to(self.device)
        # 使用共享的随机初始化权重作为 KD 学生的起点
        self.student_model = set_model_weights(self.student_model, self.shared_init_weights, self.device)
        self.student_model.train()

        current_branch_weights: Optional[Dict[str, np.ndarray]] = None
        current_branch_model: Optional[nn.Module] = None
        # 使用共享的随机初始化权重作为 Fed 分支首次轮次的起点
        current_branch_weights = dict(self.shared_init_weights)

        # For dynamic alpha phase scheduling
        self._dyn_total_rounds = len(rounds_to_process)

        # Debug-only: if enabled, print baseline once and continue
        if self.config.get("debug_baseline_only"):
            print("[Debug][HybridEraser] Baseline printed (debug_baseline_only is ON), proceeding with training/aggregation...")
            tmp_model = model_init(self.config).to(self.device)
            tmp_model = set_model_weights(tmp_model, self.shared_init_weights, self.device)
            b_loss0, b_acc0, b_bsr0 = self._evaluate_performance(tmp_model)
            print(f"[Hybrid Baseline] Test Loss: {b_loss0:.4f}, Accuracy: {b_acc0:.2f}%, Backdoor SR: {b_bsr0:.2f}%")
        
        # Debug print only: show that Fed 和 KD 使用同一共享初始化，并打印一次基线，不中断训练
        if self.config.get("debug_print_baseline"):
            # 计算共享初始化权重的校验和
            shared_sum = 0.0
            for v in self.shared_init_weights.values():
                if isinstance(v, np.ndarray):
                    shared_sum += float(np.sum(v.astype(np.float64)))
            # 学生模型与 Fed 分支起点的校验和
            stud_sum = 0.0
            for v in get_model_weights(self.student_model).values():
                if isinstance(v, np.ndarray):
                    stud_sum += float(np.sum(v.astype(np.float64)))
            fed_sum = 0.0
            for v in current_branch_weights.values():
                if isinstance(v, np.ndarray):
                    fed_sum += float(np.sum(v.astype(np.float64)))
            print(f"[Debug][Hybrid] Shared init checksum: {shared_sum:.6e}")
            print(f"[Debug][Hybrid] Student init checksum: {stud_sum:.6e}")
            print(f"[Debug][Hybrid] Fed start checksum:    {fed_sum:.6e}")
            # 基线评估（一次），确保在同一分布下评估
            tmp_model = model_init(self.config).to(self.device)
            tmp_model = set_model_weights(tmp_model, self.shared_init_weights, self.device)
            b_loss0, b_acc0, b_bsr0 = self._evaluate_performance(tmp_model)
            print(f"[Hybrid Baseline Preview] Test Loss: {b_loss0:.4f}, Accuracy: {b_acc0:.2f}%, Backdoor SR: {b_bsr0:.2f}%")

        for round_idx in rounds_to_process:
            print(f"\n--- Hybrid Round {round_idx} [{self.hybrid_branch}] ---")

            # ========== 客户端侧分支一步 ==========
            try:
                starting_global = current_branch_weights or self.branch.get_global_model_at_round(round_idx)
                remaining_client_models = [copy.deepcopy(starting_global) for _ in self.remaining_clients]
                new_client_models = self.branch.train_clients_one_step(
                    client_models=remaining_client_models,
                    global_model=starting_global,
                    train_loaders=self.train_loaders,
                    lr=lr,
                    epochs=local_epochs
                )
                sample_sizes = [len(loader.dataset) for loader in self.train_loaders]
                aggregated_weights = fedavg(new_client_models, sample_sizes)
                old_global_weights = self.branch.get_global_model_at_round(round_idx)

                # 从历史中取旧客户端（若存在）
                old_client_models = [
                    self.client_updates_history[round_idx][cid]['weights']
                    for cid in self.remaining_clients
                    if round_idx in self.client_updates_history and cid in self.client_updates_history[round_idx]
                ]
                # FedEraser 校准
                calibrated_weights = self.branch.unlearning_step_once(
                    old_client_models, new_client_models, old_global_weights, aggregated_weights
                ) if old_client_models else aggregated_weights
                current_branch_weights = calibrated_weights

            except Exception as e:
                print(f"Error in FedEraser path at round {round_idx}: {e}")
                continue

            # 将分支权重加载到临时模型以便评估
            current_branch_model = model_init(self.config).to(self.device)
            current_branch_model = set_model_weights(current_branch_model, current_branch_weights, self.device)

            # (已修改) 评估客户端分支（在 85% 测试集上）
            print("[Client-Branch Evaluation before aggregation]")
            b_loss, b_acc, b_bsr = self._evaluate_performance(current_branch_model)
            print(f"ClientBranch -> Test Loss: {b_loss:.4f}, Accuracy: {b_acc:.2f}%, Backdoor Recall: {b_bsr:.2f}%")
            # 记录 Fed 分支（聚合前）指标
            self.history['fed_round'].append(round_idx)
            self.history['fed_loss'].append(b_loss)
            self.history['fed_accuracy'].append(b_acc)
            self.history['fed_backdoor_success_rate'].append(b_bsr)

            # ========== KD 分支 ==========
            if self.kd_epochs_per_round > 0:
                # 使用 KD 数据集（来自 unlearn_clients 的原始分片）
                print(f"[KD] Using {len(self.kd_loader.dataset)} samples for knowledge distillation (from clean unlearn clients)")

                self.student_model.train()
                optimizer = optim.SGD(self.student_model.parameters(), lr=lr, momentum=0.9)
                for _ in range(self.kd_epochs_per_round):
                    for data, target in self.kd_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        optimizer.zero_grad()
                        student_out = self.student_model(data)
                        with torch.no_grad():
                            teacher_out = self.teacher_model(data)
                        soft_loss = F.kl_div(
                            F.log_softmax(student_out / kd_temperature, dim=1),
                            F.softmax(teacher_out / kd_temperature, dim=1),
                            reduction='batchmean'
                        ) * (kd_temperature ** 2)
                        hard_loss = F.cross_entropy(student_out, target)
                        loss = kd_alpha * soft_loss + (1 - kd_alpha) * hard_loss
                        loss.backward()
                        optimizer.step()

                # (已修改) 评估 KD（在 85% 测试集上）
                print("[KD Evaluation before aggregation]")
                kd_test_loss, kd_test_acc, kd_backdoor_sr = self._evaluate_performance(self.student_model)
                print(f"KD -> Test Loss: {kd_test_loss:.4f}, Accuracy: {kd_test_acc:.2f}%, Backdoor Recall: {kd_backdoor_sr:.2f}%")
                # 记录 KD 分支（聚合前）指标
                self.history['kd_round'].append(round_idx)
                self.history['kd_loss'].append(kd_test_loss)
                self.history['kd_accuracy'].append(kd_test_acc)
                self.history['kd_backdoor_success_rate'].append(kd_backdoor_sr)

            # ========== 动态/按准确率 Alpha 聚合 ==========
            if self.kd_epochs_per_round > 0 and self.aggregation_interval > 0 and (round_idx + 1) % self.aggregation_interval == 0:
                current_alpha = self.aggregation_alpha  # 默认
                
                # 只要需要基于准确率的信息（动态 or 仅准确率聚合），就计算一次验证集准确率
                fed_acc_val = None
                kd_acc_val = None
                if self.use_dynamic_alpha or self.accuracy_only_agg:
                    print("[Validation] Measuring Fed/KD accuracy on 5% val set for aggregation...")
                    _, fed_acc_val, _ = self._evaluate_performance(current_branch_model, loader=self.val_loader)
                    _, kd_acc_val, _ = self._evaluate_performance(self.student_model, loader=self.val_loader)
                
                if self.accuracy_only_agg:
                    # 仅用“老”的按准确率聚合策略（比例分配）
                    denom = float(fed_acc_val) + float(kd_acc_val) + 1e-9
                    current_alpha = float(fed_acc_val) / denom
                    print(f"[Accuracy-Only Aggregation][ratio] fed_acc={fed_acc_val:.2f}%, kd_acc={kd_acc_val:.2f}% -> alpha={current_alpha:.4f}")
                elif self.use_dynamic_alpha:
                    print("[Dynamic Alpha] Calculating optimal alpha using 5% validation set (accuracy-based)...")
                    current_alpha = self._compute_dynamic_alpha(round_idx, fed_acc_val, kd_acc_val)
                
                fed_weights = current_branch_weights
                kd_weights = get_model_weights(self.student_model)
                
                # (已修改) 使用 current_alpha
                combined_weights = self._aggregate_weights(fed_weights, kd_weights, current_alpha)
                
                # 同步合并后的权重到两侧分支
                current_branch_weights = combined_weights
                current_branch_model = set_model_weights(current_branch_model, combined_weights, self.device)
                self.student_model = set_model_weights(self.student_model, combined_weights, self.device)
                
                # 聚合后重估 BN 统计：仅在第一次聚合且开启时执行一次
                if self.reestimate_bn and not self._bn_reestimated_once:
                    prefer_loader = self.train_loaders[0] if self.train_loaders else self.val_loader
                    self._reestimate_bn_stats(current_branch_model, prefer_loader, max_batches=50)
                    self._reestimate_bn_stats(self.student_model, prefer_loader, max_batches=50)
                    self._bn_reestimated_once = True
                
                print(f"[Aggregation] Combined Client-Branch and KD models at round {round_idx} with alpha={current_alpha:.2f}")
 
                # (已修改) 最终评估在 85% 测试集上进行
                test_loss, test_acc, backdoor_sr = self._evaluate_performance(current_branch_model)
                print(f"[Aggregation Evaluation] Round {round_idx}: Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.2f}%, Backdoor Recall = {backdoor_sr:.2f}%")
 
                self.history['round'].append(round_idx)
                self.history['test_loss'].append(test_loss)
                self.history['test_accuracy'].append(test_acc)
                self.history['backdoor_success_rate'].append(backdoor_sr)
                # 同步保存到 agg_* 便于导出
                self.history['agg_round'].append(round_idx)
                self.history['agg_loss'].append(test_loss)
                self.history['agg_accuracy'].append(test_acc)
                self.history['agg_backdoor_success_rate'].append(backdoor_sr)
 
        # 可选：导出 Excel
        if output_excel_path:
            try:
                import pandas as pd
                output_excel_path = resolve_output_path(output_excel_path)
                os.makedirs(os.path.dirname(output_excel_path) or ".", exist_ok=True)
                with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
                    # Fed 分支（聚合前）
                    df_fed = pd.DataFrame({
                        "Round": self.history['fed_round'],
                        "Test Loss": self.history['fed_loss'],
                        "Test Accuracy (%)": self.history['fed_accuracy'],
                        "Backdoor Success Rate (%)": self.history['fed_backdoor_success_rate'],
                    })
                    df_fed.to_excel(writer, index=False, sheet_name="Hybrid_FedBranch")
                    # KD 分支（聚合前）
                    df_kd = pd.DataFrame({
                        "Round": self.history['kd_round'],
                        "Test Loss": self.history['kd_loss'],
                        "Test Accuracy (%)": self.history['kd_accuracy'],
                        "Backdoor Success Rate (%)": self.history['kd_backdoor_success_rate'],
                    })
                    df_kd.to_excel(writer, index=False, sheet_name="Hybrid_KD")
                    # 聚合后
                    df_agg = pd.DataFrame({
                        "Round": self.history['agg_round'],
                        "Test Loss": self.history['agg_loss'],
                        "Test Accuracy (%)": self.history['agg_accuracy'],
                        "Backdoor Success Rate (%)": self.history['agg_backdoor_success_rate'],
                    })
                    df_agg.to_excel(writer, index=False, sheet_name="Hybrid_Aggregation")
                    # 配置快照
                    try:
                        df_cfg = pd.DataFrame({
                            "Key": list(self.config.keys()),
                            "Value": [self.config[k] for k in self.config.keys()]
                        })
                        df_cfg.to_excel(writer, index=False, sheet_name="Config")
                    except Exception:
                        pass
                print(f"[HybridEraser] Results saved to {output_excel_path}")
            except Exception as e:
                print(f"[HybridEraser] Failed to save Excel: {e}")

        return current_branch_model, self.history

    def _compute_dynamic_alpha(self, round_idx: int, fed_acc_val: float, kd_acc_val: float) -> float:
        """
        Compute dynamic alpha directly using validation accuracy (higher is better).
        Goals:
          - Combine current accuracy and historical trend (EMA and slope)
          - Early sensitive, later blunt (phase decay)
          - Smoothing to avoid jitter
        Returns alpha in [alpha_min, alpha_max] for aggregation: fed_alpha=alpha, kd_alpha=1-alpha.
        """
        # --- Hyperparameters (with sensible defaults) ---
        beta = float(self.config.get('dynamic_alpha_beta', 0.7))            # EMA factor for accuracy
        window = int(self.config.get('dynamic_alpha_window', 5))            # slope window length
        tau = float(self.config.get('dynamic_alpha_tau', max(8, self._dyn_total_rounds // 2 or 8)))  # phase half-life-ish
        base_sens = float(self.config.get('dynamic_alpha_base_sensitivity', 0.6))  # stronger sensitivity
        trend_weight = float(self.config.get('dynamic_alpha_trend_weight', 0.25))  # stronger trend impact
        trend_scale = float(self.config.get('dynamic_alpha_trend_scale', 4.0))
        smooth = float(self.config.get('dynamic_alpha_smooth', 0.8))        # alpha smoothing factor
        alpha_min = float(self.config.get('dynamic_alpha_min', 0.05))       # wider bounds
        alpha_max = float(self.config.get('dynamic_alpha_max', 0.95))
        max_step = float(self.config.get('dynamic_alpha_max_step', 0.20))   # allow faster moves
        gain = float(self.config.get('dynamic_alpha_gain', 1.5))            # amplify deviation from 0.5
        bias = float(self.config.get('dynamic_alpha_bias', 0.0))            # optional directional bias (Fed positive)

        # --- Update EMAs ---
        self._dyn_fed_ema = float(fed_acc_val) if self._dyn_fed_ema is None else (beta * self._dyn_fed_ema + (1.0 - beta) * float(fed_acc_val))
        self._dyn_kd_ema  = float(kd_acc_val)  if self._dyn_kd_ema  is None else (beta * self._dyn_kd_ema  + (1.0 - beta) * float(kd_acc_val))

        # --- Update raw history (use EMA values for slope to reduce noise) ---
        self._dyn_fed_hist.append(float(self._dyn_fed_ema))
        self._dyn_kd_hist.append(float(self._dyn_kd_ema))
        if len(self._dyn_fed_hist) > 100:
            self._dyn_fed_hist = self._dyn_fed_hist[-100:]
            self._dyn_kd_hist = self._dyn_kd_hist[-100:]

        def slope_last(xs: List[float], k: int) -> float:
            if len(xs) < 2:
                return 0.0
            n = min(k, len(xs) - 1)
            return (xs[-1] - xs[-(n+1)]) / float(n)

        fed_slope = slope_last(self._dyn_fed_hist, window)
        kd_slope  = slope_last(self._dyn_kd_hist, window)

        # --- Phase factor: early sensitive, later blunt ---
        # Decays with round index to reduce sensitivity over time
        phase = float(np.exp(- max(0, round_idx) / max(1e-6, tau)))

        # --- Current relative accuracy difference (normalized) ---
        avg_acc = 0.5 * (float(fed_acc_val) + float(kd_acc_val)) + 1e-9
        # diff > 0 means Fed better -> alpha should go up (favor Fed)
        diff = (float(fed_acc_val) - float(kd_acc_val)) / avg_acc

        # --- Trend adjustment (accuracy): if Fed improving faster than KD, further favor Fed ---
        trend_diff = fed_slope - kd_slope
        trend_term = trend_weight * np.tanh(trend_scale * trend_diff)

        # --- Sensitivity scales with phase ---
        sensitivity = base_sens * phase

        # --- Raw target around 0.5, shift by diff (Fed better -> alpha up) and trend (Fed improving faster -> alpha up) ---
        alpha_target = 0.5 + bias + gain * (sensitivity * diff + phase * trend_term)

        # --- Clamp to bounds ---
        alpha_target = float(np.clip(alpha_target, alpha_min, alpha_max))

        # --- Smooth and clamp per-step change ---
        alpha_smoothed = smooth * float(self._dyn_prev_alpha) + (1.0 - smooth) * alpha_target
        # Limit change per aggregation to avoid jumps
        low = float(self._dyn_prev_alpha) - max_step
        high = float(self._dyn_prev_alpha) + max_step
        alpha_new = float(np.clip(alpha_smoothed, low, high))
        alpha_new = float(np.clip(alpha_new, alpha_min, alpha_max))

        print(f"[Dynamic Alpha] fed_acc={fed_acc_val:.2f}%, kd_acc={kd_acc_val:.2f}%, diff={diff:.4f}, "
              f"fed_slope={fed_slope:.5f}, kd_slope={kd_slope:.5f}, phase={phase:.3f}, "
              f"gain={gain:.2f}, bias={bias:.2f}")
        print(f"[Dynamic Alpha] target={alpha_target:.4f} (min={alpha_min:.2f}, max={alpha_max:.2f}), "
              f"smoothed={alpha_smoothed:.4f}, step<=±{max_step:.2f} -> alpha={alpha_new:.4f}")

        self._dyn_prev_alpha = alpha_new
        return alpha_new


# --- (已修改) 主函数 (run_hybrid_unlearning) ---
def run_hybrid_unlearning(config, train_loaders, test_loader, val_loader, kd_loader, global_model):
    """ 接受外部准备好的 test_loader, val_loader, kd_loader """
    
    # 数据已经在主notebook中分割好了，直接使用传入的loader
    print(f"[Data Split] Using provided data: test={len(test_loader.dataset)}, kd={len(kd_loader.dataset)}, val={len(val_loader.dataset)}")

    # 2. 初始化 HybridEraser。
    hybrid = HybridEraser(
        model_path=config["model_path"],
        history_path=config["history_path"],
        config=config,
        device=next(global_model.parameters()).device,
        kd_epochs_per_round=config.get("hybrid_kd_epochs_per_round", 2),
        aggregation_interval=config.get("aggregation_interval", 5),
        aggregation_alpha=config.get("aggregation_alpha", 0.5),
        use_dynamic_alpha=config.get("use_dynamic_alpha", True),
        save_dir=os.path.join(config.get("save_dir", "."), "hybrid")
    )

    # 3. (已修改) 运行，传入已经分割好的 loader。
    erased_model, history = hybrid.run(
        clients_to_remove=config["unlearn_clients"],
        train_loaders=train_loaders,
        test_loader=test_loader,  # 传入测试集
        val_loader=val_loader,    # 传入验证集
        kd_loader=kd_loader,      # 传入 KD 集
        max_rounds=config.get("federaser_max_rounds"),
        batch_size=config.get("batch_size"),
        lr=config.get("lr"),
        local_epochs=config.get("federaser_epochs"),
        kd_temperature=config.get("kd_temperature"),
        kd_alpha=config.get("kd_alpha")
    )
    return erased_model, history
