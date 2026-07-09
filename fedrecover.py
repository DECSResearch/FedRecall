import os
import time
import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import model_init
from path_utils import resolve_output_path
from backdoor import evaluate_backdoor


class FedRecover:
    """
    FedRecover implementation adapted to this project's data/model pipeline.
    Works directly on the saved federated learning history (client weights per round + aggregated globals).
    """
    def __init__(self,
                 model_path: str,
                 history_path: str,
                 config: dict,
                 history_dir: str = './fl_history',
                 save_dir: str = './unlearned_models',
                 device: Optional[torch.device] = None):
        self.config = config or {}
        self.history_dir = history_dir
        self.save_dir = resolve_output_path(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Model structure (weights are not needed for initialization)
        self.global_model = model_init(self.config).to(self.device)
        if model_path and os.path.exists(model_path):
            # Not strictly required for recovery, but useful for evaluation/reference
            self.global_model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Load complete FL history
        import pickle
        with open(history_path, 'rb') as f:
            complete_data = pickle.load(f)

        self.client_updates_history = complete_data['client_updates_history']  # {round: {client_id: {'weights': dict, 'sample_size': int}}}
        self.aggregation_history = complete_data['aggregation_history']        # {round: global_weights_dict}
        self.total_rounds = complete_data['current_round']

        # Derive sorted client id list across all rounds
        client_ids = set()
        for r in self.client_updates_history:
            client_ids.update(self.client_updates_history[r].keys())
        self.client_ids = sorted(list(client_ids))

    # -------------------------- Helpers (weights/tensors/shapes) --------------------------
    @staticmethod
    def numpy_dict_to_torch(weights_np: Dict[str, np.ndarray],
                            device: torch.device,
                            dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in weights_np.items():
            # Convert scalars/ints to float tensors for math
            if isinstance(v, np.ndarray):
                tensor = torch.from_numpy(v)
            else:
                tensor = torch.tensor(v)
            # Cast to float for vector math
            if not torch.is_floating_point(tensor):
                tensor = tensor.float()
            out[k] = tensor.to(device)
        return out

    @staticmethod
    def torch_dict_to_numpy(weights_t: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        return {k: (v.detach().cpu().numpy().astype(np.float32)) for k, v in weights_t.items()}

    @staticmethod
    def model_difference(model1: Dict[str, torch.Tensor],
                         model2: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # model1 - model2 per key
        return {k: model1[k] - model2[k] for k in model1.keys()}

    @staticmethod
    def flatten(state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([p.view(-1) for p in state.values()])

    @staticmethod
    def unflatten(flat: torch.Tensor, template: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        idx = 0
        for k, v in template.items():
            n = v.numel()
            out[k] = flat[idx:idx + n].view_as(v)
            idx += n
        return out

    @staticmethod
    def average_model_updates(update_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not update_list:
            raise ValueError("No updates to average")
        avg = {}
        for k in update_list[0].keys():
            avg[k] = sum([upd[k] for upd in update_list]) / float(len(update_list))
        return avg

    @staticmethod
    def lbfgs(buffer: List[Tuple[torch.Tensor, torch.Tensor]], v: torch.Tensor) -> torch.Tensor:
        """
        Lightweight L-BFGS Hessian-vector approximation used in original scripts.
        Buffer holds tuples of (delta_w_flat, delta_g_flat).
        """
        if len(buffer) == 0:
            return torch.zeros_like(v)

        delta_w = torch.stack([s for s, _ in buffer], dim=0)  # [m, d]
        delta_g = torch.stack([y for _, y in buffer], dim=0)  # [m, d]

        device = v.device
        delta_w = delta_w.to(device)
        delta_g = delta_g.to(device)
        v_orig = v.to(device)

        # sigma term
        w_prev = delta_w[-1]
        g_prev = delta_g[-1]
        denom = torch.dot(w_prev, w_prev)
        if denom == 0:
            sigma = torch.tensor(1.0, device=device)
        else:
            sigma = torch.dot(g_prev, w_prev) / denom

        # Approximate Hv
        Hv_approx = sigma * v_orig - torch.sum(delta_g * delta_w, dim=1).unsqueeze(1) * v_orig
        if Hv_approx.dim() == 2:
            Hv_approx = Hv_approx.mean(dim=0)
        return Hv_approx

    def _build_original_structures(self,
                                   rounds_to_use: List[int]) -> Tuple[Dict[int, Dict[str, torch.Tensor]],
                                                                      Dict[int, Dict[int, Dict[str, torch.Tensor]]]]:
        """
        Construct:
        - original_global_models[r] = aggregated global (after round r) as torch tensors
        - original_model_updates[r][client_id] = (global_before_round_r - client_weights_r) as torch tensors
        Skip round 0 updates if no "before" model is available.
        """
        original_global_models: Dict[int, Dict[str, torch.Tensor]] = {}
        original_model_updates: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}

        for r in rounds_to_use:
            if r in self.aggregation_history:
                original_global_models[r] = self.numpy_dict_to_torch(self.aggregation_history[r], self.device)

        for r in rounds_to_use:
            # Determine the "before round r" global weights
            if r - 1 in self.aggregation_history:
                before_np = self.aggregation_history[r - 1]
            else:
                # If there is no r-1 (i.e., r == 0), we cannot form updates reliably; skip
                continue

            before_t = self.numpy_dict_to_torch(before_np, self.device)

            if r in self.client_updates_history:
                round_updates: Dict[int, Dict[str, torch.Tensor]] = {}
                for cid, entry in self.client_updates_history[r].items():
                    client_w_np = entry['weights']
                    client_w_t = self.numpy_dict_to_torch(client_w_np, self.device)
                    # Update defined as (global_before - client_after_local)
                    upd = self.model_difference(before_t, client_w_t)
                    round_updates[cid] = upd
                original_model_updates[r] = round_updates

        return original_global_models, original_model_updates

    def _train_local_once(self,
                          model: nn.Module,
                          dataloader: torch.utils.data.DataLoader,
                          lr: float,
                          epochs: int = 1) -> nn.Module:
        model = model.to(self.device)
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=self.config.get("momentum", 0.9))
        for _ in range(max(1, int(epochs))):
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                out = model(data)
                loss = nn.functional.cross_entropy(out, target)
                loss.backward()
                optimizer.step()
        return model

    def _exact_model_update(self,
                            recovered_model: nn.Module,
                            clean_clients: List[int],
                            cid_to_loader: Dict[int, torch.utils.data.DataLoader],
                            lr: float,
                            epochs: int) -> nn.Module:
        model = recovered_model
        for cid in clean_clients:
            if cid not in cid_to_loader:
                continue
            model = self._train_local_once(model, cid_to_loader[cid], lr=lr, epochs=epochs)
        return model

    def run(self,
            clients_to_remove: List[int],
            train_loaders: List[torch.utils.data.DataLoader],
            test_loader: torch.utils.data.DataLoader,
            save_model_path: Optional[str] = None,
            unlearn_rounds: Optional[List[int]] = None,
            max_rounds: Optional[int] = None,
            lr: Optional[float] = None,
            local_epochs: Optional[int] = None,
            init_state_dict: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Execute FedRecover.
        Returns (recovered_weights_numpy, history_dict).
        """
        # Hyperparameters (with sensible defaults)
        recover_lr = lr if lr is not None else float(self.config.get("lr", 0.01))
        warmup_rounds = int(self.config.get("fedrecover_warmup_rounds", 1))
        correction_period = int(self.config.get("fedrecover_correction_period", 5))
        final_tuning_rounds = int(self.config.get("fedrecover_final_tuning_rounds", 1))
        buffer_size = int(self.config.get("fedrecover_buffer_size", 5))
        abnormal_threshold = float(self.config.get("fedrecover_abnorm_threshold", 40.0))
        local_epochs = int(local_epochs if local_epochs is not None else self.config.get("federaser_epochs", 1))

        # Rounds to use
        if unlearn_rounds is None:
            if max_rounds is not None and 0 < max_rounds < self.total_rounds:
                rounds_to_use = list(range(max_rounds))
            else:
                rounds_to_use = list(range(self.total_rounds))
        else:
            rounds_to_use = sorted([r for r in unlearn_rounds if 0 <= r < self.total_rounds])

        # Map client id -> loader (robust alignment)
        cid_to_loader: Dict[int, torch.utils.data.DataLoader] = {}
        # Common case: len(train_loaders) == len(self.client_ids) and same ordering
        if len(train_loaders) == len(self.client_ids):
            for cid, loader in zip(self.client_ids, train_loaders):
                cid_to_loader[cid] = loader
        else:
            for cid in self.client_ids:
                if 0 <= cid < len(train_loaders):
                    cid_to_loader[cid] = train_loaders[cid]

        # Clean clients (kept clients)
        remove_set = set(clients_to_remove or [])
        clean_clients = [cid for cid in self.client_ids if cid not in remove_set]
        if not clean_clients:
            raise ValueError("No clean clients left after removal list is applied.")

        print(f"[FedRecover] clean_clients={len(clean_clients)}, rounds={len(rounds_to_use)}, "
              f"warmup={warmup_rounds}, correction_period={correction_period}, final={final_tuning_rounds}, "
              f"lr={recover_lr}, local_epochs={local_epochs}")

        # Build original structures for FedRecover
        original_global_models, original_model_updates = self._build_original_structures(rounds_to_use)

        # Initialize recovered model (fresh)
        recovered_model = model_init(self.config).to(self.device)
        if init_state_dict is not None:
            recovered_model.load_state_dict(init_state_dict)
        else:
            recovered_model.apply(self._init_weights_like_torch)

        # Client buffers for L-BFGS (per clean client)
        client_buffers: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {cid: [] for cid in clean_clients}

        # History
        history = {
            'round': [],
            'test_loss': [],
            'test_accuracy': [],
            'backdoor_success_rate': []
        }

        # Baseline before any recovery
        base_loss, base_acc = self._evaluate_model(recovered_model, test_loader)
        base_bdsr = evaluate_backdoor(
            recovered_model,
            test_loader,
            self.device,
            trigger_pattern=self.config.get("backdoor_pattern"),
            trigger_size=self.config.get("backdoor_size"),
            target_label=self.config.get("backdoor_target")
        )
        print(f"[FedRecover][Baseline] Test Loss = {base_loss:.4f}, Acc = {base_acc:.2f}%, BDSR = {base_bdsr:.2f}%")
        history['round'].append(-1)
        history['test_loss'].append(base_loss)
        history['test_accuracy'].append(base_acc)
        history['backdoor_success_rate'].append(base_bdsr)

        # Warmup: exact local training on clean clients for a few rounds
        for w in range(warmup_rounds):
            recovered_model = self._exact_model_update(recovered_model, clean_clients, cid_to_loader,
                                                       lr=recover_lr, epochs=local_epochs)
            wl, wa = self._evaluate_model(recovered_model, test_loader)
            wb = evaluate_backdoor(
                recovered_model, test_loader, self.device,
                trigger_pattern=self.config.get("backdoor_pattern"),
                trigger_size=self.config.get("backdoor_size"),
                target_label=self.config.get("backdoor_target")
            )
            print(f"[FedRecover][Warmup {w+1}/{warmup_rounds}] Test Loss = {wl:.4f}, Acc = {wa:.2f}%, BDSR = {wb:.2f}%")
            history['round'].append(f"warmup-{w+1}")
            history['test_loss'].append(wl)
            history['test_accuracy'].append(wa)
            history['backdoor_success_rate'].append(wb)

        # Recovery rounds
        for step_idx, r in enumerate(rounds_to_use[:-final_tuning_rounds] if final_tuning_rounds > 0 else rounds_to_use):
            # Skip if we don't have updates for this round (e.g., r==0 without before model)
            if r not in original_model_updates:
                continue
            # Periodic correction: exact update and buffer refresh
            if (step_idx + 1) % correction_period == 0:
                recovered_model = self._exact_model_update(recovered_model, clean_clients, cid_to_loader,
                                                           lr=recover_lr, epochs=local_epochs)
                # Update client buffers using differences between recovered and original global
                if r in original_global_models:
                    with torch.no_grad():
                        rec_sd = {k: v.detach().clone() for k, v in recovered_model.state_dict().items()}
                        # Align to torch dict
                        rec_sd_t = {k: vv.to(self.device) for k, vv in rec_sd.items()}
                        global_diff = self.model_difference(rec_sd_t, original_global_models[r])
                        global_diff_flat = self.flatten(global_diff)
                    # Add pair (delta_w, delta_g) per client to buffer (here we approximate both with diffs)
                    for cid in clean_clients:
                        client_update = original_model_updates[r][cid]
                        client_update_flat = self.flatten(client_update).detach()
                        client_buffers[cid].append((global_diff_flat.detach(), client_update_flat))
                        if len(client_buffers[cid]) > buffer_size:
                            client_buffers[cid] = client_buffers[cid][-buffer_size:]
                # Evaluate and log after correction step
                rl, ra = self._evaluate_model(recovered_model, test_loader)
                rb = evaluate_backdoor(
                    recovered_model, test_loader, self.device,
                    trigger_pattern=self.config.get("backdoor_pattern"),
                    trigger_size=self.config.get("backdoor_size"),
                    target_label=self.config.get("backdoor_target")
                )
                print(f"[FedRecover][Round {step_idx+1}/{len(rounds_to_use)}] Test Loss = {rl:.4f}, Acc = {ra:.2f}%, BDSR = {rb:.2f}%")
                history['round'].append(int(r))
                history['test_loss'].append(rl)
                history['test_accuracy'].append(ra)
                history['backdoor_success_rate'].append(rb)
                continue

            # Otherwise: estimate updates via L-BFGS and aggregate
            estimated_updates: List[Dict[str, torch.Tensor]] = []
            for cid in clean_clients:
                v = self.flatten(original_model_updates[r][cid]).detach()
                Hv = self.lbfgs(client_buffers[cid], v)
                g_est = Hv
                g_orig = v
                diff_norm = torch.norm(g_est - g_orig)
                if diff_norm > abnormal_threshold:
                    # Fallback: train locally and use true update direction
                    local_model = copy.deepcopy(recovered_model).to(self.device)
                    local_model = self._train_local_once(local_model,
                                                         cid_to_loader.get(cid),
                                                         lr=recover_lr,
                                                         epochs=local_epochs)
                    with torch.no_grad():
                        rec_sd = {k: v.detach().clone() for k, v in recovered_model.state_dict().items()}
                        new_sd = {k: v.detach().clone() for k, v in local_model.state_dict().items()}
                        new_update = self.model_difference(rec_sd, new_sd)  # recovered - local
                    estimated_updates.append(new_update)
                else:
                    # Unflatten into layer-wise dict using this client's update shape
                    template = original_model_updates[r][cid]
                    est_update_dict = self.unflatten(g_est, template)
                    estimated_updates.append(est_update_dict)

            # Apply averaged update to recovered model
            with torch.no_grad():
                rec_sd = {k: v.detach().clone() for k, v in recovered_model.state_dict().items()}
                flat_model = self.flatten(rec_sd)
                flat_updates = [self.flatten(u) for u in estimated_updates]
                avg_update_flat = sum(flat_updates) / float(len(flat_updates))
                flat_model -= recover_lr * avg_update_flat
                new_sd = self.unflatten(flat_model, rec_sd)
                recovered_model.load_state_dict(new_sd)

            rl, ra = self._evaluate_model(recovered_model, test_loader)
            rb = evaluate_backdoor(
                recovered_model, test_loader, self.device,
                trigger_pattern=self.config.get("backdoor_pattern"),
                trigger_size=self.config.get("backdoor_size"),
                target_label=self.config.get("backdoor_target")
            )
            print(f"[FedRecover][Round {step_idx+1}/{len(rounds_to_use)}] Test Loss = {rl:.4f}, Acc = {ra:.2f}%, BDSR = {rb:.2f}%")
            history['round'].append(int(r))
            history['test_loss'].append(rl)
            history['test_accuracy'].append(ra)
            history['backdoor_success_rate'].append(rb)

        # Final tuning
        for f in range(final_tuning_rounds):
            recovered_model = self._exact_model_update(recovered_model, clean_clients, cid_to_loader,
                                                       lr=recover_lr, epochs=local_epochs)
            fl, fa = self._evaluate_model(recovered_model, test_loader)
            fb = evaluate_backdoor(
                recovered_model, test_loader, self.device,
                trigger_pattern=self.config.get("backdoor_pattern"),
                trigger_size=self.config.get("backdoor_size"),
                target_label=self.config.get("backdoor_target")
            )
            print(f"[FedRecover][Final {f+1}/{final_tuning_rounds}] Test Loss = {fl:.4f}, Acc = {fa:.2f}%, BDSR = {fb:.2f}%")
            history['round'].append(f"final-{f+1}")
            history['test_loss'].append(fl)
            history['test_accuracy'].append(fa)
            history['backdoor_success_rate'].append(fb)

        # Evaluate
        test_loss, test_accuracy = self._evaluate_model(recovered_model, test_loader)
        backdoor_sr = evaluate_backdoor(
            recovered_model,
            test_loader,
            self.device,
            trigger_pattern=self.config.get("backdoor_pattern"),
            trigger_size=self.config.get("backdoor_size"),
            target_label=self.config.get("backdoor_target")
        )
        history['round'].append(-1)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_accuracy)
        history['backdoor_success_rate'].append(backdoor_sr)

        # Save (optional)
        if save_model_path:
            save_model_path = resolve_output_path(save_model_path)
            torch.save(recovered_model.state_dict(), save_model_path)

        # Return weights as numpy dict for consistency
        recovered_weights_np = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in recovered_model.state_dict().items()}
        return recovered_weights_np, history

    def _evaluate_model(self,
                        model: nn.Module,
                        test_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                out = model(data)
                loss = nn.functional.cross_entropy(out, target, reduction='sum')
                total_loss += float(loss.item())
                pred = out.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        avg_loss = total_loss / max(1, total)
        acc = 100.0 * float(correct) / max(1, total)
        return avg_loss, acc

    @staticmethod
    def _init_weights_like_torch(m):
        # Simple, safe init for linear/conv layers; keep buffers as-is
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)


