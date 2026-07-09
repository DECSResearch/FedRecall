import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
from path_utils import resolve_output_path


def _safe_list(x: Optional[List]) -> List:
    return list(x) if x is not None else []


def _df_from_series(series: Dict[str, List], rename: Dict[str, str]) -> pd.DataFrame:
    data = {}
    for k, v in series.items():
        if isinstance(v, list):
            data[rename.get(k, k)] = v
    # Ensure a Round column when lengths match or inferable
    if "Round" not in data:
        # try to infer length from first series
        for v in data.values():
            data["Round"] = list(range(len(v)))
            break
    return pd.DataFrame(data)


def to_df_federaser(history: Dict) -> pd.DataFrame:
    """
    Convert FedEraser history to a DataFrame.
    Expected keys in history:
      - round, test_loss, test_accuracy, backdoor_success_rate, round_time (optional), total_time (footer)
    """
    rename = {
        "round": "Round",
        "test_loss": "Test Loss",
        "test_accuracy": "Test Accuracy (%)",
        "backdoor_success_rate": "Backdoor Success Rate (%)",
        "round_time": "Round Time (s)",
    }
    df = _df_from_series(history or {}, rename)
    return df


def to_df_kd(history: Dict) -> pd.DataFrame:
    """
    Convert KD history to a DataFrame.
    Expected keys:
      - epoch, train_loss, test_loss, test_accuracy, backdoor_success_rate, epoch_time (optional)
    """
    rename = {
        "epoch": "Epoch",
        "train_loss": "Train Loss",
        "test_loss": "Test Loss",
        "test_accuracy": "Test Accuracy (%)",
        "backdoor_success_rate": "Backdoor Success Rate (%)",
        "epoch_time": "Epoch Time (s)",
    }
    # KD uses Epoch instead of Round
    data = {}
    hist = history or {}
    for k, v in hist.items():
        if isinstance(v, list):
            data[rename.get(k, k)] = v
    if "Epoch" not in data:
        for v in data.values():
            data["Epoch"] = list(range(1, len(v) + 1))
            break
    return pd.DataFrame(data)


def to_dfs_hybrid(history: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build three DataFrames for HybridEraser:
      - Fed branch before aggregation (Hybrid_FedBranch)
      - KD branch before aggregation (Hybrid_KD)
      - Aggregation after merge (Hybrid_Aggregation)
    """
    hist = history or {}
    df_fed = pd.DataFrame({
        "Round": _safe_list(hist.get("fed_round")),
        "Test Loss": _safe_list(hist.get("fed_loss")),
        "Test Accuracy (%)": _safe_list(hist.get("fed_accuracy")),
        "Backdoor Success Rate (%)": _safe_list(hist.get("fed_backdoor_success_rate")),
    })
    df_kd = pd.DataFrame({
        "Round": _safe_list(hist.get("kd_round")),
        "Test Loss": _safe_list(hist.get("kd_loss")),
        "Test Accuracy (%)": _safe_list(hist.get("kd_accuracy")),
        "Backdoor Success Rate (%)": _safe_list(hist.get("kd_backdoor_success_rate")),
    })
    # Aggregation results
    # Fallback to 'round'/'test_*' if 'agg_*' not present
    agg_round = hist.get("agg_round", hist.get("round"))
    agg_loss = hist.get("agg_loss", hist.get("test_loss"))
    agg_acc = hist.get("agg_accuracy", hist.get("test_accuracy"))
    agg_bsr = hist.get("agg_backdoor_success_rate", hist.get("backdoor_success_rate"))
    df_agg = pd.DataFrame({
        "Round": _safe_list(agg_round),
        "Test Loss": _safe_list(agg_loss),
        "Test Accuracy (%)": _safe_list(agg_acc),
        "Backdoor Success Rate (%)": _safe_list(agg_bsr),
    })
    return df_fed, df_kd, df_agg


def write_results_excel(
    output_path: str,
    *,
    config: Optional[Dict] = None,
    train_history: Optional[Dict] = None,
    retrain_history: Optional[Dict] = None,
    fed_history: Optional[Dict] = None,
    kd_history: Optional[Dict] = None,
    hybrid_history: Optional[Dict] = None
) -> None:
    """
    Write all method results into a single Excel file with multiple sheets.
    Sheets (written if provided):
      - Train
      - Retrain
      - FedEraser
      - KD
      - Hybrid_FedBranch
      - Hybrid_KD
      - Hybrid_Aggregation
      - Config
    """
    output_path = resolve_output_path(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Train (expect dict with "accuracy", "backdoor_success_rate" or similarly named lists)
        if train_history:
            # Flexible: try several common keys
            df_train = pd.DataFrame({
                "Round": _safe_list(train_history.get("round")) or list(range(1, len(_safe_list(train_history.get("accuracy") or train_history.get("test_accuracy"))) + 1)),
                "Test Accuracy (%)": _safe_list(train_history.get("accuracy") or train_history.get("test_accuracy")),
                "Backdoor Success Rate (%)": _safe_list(train_history.get("backdoor_success_rate")),
            })
            df_train.to_excel(writer, index=False, sheet_name="Train")

        # Retrain
        if retrain_history:
            df_retrain = pd.DataFrame({
                "Round": _safe_list(retrain_history.get("round")) or list(range(1, len(_safe_list(retrain_history.get("accuracy") or retrain_history.get("test_accuracy"))) + 1)),
                "Test Accuracy (%)": _safe_list(retrain_history.get("accuracy") or retrain_history.get("test_accuracy")),
                "Backdoor Success Rate (%)": _safe_list(retrain_history.get("backdoor_success_rate")),
            })
            df_retrain.to_excel(writer, index=False, sheet_name="Retrain")

        # FedEraser
        if fed_history:
            df_fed = to_df_federaser(fed_history)
            df_fed.to_excel(writer, index=False, sheet_name="FedEraser")

        # KD
        if kd_history:
            df_kd = to_df_kd(kd_history)
            df_kd.to_excel(writer, index=False, sheet_name="KD")

        # Hybrid
        if hybrid_history:
            df_h_fed, df_h_kd, df_h_agg = to_dfs_hybrid(hybrid_history)
            df_h_fed.to_excel(writer, index=False, sheet_name="Hybrid_FedBranch")
            df_h_kd.to_excel(writer, index=False, sheet_name="Hybrid_KD")
            df_h_agg.to_excel(writer, index=False, sheet_name="Hybrid_Aggregation")

        # Config
        if config:
            df_cfg = pd.DataFrame({
                "Key": list(config.keys()),
                "Value": [config[k] for k in config.keys()]
            })
            df_cfg.to_excel(writer, index=False, sheet_name="Config")

    print(f"[Export] Results have been saved to {output_path}")


