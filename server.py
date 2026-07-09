import os
import numpy as np
import torch
import copy
import pickle
from typing import Dict, List, Tuple, Any
from path_utils import resolve_output_path


class Server:
    """
    Federated Learning Server that manages client updates and their history.
    Provides functionality to store and retrieve historical updates for future unlearning.
    """
    
    def __init__(self, model_structure: Dict, save_dir: str = './fl_history', map_to_mydrive: bool = True):
        """
        Initialize the FL Server
        
        Args:
            model_structure: Model parameter structure (empty dict with correct keys)
            save_dir: Directory to save history files
        """
        self.client_updates_history = {}  # Store client updates for all rounds
        self.aggregation_history = {}     # Store aggregated weight results for all rounds
        self._map_to_mydrive = map_to_mydrive
        self.save_dir = resolve_output_path(save_dir) if map_to_mydrive else save_dir
        self.current_round = 0
        self.model_structure = model_structure
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"Created directory for FL history: {self.save_dir}")
    
    def store_client_updates(self, client_id: int, 
                            client_weights: Dict[str, np.ndarray],
                            sample_size: int) -> None:
        """
        Store a client's update for the current round
        
        Args:
            client_id: Client identifier
            client_weights: Client model weights
            sample_size: Number of samples used by this client
        """
        if self.current_round not in self.client_updates_history:
            self.client_updates_history[self.current_round] = {}
        
        self.client_updates_history[self.current_round][client_id] = {
            'weights': copy.deepcopy(client_weights),
            'sample_size': sample_size
        }
    
    def store_aggregation_result(self, global_weights: Dict[str, np.ndarray]) -> None:
        """
        Store the aggregation result for the current round
        
        Args:
            global_weights: Aggregated global model weights
        """
        self.aggregation_history[self.current_round] = copy.deepcopy(global_weights)
    
    def save_history(self, filename_prefix: str = 'fl_history') -> str:
        """
        Save the current history to disk
        
        Args:
            filename_prefix: Prefix for the saved files
            
        Returns:
            Path to the saved file
        """
        history_path = os.path.join(self.save_dir, f"{filename_prefix}_r{self.current_round}.pkl")
        
        history_data = {
            'client_updates': self.client_updates_history[self.current_round],
            'aggregation_result': self.aggregation_history[self.current_round],
            'round': self.current_round
        }
        
        with open(history_path, 'wb') as f:
            pickle.dump(history_data, f)
        
        print(f"Saved round {self.current_round} history to {history_path}")
        return history_path
    
    def next_round(self) -> None:
        """
        Advance to the next round
        """
        self.current_round += 1
    
    def get_client_update(self, round_num: int, client_id: int) -> Dict:
        """
        Retrieve a specific client's update from history
        
        Args:
            round_num: Round number
            client_id: Client identifier
            
        Returns:
            Client update data or None if not found
        """
        if round_num in self.client_updates_history and client_id in self.client_updates_history[round_num]:
            return self.client_updates_history[round_num][client_id]
        return None
    
    def get_aggregation_result(self, round_num: int) -> Dict[str, np.ndarray]:
        """
        Retrieve aggregation result for a specific round
        
        Args:
            round_num: Round number
            
        Returns:
            Aggregation weights or None if not found
        """
        if round_num in self.aggregation_history:
            return self.aggregation_history[round_num]
        return None
    
    def load_history(self, filepath: str) -> Dict:
        """
        Load history from a saved file
        
        Args:
            filepath: Path to the history file
            
        Returns:
            Loaded history data
        """
        with open(filepath, 'rb') as f:
            history_data = pickle.load(f)
        
        round_num = history_data['round']
        self.client_updates_history[round_num] = history_data['client_updates']
        self.aggregation_history[round_num] = history_data['aggregation_result']
        
        return history_data
    
    def save_complete_history(self, filepath: str = None) -> str:
        """
        Save complete history (all rounds) to a file
        
        Args:
            filepath: Path to save the file (if None, generates default path)
            
        Returns:
            Path to the saved file
        """
        if filepath is None:
            filepath = os.path.join(self.save_dir, f"complete_history_r0-r{self.current_round-1}.pkl")
        # 仅在需要写回 MyDrive 时做路径解析；本地训练时保持本地路径
        if self._map_to_mydrive:
            filepath = resolve_output_path(filepath)
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        
        complete_data = {
            'client_updates_history': self.client_updates_history,
            'aggregation_history': self.aggregation_history,
            'current_round': self.current_round
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(complete_data, f)
        
        print(f"Saved complete history to {filepath}")
        return filepath
    
    def load_complete_history(self, filepath: str) -> None:
        """
        Load complete history from a file
        
        Args:
            filepath: Path to the history file
        """
        with open(filepath, 'rb') as f:
            complete_data = pickle.load(f)
        
        self.client_updates_history = complete_data['client_updates_history']
        self.aggregation_history = complete_data['aggregation_history']
        self.current_round = complete_data['current_round']
        
        print(f"Loaded complete history from {filepath}")