import os
import logging
from typing import Dict, List, Tuple, Union
import torch
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """
    Model and training configuration class.

    ...

    Attributes
    ----------
    model_type : str
        Type of the model, e.g. 'llama_maverick'.
    model_name : str
        Name of the specific model, e.g. 'llama_maverick_v1'.
    transformer_dim : int
        Dimension of the transformer embeddings.
    num_layers : int
        Number of transformer layers.
    hidden_dim : int
        Dimension of the feed-forward network in transformer layers.
    num_attention_heads : int
        Number of attention heads in transformer layers.
    dropout : float
        Dropout probability for transformer layers.
    max_sequence_length : int
        Maximum allowed sequence length for input data.
    vocab_size : int
        Size of the input vocabulary.
    eagle_layers : int
        Number of EAGLE layers for speculative decoding.
    eagle_hidden_dim : int
        Dimension of the feed-forward network in EAGLE layers.
    spec_decode_probability : float
        Probability of activating speculative decoding during inference.
    tree_attention_depth : int
        Depth of the tree for tree-based attention mechanism.
    optimizer : str
        Optimizer to use for training, e.g. 'adam' or 'adagrad'.
    learning_rate : float
        Learning rate for the optimizer.
    weight_decay : float
        Weight decay (L2 regularization) coefficient.
    batch_size : int
        Batch size for training and validation.
    num_epochs : int
        Number of epochs to train for.
    early_stopping_patience : int
        Number of epochs to wait before early stopping if validation loss doesn't improve.
    checkpoint_dir : str
        Directory to save model checkpoints.
    log_dir : str
        Directory to save training logs.
    device : torch.device
        Device to use for training (cpu or cuda).

    Methods
    -------
    from_dict(config_dict):
        Creates a Config object from a dictionary.
    to_dict():
        Returns the Config object as a dictionary.
    load_from_file(file_path):
        Loads configuration from a JSON or YAML file.
    save_to_file(file_path):
        Saves the configuration to a JSON or YAML file.
    validate():
        Validates the configuration values.

    """

    def __init__(self,
                 model_type: str = 'llama_maverick',
                 model_name: str = 'llama_maverick_v1',
                 transformer_dim: int = 512,
                 num_layers: int = 6,
                 hidden_dim: int = 2048,
                 num_attention_heads: int = 8,
                 dropout: float = 0.1,
                 max_sequence_length: int = 512,
                 vocab_size: int = 30000,
                 eagle_layers: int = 3,
                 eagle_hidden_dim: int = 1024,
                 spec_decode_probability: float = 0.2,
                 tree_attention_depth: int = 3,
                 optimizer: str = 'adam',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 batch_size: int = 64,
                 num_epochs: int = 20,
                 early_stopping_patience: int = 5,
                 checkpoint_dir: str = 'checkpoints',
                 log_dir: str = 'logs',
                 device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.model_type = model_type
        self.model_name = model_name
        self.transformer_dim = transformer_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.eagle_layers = eagle_layers
        self.eagle_hidden_dim = eagle_hidden_dim
        self.spec_decode_probability = spec_decode_probability
        self.tree_attention_depth = tree_attention_depth
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.device = torch.device(device)

        # Create directories if they don't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'Config':
        """
        Creates a Config object from a dictionary.

        Parameters
        ----------
        config_dict : Dict
            Dictionary containing configuration values.

        Returns
        -------
        Config
            Config object initialized with the provided values.

        """
        return cls(**config_dict)

    def to_dict(self) -> Dict:
        """
        Returns the Config object as a dictionary.

        Returns
        -------
        Dict
            Dictionary containing configuration values.

        """
        return {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'transformer_dim': self.transformer_dim,
            'num_layers': self.num_layers,
            'hidden_dim': self.hidden_dim,
            'num_attention_heads': self.num_attention_heads,
            'dropout': self.dropout,
            'max_sequence_length': self.max_sequence_length,
            'vocab_size': self.vocab_size,
            'eagle_layers': self.eagle_layers,
            'eagle_hidden_dim': self.eagle_hidden_dim,
            'spec_decode_probability': self.spec_decode_probability,
            'tree_attention_depth': self.tree_attention_depth,
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
            'device': str(self.device)
        }

    @classmethod
    def load_from_file(cls, file_path: str) -> 'Config':
        """
        Loads configuration from a JSON or YAML file.

        Parameters
        ----------
        file_path : str
            Path to the configuration file.

        Returns
        -------
        Config
            Config object initialized with the values from the file.

        """
        try:
            with open(file_path, 'r') as file:
                config_dict = pd.read_json(file, orient='records').to_dict(orient='records')[0]
            return cls.from_dict(config_dict)
        except FileNotFoundError:
            logger.error(f"Configuration file not found at path: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration from file: {e}")
            raise

    def save_to_file(self, file_path: str):
        """
        Saves the configuration to a JSON or YAML file.

        Parameters
        ----------
        file_path : str
            Path to save the configuration file.

        """
        try:
            pd.DataFrame(self.to_dict(), index=[0]).to_json(file_path, orient='records', lines=True)
            logger.info(f"Configuration saved to file: {file_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to file: {e}")
            raise

    def validate(self):
        """
        Validates the configuration values.

        Raises
        ------
        ValueError
            If any configuration value is invalid.

        """
        if self.model_type not in ['llama_maverick', 'llama_explorer']:
            raise ValueError(f"Invalid model_type: {self.model_type}. Expected 'llama_maverick' or 'llama_explorer'.")

        if self.optimizer not in ['adam', 'adagrad']:
            raise ValueError(f"Invalid optimizer: {self.optimizer}. Expected 'adam' or 'adagrad'.")

        if not isinstance(self.learning_rate, float) or self.learning_rate <= 0:
            raise ValueError("Invalid learning_rate. Expected a positive float value.")

        if not isinstance(self.weight_decay, float) or self.weight_decay < 0:
            raise ValueError("Invalid weight_decay. Expected a non-negative float value.")

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("Invalid batch_size. Expected a positive integer value.")

        if not isinstance(self.num_epochs, int) or self.num_epochs <= 0:
            raise ValueError("Invalid num_epochs. Expected a positive integer value.")

        if not isinstance(self.early_stopping_patience, int) or self.early_stopping_patience < 0:
            raise ValueError("Invalid early_stopping_patience. Expected a non-negative integer value.")

# Example usage
if __name__ == '__main__':
    config = Config()
    config.save_to_file('config.json')

    loaded_config = Config.load_from_file('config.json')
    print(loaded_config.to_dict())