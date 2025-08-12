# positional_encoding.py

import logging
import numpy as np
import torch
from typing import Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PositionalEncoding:
    """
    Positional encoding implementation based on the Transformer paper.

    Args:
        d_model (int): The dimensionality of the input data.
        max_len (int): The maximum sequence length.
        dropout (float): The dropout probability.
        device (torch.device): The device to use for computations.

    Attributes:
        pe (torch.Tensor): The positional encoding tensor.
    """

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1, device: torch.device = torch.device("cpu")):
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = dropout
        self.device = device

        # Create the positional encoding tensor
        self.pe = self._create_positional_encoding()

    def _create_positional_encoding(self) -> torch.Tensor:
        """
        Create the positional encoding tensor.

        Returns:
            torch.Tensor: The positional encoding tensor.
        """
        pe = torch.zeros(self.max_len, self.d_model, device=self.device)
        position = torch.arange(0, self.max_len, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=self.device) * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The input tensor with positional encoding applied.
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

class PositionalEncodingConfig:
    """
    Configuration class for the positional encoding.

    Attributes:
        d_model (int): The dimensionality of the input data.
        max_len (int): The maximum sequence length.
        dropout (float): The dropout probability.
        device (torch.device): The device to use for computations.
    """

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1, device: torch.device = torch.device("cpu")):
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = dropout
        self.device = device

class PositionalEncodingError(Exception):
    """
    Exception class for positional encoding errors.
    """

    pass

def create_positional_encoding(d_model: int, max_len: int, dropout: float = 0.1, device: torch.device = torch.device("cpu")) -> PositionalEncoding:
    """
    Create a positional encoding instance.

    Args:
        d_model (int): The dimensionality of the input data.
        max_len (int): The maximum sequence length.
        dropout (float): The dropout probability.
        device (torch.device): The device to use for computations.

    Returns:
        PositionalEncoding: The positional encoding instance.
    """
    try:
        positional_encoding = PositionalEncoding(d_model, max_len, dropout, device)
        return positional_encoding
    except Exception as e:
        raise PositionalEncodingError(f"Failed to create positional encoding: {str(e)}")

def get_positional_encoding_config(d_model: int, max_len: int, dropout: float = 0.1, device: torch.device = torch.device("cpu")) -> PositionalEncodingConfig:
    """
    Get the positional encoding configuration.

    Args:
        d_model (int): The dimensionality of the input data.
        max_len (int): The maximum sequence length.
        dropout (float): The dropout probability.
        device (torch.device): The device to use for computations.

    Returns:
        PositionalEncodingConfig: The positional encoding configuration.
    """
    try:
        config = PositionalEncodingConfig(d_model, max_len, dropout, device)
        return config
    except Exception as e:
        raise PositionalEncodingError(f"Failed to get positional encoding configuration: {str(e)}")

# Example usage
if __name__ == "__main__":
    d_model = 512
    max_len = 1024
    dropout = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    positional_encoding = create_positional_encoding(d_model, max_len, dropout, device)
    config = get_positional_encoding_config(d_model, max_len, dropout, device)

    logger.info(f"Positional encoding created: {positional_encoding}")
    logger.info(f"Positional encoding configuration: {config}")