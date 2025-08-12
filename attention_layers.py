import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class AttentionLayer:
    """
    Base class for attention layers.
    """

    def __init__(self, hidden_size: int, num_attention_heads: int, attention_head_size: int, all_head_size: int):
        """
        Initializes the attention layer.

        Parameters:
        - hidden_size (int): The hidden size of the transformer model.
        - num_attention_heads (int): Number of attention heads.
        - attention_head_size (int): Size of each attention head.
        - all_head_size (int): Size of the total attention block.
        """
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.all_head_size = all_head_size

        self.query = torch.empty(size=(self.num_attention_heads, self.attention_head_size))
        self.key = torch.empty(size=(self.num_attention_heads, self.attention_head_size))
        self.value = torch.empty(size=(self.num_attention_heads, self.attention_head_size))

        self.dropout = torch.nn.Dropout(p=0.1)  # Dropout layer for regularization

    def compute_attention_score(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Computes the attention score between the query and key tensors.

        Parameters:
        - query (torch.Tensor): Tensor containing the query sequence.
        - key (torch.Tensor): Tensor containing the key sequence.

        Returns:
        - torch.Tensor: Attention score tensor.
        """
        # Reshape query and key tensors to match the attention head layout
        query = query.view(query.size(0), self.num_attention_heads, self.attention_head_size)
        key = key.view(key.size(0), self.num_attention_heads, self.attention_head_size)

        # Transpose to prepare for batch matrix multiplication
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)

        # Compute attention scores using batched matrix multiplication
        attention_scores = torch.bmm(query, key.transpose(1, 2))

        return attention_scores

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the attention layer.

        Parameters:
        - query (torch.Tensor): Tensor containing the query sequence.
        - key (torch.Tensor): Tensor containing the key sequence.
        - value (torch.Tensor): Tensor containing the values to be attended to.

        Returns:
        - torch.Tensor: Output tensor after applying attention.
        """
        # Compute attention scores
        attention_scores = self.compute_attention_score(query, key)

        # Apply dropout for regularization
        attention_scores = self.dropout(attention_scores)

        # Softmax to obtain attention probabilities
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # Compute context vector by combining values and attention probabilities
        context_vector = torch.bmm(attention_probs, value)

        # Transpose and reshape back to original layout
        context_vector = context_vector.transpose(0, 1).contiguous()
        context_vector = context_vector.view(context_vector.size(0), -1)

        return context_vector

class TreeAttentionLayer(AttentionLayer):
    """
    Attention layer that incorporates tree-based structures.
    """

    def __init__(self, hidden_size: int, num_attention_heads: int, attention_head_size: int, all_head_size: int, tree: Dict[int, List[int]]):
        """
        Initializes the tree attention layer.

        Parameters:
        - hidden_size (int): The hidden size of the transformer model.
        - num_attention_heads (int): Number of attention heads.
        - attention_head_size (int): Size of each attention head.
        - all_head_size (int): Size of the total attention block.
        - tree (Dict[int, List[int]]): Dictionary representing the tree structure. Keys are parent nodes, values are lists of child nodes.
        """
        super().__init__(hidden_size, num_attention_heads, attention_head_size, all_head_size)
        self.tree = tree

    def build_tree_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Builds a tree representation of the input tensor based on the provided tree structure.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, seq_length, hidden_size).

        Returns:
        - torch.Tensor: Tree representation of the input tensor.
        """
        batch_size, seq_length, _ = x.size()
        tree_representation = torch.empty(batch_size, seq_length, self.hidden_size)

        for i in range(batch_size):
            for parent, children in self.tree.items():
                parent_representation = x[i, parent, :]
                child_representations = [x[i, child, :] for child in children]
                tree_representation[i, parent, :] = torch.sum(torch.stack(child_representations), dim=0) + parent_representation

        return tree_representation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the tree attention layer.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, seq_length, hidden_size).

        Returns:
        - torch.Tensor: Output tensor after applying tree attention.
        """
        # Build tree representation of the input tensor
        tree_representation = self.build_tree_representation(x)

        # Perform self-attention on the tree representation
        query = tree_representation
        key = tree_representation
        value = tree_representation
        output = super().forward(query, key, value)

        return output

class MultiRoundSpeculativeDecodingLayer(AttentionLayer):
    """
    Attention layer that performs multi-round speculative decoding.
    """

    def __init__(self, hidden_size: int, num_attention_heads: int, attention_head_size: int, all_head_size: int, num_rounds: int, velocity_threshold: float):
        """
        Initializes the multi-round speculative decoding layer.

        Parameters:
        - hidden_size (int): The hidden size of the transformer model.
        - num_attention_heads (int): Number of attention heads.
        - attention_head_size (int): Size of each attention head.
        - all_head_size (int): Size of the total attention block.
        - num_rounds (int): Number of speculative decoding rounds.
        - velocity_threshold (float): Threshold for accepting speculative predictions.
        """
        super().__init__(hidden_size, num_attention_heads, attention_head_size, all_head_size)
        self.num_rounds = num_rounds
        self.velocity_threshold = velocity_threshold

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Performs the forward pass of the multi-round speculative decoding layer.

        Parameters:
        - query (torch.Tensor): Tensor containing the query sequence.
        - key (torch.Tensor): Tensor containing the key sequence.
        - value (torch.Tensor): Tensor containing the values to be attended to.

        Returns:
        - Dict[str, torch.Tensor]: Dictionary containing the output tensor and additional speculative decoding information.
        """
        outputs = {}
        speculative_outputs = []
        velocities = []

        for round_num in range(self.num_rounds):
            # Perform attention for this round
            output = super().forward(query, key, value)
            speculative_outputs.append(output)

            # Compute velocity based on Flow Theory
            if round_num > 0:
                velocity = torch.mean(torch.abs(speculative_outputs[round_num] - speculative_outputs[round_num - 1]))
                velocities.append(velocity)

            # Log the velocity for this round
            logger.info(f"Round {round_num} velocity: {velocity}")

            # Check if velocity is below the threshold
            if velocities and velocities[-1] < self.velocity_threshold:
                logger.info("Velocity threshold met. Accepting speculative prediction.")
                break

        # Store the final output and additional speculative decoding info
        outputs["final_output"] = speculative_outputs[-1]
        outputs["speculative_outputs"] = speculative_outputs
        outputs["velocities"] = velocities

        return outputs

# Example usage
if __name__ == "__main__":
    # Example tree structure
    tree = {0: [1, 2], 1: [3, 4], 2: [5]}

    # Example input tensor
    x = torch.randn(3, 6, 10)

    # Create an instance of the tree attention layer
    tree_attention = TreeAttentionLayer(hidden_size=10, num_attention_heads=2, attention_head_size=5, all_head_size=10, tree=tree)

    # Perform forward pass on the tree attention layer
    output = tree_attention(x)
    print(output.shape)  # Output shape: [3, 6, 10]

    # Create an instance of the multi-round speculative decoding layer
    num_rounds = 3
    velocity_threshold = 0.5
    speculative_layer = MultiRoundSpeculativeDecodingLayer(hidden_size=10, num_attention_heads=2, attention_head_size=5, all_head_size=10,
                                                          num_rounds=num_rounds, velocity_threshold=velocity_threshold)

    # Example query, key, and value tensors
    query = torch.randn(3, 8, 10)
    key = torch.randn(3, 8, 10)
    value = torch.randn(3, 8, 10)

    # Perform forward pass on the multi-round speculative decoding layer
    speculative_outputs = speculative_layer(query, key, value)
    print(speculative_outputs["final_output"].shape)  # Output shape: [3, 8, 10]
    print(len(speculative_outputs["speculative_outputs"]))  # Number of speculative outputs: 3
    print(len(speculative_outputs["velocities"]))  # Number of velocity values: 2