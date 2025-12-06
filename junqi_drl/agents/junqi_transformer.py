from typing import Dict, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

class JunqiMoveTransformer(nn.Module):
    """
    Ataraxos-style Transformer for Junqi.
    Implements Key-Query Matrix for explicit (From -> To) move selection.
    """

    BOARD_CONFIGS: Dict[str, Dict[str, int]] = {
        "standard": {"height": 12, "width": 5, "channels": 31},
        "small": {"height": 8, "width": 3, "channels": 23},
    }

    def __init__(
        self,
        board_variant: str = "standard",
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if board_variant not in self.BOARD_CONFIGS:
            raise ValueError(f"Unknown board variant '{board_variant}'")
        board = self.BOARD_CONFIGS[board_variant]
        self.height = board["height"]
        self.width = board["width"]
        self.channels = board["channels"]
        self.seq_len = self.height * self.width
        self.total_actions = self.seq_len * self.seq_len + 1  # include pass action
        
        # 1. Input Projection & Positional Embeddings 
        self.input_proj = nn.Linear(self.channels, d_model)
        self.row_embeddings = nn.Parameter(torch.zeros(self.height, d_model))
        self.col_embeddings = nn.Parameter(torch.zeros(self.width, d_model))

        # 2. Transformer Backbone 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. [NEW] Key-Query Heads for Moving Pieces
        self.query_head = nn.Linear(d_model, d_model)  # W_q: starting square features
        self.key_head = nn.Linear(d_model, d_model)    # W_k: ending square features

        self.value_head = nn.Linear(d_model, 1) # V: state value
        self.pass_head = nn.Linear(d_model, 1)  # Separate head for pass

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # ... (Standard init) ...
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.normal_(self.row_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.col_embeddings, mean=0.0, std=0.02)
        # Init new heads
        nn.init.xavier_uniform_(self.query_head.weight)
        nn.init.xavier_uniform_(self.key_head.weight)
        nn.init.xavier_uniform_(self.pass_head.weight)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, H, W, C)
            mask: (B, Seq_Len * Seq_Len + 1) - flattened mask for From-To pairs + pass
        Returns:
            move_logits: (B, Seq_Len * Seq_Len) representing Flat (From -> To)
            values: (B, 1)
        """
        batch_size = x.shape[0]
        
        # --- 1. Encode Board ---
        tokens = x.reshape(batch_size, self.seq_len, self.channels) # (B, S, C)
        tokens = self.input_proj(tokens)

        # Add 2D Positional Embeddings
        row_emb = self.row_embeddings.unsqueeze(1).expand(self.height, self.width, -1)
        col_emb = self.col_embeddings.unsqueeze(0).expand(self.height, self.width, -1)
        pos_emb = (row_emb + col_emb).reshape(1, self.seq_len, -1)
        tokens = tokens + pos_emb

        encoded = self.encoder(tokens) # (B, S, d_model)

        # --- 2. [NEW] Compute Move Logits (Key-Query Attention) ---
        # Query: will represent the starting square
        # Key:  will represent the ending square
        
        Q = self.query_head(encoded) # (B, S, d_model)
        K = self.key_head(encoded)   # (B, S, d_model)

        # Matrix Multiplication: (B, S, d) @ (B, d, S) -> (B, S, S)
        # move_matrix[b, i, j] = Score of moving from i to j
        move_matrix = torch.matmul(Q, K.transpose(-2, -1)) 
        
        # Scale by sqrt(d_model) (Standard attention scaling)
        d_k = Q.size(-1)
        move_matrix = move_matrix / (d_k ** 0.5)

        # Flatten to match RL action space: (B, S * S)
        # Index k corresponds to: From (k // S) -> To (k % S)
        action_logits = move_matrix.flatten(start_dim=1) 

        # --- 3. Append Pass Logit ---
        pooled = encoded.mean(dim=1)
        pass_logit = self.pass_head(pooled)  # (B, 1)
        action_logits = torch.cat([action_logits, pass_logit], dim=1)

        # --- 4. Masking ---
        if mask is not None:
            if mask.shape[-1] != self.total_actions:
                raise ValueError(
                    f"Mask has wrong size {mask.shape[-1]} (expected {self.total_actions})"
                )
            illegal = mask <= 0
            action_logits = action_logits.masked_fill(illegal, -1e9)

        # --- 5. Value ---
        values = self.value_head(pooled)

        return action_logits, values
