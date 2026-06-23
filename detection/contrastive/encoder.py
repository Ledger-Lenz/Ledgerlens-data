import torch
import torch.nn as nn

class TransactionEncoder(nn.Module):
    """2-layer MLP on aggregated features -> 128-dim embedding."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Projection head for contrastive loss (simclr typical practice)
        self.projector = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        """Returns the embeddings and projections."""
        h = self.encoder(x)
        z = self.projector(h)
        return h, z
