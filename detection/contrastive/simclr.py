import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
    
    Given a batch of N original examples, they are augmented to produce
    2N examples (N positive pairs). The loss encourages the representations
    of the positive pairs to be close and those of the 2N-2 negative pairs to be far apart.
    """
    
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j):
        """
        z_i: Representations of View 1 [Batch_Size, Dim]
        z_j: Representations of View 2 [Batch_Size, Dim]
        """
        batch_size = z_i.size(0)
        
        # Normalize representations along the feature dimension
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate all representations: shape [2*Batch_Size, Dim]
        z = torch.cat([z_i, z_j], dim=0)
        
        # Compute cosine similarity matrix: shape [2*Batch_Size, 2*Batch_Size]
        sim_matrix = torch.matmul(z, z.T)
        
        # Apply temperature scaling
        sim_matrix = sim_matrix / self.temperature
        
        # Labels: the positive pair for index k is k + batch_size (and vice versa)
        labels = torch.cat([torch.arange(batch_size) + batch_size, 
                            torch.arange(batch_size)], dim=0).to(z.device)
                            
        # Mask out self-similarities (diagonal elements)
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim_matrix.masked_fill_(mask, -9e15)
        
        # Calculate loss
        loss = self.criterion(sim_matrix, labels)
        
        # Average over the batch
        return loss / (2 * batch_size)
