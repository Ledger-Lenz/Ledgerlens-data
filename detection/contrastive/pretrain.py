import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

from detection.contrastive.encoder import TransactionEncoder
from detection.contrastive.simclr import NTXentLoss
from detection.contrastive.augmentations import augment_trade_sequence
from utils.logging import get_logger

logger = get_logger(__name__)

class UnlabeledWalletDataset(Dataset):
    """Dataset for loading unlabeled wallets and their trades."""
    def __init__(self, wallets_trades_list):
        self.wallets_trades_list = wallets_trades_list

    def __len__(self):
        return len(self.wallets_trades_list)

    def __getitem__(self, idx):
        # In a real implementation, this would load the trade history for the wallet
        df_trades = self.wallets_trades_list[idx]
        
        # We need two views for SimCLR
        view1 = augment_trade_sequence(df_trades)
        view2 = augment_trade_sequence(df_trades)
        
        # Placeholder for feature aggregation (turning a sequence of trades into a fixed-size vector)
        # e.g. calculating mean trade amount, standard dev of timestamps, etc.
        # Here we just generate a dummy aggregated feature vector of size 50 for the sake of example
        
        features_1 = np.random.randn(50).astype(np.float32)
        features_2 = np.random.randn(50).astype(np.float32)
        
        return features_1, features_2

def pretrain(dataset, epochs=10, batch_size=256, learning_rate=1e-3, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Pre-trains the TransactionEncoder using SimCLR."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Assuming input feature dimension is 50
    input_dim = 50
    encoder = TransactionEncoder(input_dim=input_dim).to(device)
    
    criterion = NTXentLoss(temperature=0.5).to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    
    logger.info(f"Starting pre-training for {epochs} epochs on {device}")
    encoder.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (view1, view2) in enumerate(dataloader):
            view1, view2 = view1.to(device), view2.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            _, z1 = encoder(view1)
            _, z2 = encoder(view2)
            
            loss = criterion(z1, z2)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
        
    return encoder

if __name__ == "__main__":
    # Dummy list of trades for 1000 wallets to simulate the pre-training loop
    dummy_wallets = [pd.DataFrame({'amount': np.random.rand(10), 'timestamp': np.arange(10)}) for _ in range(1000)]
    dataset = UnlabeledWalletDataset(dummy_wallets)
    pretrained_encoder = pretrain(dataset, epochs=2, batch_size=32)
    
    os.makedirs("./models", exist_ok=True)
    torch.save(pretrained_encoder.state_dict(), "./models/pretrained_encoder.pt")
    logger.info("Pre-training completed and encoder saved.")
