import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

from detection.contrastive.encoder import TransactionEncoder
from utils.logging import get_logger

logger = get_logger(__name__)

class LabeledWalletDataset(Dataset):
    """Dataset for labeled wallets for fine-tuning."""
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class LinearProbe(nn.Module):
    """Linear probe classifier on top of a frozen encoder."""
    def __init__(self, encoder, num_classes=2):
        super().__init__()
        self.encoder = encoder
        
        # Freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # The encoder returns (h, z). We use h (the representation before the projector)
        # Assuming hidden_dim of encoder was 256. We need to know it dynamically.
        # But based on the encoder implementation, the output of the first block is what we want.
        # We'll just define a new linear layer.
        # Assuming the hidden dimension is 256
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        with torch.no_grad():
            h, _ = self.encoder(x)
        logits = self.classifier(h)
        return logits

def finetune(pretrained_encoder_path, train_features, train_labels, val_features, val_labels, epochs=20, batch_size=32, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Fine-tunes a linear probe on top of a frozen pre-trained encoder."""
    input_dim = train_features.shape[1]
    encoder = TransactionEncoder(input_dim=input_dim).to(device)
    
    if os.path.exists(pretrained_encoder_path):
        encoder.load_state_dict(torch.load(pretrained_encoder_path, map_location=device))
        logger.info(f"Loaded pre-trained encoder from {pretrained_encoder_path}")
    else:
        logger.warning(f"Pre-trained encoder not found at {pretrained_encoder_path}. Using random weights!")
        
    model = LinearProbe(encoder).to(device)
    
    train_dataset = LabeledWalletDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = LabeledWalletDataset(val_features, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    # Only optimize the classifier head
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    
    logger.info(f"Starting fine-tuning for {epochs} epochs on {device}")
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                logits = model(features)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        val_acc = correct / total
        logger.info(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "./models/finetuned_linear_probe.pt")
            
    logger.info(f"Fine-tuning completed. Best Val Acc: {best_val_acc:.4f}")
    return model

if __name__ == "__main__":
    # Dummy data for linear probe testing
    train_feats = np.random.randn(200, 50).astype(np.float32)
    train_labels = np.random.randint(0, 2, size=(200,))
    
    val_feats = np.random.randn(50, 50).astype(np.float32)
    val_labels = np.random.randint(0, 2, size=(50,))
    
    finetune("./models/pretrained_encoder.pt", train_feats, train_labels, val_feats, val_labels, epochs=5)
