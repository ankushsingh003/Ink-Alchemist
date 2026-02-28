import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import wandb
from model import VesuviusModel, dice_loss

class VesuviusDataset(Dataset):
    def __init__(self, volume_path, label_path, mask_path, tile_size=256, stride=128, is_train=True):
        self.volume = np.load(volume_path, mmap_mode='r') # Use mmap for efficiency
        self.labels = np.load(label_path, mmap_mode='r')
        self.mask = np.load(mask_path, mmap_mode='r')
        self.tile_size = tile_size
        self.stride = stride
        
        # Pre-calculate tile locations
        self.tiles = []
        h, w = self.mask.shape
        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                # Only include tiles that have at least some mask pixels
                if np.sum(self.mask[y:y+tile_size, x:x+tile_size]) > 0:
                    self.tiles.append((y, x))
        
        # Split into train/val if needed (for simplicity, we just use a small subset for val here)
        if is_train:
            self.tiles = self.tiles[:-len(self.tiles)//10]
        else:
            self.tiles = self.tiles[-len(self.tiles)//10:]

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        y, x = self.tiles[idx]
        
        # Extract tile and normalize
        tile = self.volume[:, y:y+self.tile_size, x:x+self.tile_size].astype(np.float32) / 255.0
        label = self.labels[y:y+self.tile_size, x:x+self.tile_size].astype(np.float32) / 255.0
        
        # PyTorch expects (C, Z, H, W) for 3D or (C, H, W) for 2D
        tile = torch.from_numpy(tile).unsqueeze(0) # (1, 32, 256, 256)
        label = torch.from_numpy(label).unsqueeze(0) # (1, 256, 256)
        
        return tile, label

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training")
    for tiles, labels in pbar:
        tiles, labels = tiles.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(tiles)
        
        # Combined Loss: BCE + Dice
        bce = criterion(outputs, labels)
        dice = dice_loss(outputs, labels)
        loss = 0.5 * bce + 0.5 * dice
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    
    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    total_dice = 0
    with torch.no_grad():
        for tiles, labels in tqdm(loader, desc="Validating"):
            tiles, labels = tiles.to(device), labels.to(device)
            outputs = model(tiles)
            total_dice += 1 - dice_loss(outputs, labels).item() # Dice Score = 1 - Dice Loss
            
    return total_dice / len(loader)

if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 4
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    TILE_SIZE = 256
    
    # Initialize WandB (Experiment Tracking)
    wandb.init(
        project="InkSense",
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "tile_size": TILE_SIZE,
            "architecture": "3D-CNN + Transformer",
            "loss": "BCE + Dice"
        }
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data Paths
    data_dir = "d:/ink_detection_fragments/processed"
    vol_path = os.path.join(data_dir, "fragment1_volume.npy")
    lbl_path = os.path.join(data_dir, "fragment1_labels.npy")
    msk_path = os.path.join(data_dir, "fragment1_mask.npy")
    
    # 1. Dataset & Loader (Fragment-Wise Strategy)
    # We use Fragment 1 as both proof-of-concept and initial training zone
    train_ds = VesuviusDataset(vol_path, lbl_path, msk_path, tile_size=TILE_SIZE, is_train=True)
    val_ds = VesuviusDataset(vol_path, lbl_path, msk_path, tile_size=TILE_SIZE, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 2. Model, Optimizer, Criterion
    model = VesuviusModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    
    # 3. Training Loop
    os.makedirs("checkpoints", exist_ok=True)
    for epoch in range(EPOCHS):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        avg_dice = validate(model, val_loader, device)
        
        # Log to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_dice": avg_dice
        })
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Dice: {avg_dice:.4f}")
        
        # Save Checkpoint
        torch.save(model.state_dict(), f"checkpoints/vesuvius_e{epoch+1}.pt")
    
    wandb.finish()
    print("Training Complete!")
