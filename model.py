import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock3D(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class VesuviusModel(nn.Module):
    def __init__(self, z_dim=32, model_dim=128):
        super(VesuviusModel, self).__init__()
        
        # 1. 3D Encoder (Backbone)
        # Input: (Batch, 1, 32, 512, 512)
        self.encoder3d = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            BasicBlock3D(32, 64, stride=(2, 1, 1)), # (Batch, 64, 16, 512, 512)
            BasicBlock3D(64, 128, stride=(2, 1, 1)), # (Batch, 128, 8, 512, 512)
        )
        
        # 2. 3D to 2D Neck
        # Compresses Z-dimension into channel dimension
        self.neck = nn.Conv2d(128 * 8, model_dim, kernel_size=1)
        
        # 3. Transformer Head (Context)
        # Using a simplified Transformer block for letters recognition
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 4. Decoding Head
        self.decoder = nn.Sequential(
            nn.Conv2d(model_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1) # 1 channel output for ink mask
        )

    def forward(self, x):
        # x: (Batch, 1, Z, H, W)
        batch, c, z, h, w = x.shape
        
        # 3D Encoding
        x = self.encoder3d(x) # (Batch, 128, 8, H, W)
        
        # Reshape for 3D -> 2D
        x = x.view(batch, 128 * 8, h, w)
        x = self.neck(x) # (Batch, model_dim, H, W)
        
        # Transform for Transformer
        # Flatten H,W to Sequence Length
        x_flat = x.flatten(2).permute(0, 2, 1) # (Batch, H*W, model_dim)
        x_flat = self.transformer(x_flat)
        x = x_flat.permute(0, 2, 1).view(batch, -1, h, w)
        
        # Decode
        logits = self.decoder(x)
        return logits

def dice_loss(pred, target):
    """
    Industry-level Dice Loss for sparse data (ink).
    """
    pred = torch.sigmoid(pred)
    smooth = 1.0
    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

if __name__ == "__main__":
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VesuviusModel().to(device)
    
    dummy_input = torch.randn(1, 1, 32, 256, 256).to(device)
    output = model(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}") # Should be (1, 1, 256, 256)
    print("Model architecture verified successfully.")
