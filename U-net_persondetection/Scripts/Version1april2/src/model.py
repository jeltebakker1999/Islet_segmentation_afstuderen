import torch as tt
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder (contracting path)
        self.enc1 = self.conv_block(1, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        
        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)
        
        # Decoder (expanding path)
        self.dec4 = self.upconv_block(512, 256)
        self.dec3 = self.upconv_block(256, 128)
        self.dec2 = self.upconv_block(128, 64)
        self.dec1 = self.upconv_block(64, 32)
        
        # Output layer
        self.final_conv = nn.Conv2d(32, 6, kernel_size=1)  # 5 classes (circle, square, star, moon, cross) + background

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU()
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoder (Upsampling)
        dec4 = self.dec4(bottleneck)
        dec4 = F.interpolate(dec4, size=enc3.shape[2:], mode="bilinear", align_corners=False)
        dec3 = self.dec3(dec4)

        dec3 = F.interpolate(dec3, size=enc3.shape[2:], mode="bilinear", align_corners=False)
        dec3 = dec3 + enc3  # skip connections

        dec2 = self.dec2(dec3)
        dec2 = F.interpolate(dec2, size=enc2.shape[2:], mode="bilinear", align_corners=False)
        dec2 = dec2 + enc2 #skip connections

        dec1 = self.dec1(dec2)
        dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode="bilinear", align_corners=False)
        dec1 = dec1 + enc1 #skip connections

        dec1 = F.interpolate(dec1, size=x.shape[2:], mode="bilinear", align_corners=False)
        out = self.final_conv(dec1)

        return out

if __name__ == "__main__":
    net = Model()
    data = tt.rand(3, 1, 140, 140)  # Batch of 3 images
    output = net(data)
    print("Output shape:", output.shape)  # Expected: (3, 5, 140, 140)
