import torch
import torch.nn as nn

class StudentModel(nn.Module):
    def __init__(self, in_channels=1, n_mels=64, latent_dim=128):
        super().__init__()
        # a smaller CNN-based AE
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc_enc = nn.Linear(64, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 3, stride=1, padding=1)
        )

    def forward(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        z = self.fc_enc(h)
        h2 = self.fc_dec(z).view(x.size(0),64,1,1)
        rec = self.decoder(h2)
        # produce anomaly score directly (e.g. mean rec error)
        score = torch.mean((x - rec)**2, dim=[1,2,3])
        return score, z