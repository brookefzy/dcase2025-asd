import torch
import torch.nn as nn
import torchvision.models as models

class ResNetAutoEncoder(nn.Module):
    def __init__(self, latent_dim=256, pretrained=False):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        # modify first conv for single-channel input
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # remove avgpool & fc
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc_enc = nn.Linear(512, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 512)
        # decoder: upsample + conv layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            # output is [1, n_mels, T]
        )

    def forward(self, x):
        # x: [B,1,n_mels,T]
        h = self.encoder(x)     # [B,512,h,w]
        h_pool = self.pool(h).view(x.size(0), -1)  # [B,512]
        z = self.fc_enc(h_pool)  # [B,latent_dim]
        h_rec = self.fc_dec(z).view(x.size(0),512,1,1)
        x_rec = self.decoder(h_rec)
        return x_rec, z