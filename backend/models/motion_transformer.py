
import torch
import torch.nn as nn
class MotionTransformer(nn.Module):
    def __init__(self, input_dim=3, d_model=128, nhead=8, layers=4, num_classes=4):
        super().__init__(); self.embed = nn.Linear(input_dim, d_model); enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True); self.encoder = nn.TransformerEncoder(enc, num_layers=layers); self.classifier = nn.Linear(d_model, num_classes)
    def forward(self, x):
        x = self.embed(x); x = self.encoder(x); x = x.mean(dim=1); return self.classifier(x)
