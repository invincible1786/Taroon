"""Model architectures for Part 2: Build & Defend.

Contains:
1. EEGNet (baseline validated in Part 1)
2. VanillaEEGTransformer (per-timestep tokens, no EEG prior)
3. PatchStem (temporal conv -> depthwise spatial -> power -> pool -> tokens)
4. YourModel (learned temporal patch tokenisation with conv stem + transformer)
"""
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """EEGNet-8,2 (Lawhern et al., 2018), adapted to 160 Hz."""

    def __init__(self, n_ch, n_times, n_classes=2, F1=8, D=2, F2=16,
                 kern=80, drop=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(drop))
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(drop))
        with torch.no_grad():
            n_feat = self._features(torch.zeros(1, n_ch, n_times)).shape[1]
        self.head = nn.Linear(n_feat, n_classes)

    def _features(self, x):
        x = self.block2(self.block1(x.unsqueeze(1)))
        return x.flatten(1)

    def forward(self, x):
        return self.head(self._features(x))


class VanillaEEGTransformer(nn.Module):
    """Vanilla Transformer operating on raw per-timestep tokens without EEG priors.
    
    Each timestep sample across electrodes is linearly projected to d_model,
    yielding sequence length T and an O(T^2) attention matrix.
    """

    def __init__(self, n_ch, n_times, n_classes=2, d_model=32, n_heads=4,
                 n_layers=2, ff=64, drop=0.25):
        super().__init__()
        self.proj = nn.Linear(n_ch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_times, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, ff, dropout=drop, batch_first=True,
            norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(d_model, n_classes))

    def forward(self, x):
        # x: (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        h = self.proj(x) + self.pos
        h = self.enc(h)
        return self.head(self.norm(h).mean(1))


class PatchStem(nn.Module):
    """Temporal conv bank -> depthwise spatial filters -> power -> windowed average -> log -> tokens.
    
    Embeds prior knowledge of EEG:
    1. Local temporal bandpass filtering (0.5 s kernel)
    2. Spatial dipole un-mixing (medial vs lateral electrodes)
    3. Power squaring (ERD is a power decrease)
    4. Adaptive temporal pooling to N patch tokens
    """

    def __init__(self, n_ch, n_tokens=16, F1=8, D=2, k=80, d_model=32, drop=0.25):
        super().__init__()
        self.temporal = nn.Conv2d(1, F1, (1, k), padding=(0, k // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.spatial = nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool = nn.AdaptiveAvgPool2d((1, n_tokens))
        self.proj = nn.Linear(F1 * D, d_model)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x: (B, C, T)
        x = self.bn1(self.temporal(x.unsqueeze(1)))
        x = self.bn2(self.spatial(x))  # (B, F1*D, 1, T')
        x = self.pool(x.pow(2))        # windowed band power
        x = torch.log(x + 1e-6).squeeze(2).transpose(1, 2)  # (B, N, F1*D)
        return self.drop(self.proj(x))


class YourModel(nn.Module):
    """Attention-based decoder with learned temporal-patch conv stem prior.
    
    Shrinks sequence length from T (e.g. 561) to n_tokens (16), cutting attention
    from 561^2 (314,721) to 16^2 (256) and bounding overfitting on scarce data.
    """

    def __init__(self, n_ch, n_times, n_classes=2, n_tokens=16, d_model=32,
                 n_heads=4, n_layers=2, ff=64, attn_drop=0.1, drop=0.25):
        super().__init__()
        self.n_tokens = n_tokens
        self.stem = PatchStem(n_ch, n_tokens=n_tokens, d_model=d_model, drop=drop)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        if n_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model, n_heads, ff, dropout=attn_drop, batch_first=True,
                norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        else:
            self.enc = nn.Identity()
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(d_model, n_classes))

    def forward(self, x):
        tokens = self.stem(x) + self.pos
        h = self.enc(tokens)
        return self.head(self.norm(h).mean(1))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

