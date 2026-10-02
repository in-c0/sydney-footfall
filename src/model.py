"""Neural network definitions."""
import torch
import torch.nn as nn

SOURCES = ["cos_survey", "cos_sensor", "liverpool_camera"]

def mlp(i, h, o, depth=2, p=0.1):
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(p)]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))

class FootfallNet(nn.Module):
    """Spatio-temporal footfall model.

    A spatial tower embeds OSM context around a point, a temporal tower embeds
    hour/day/season/weather/COVID-era, and a fusion head models their interaction
    (e.g. office blocks peak at 8am and 5pm on weekdays, nightlife streets at 10pm
    on Saturdays). A per-source offset absorbs differences in how each dataset counts
    (two-way survey cross-sections vs single-direction camera fields of view).
    Output: predicted log1p(people per hour).
    """
    def __init__(self, n_spatial, n_temporal, d=64):
        super().__init__()
        self.spatial = mlp(n_spatial, 128, d, p=0.2)
        self.temporal = mlp(n_temporal, 128, d, p=0.1)
        self.head = mlp(3 * d, 128, 1, depth=2, p=0.1)
        self.src_bias = nn.Embedding(len(SOURCES), 1)
        self.src_scale = nn.Embedding(len(SOURCES), 1)
        nn.init.zeros_(self.src_bias.weight); nn.init.zeros_(self.src_scale.weight)

    def forward(self, xs, xt, src):
        s, t = self.spatial(xs), self.temporal(xt)
        z = self.head(torch.cat([s, t, s * t], -1)).squeeze(-1)
        return z * torch.exp(self.src_scale(src).squeeze(-1)) + self.src_bias(src).squeeze(-1)

class ShortTermForecaster(nn.Module):
    """Next-24h forecaster for live counters.

    Residual design: starts from the seasonal-naive forecast (same hours one week
    earlier) and learns a correction from (a) a GRU summary of the past 7 days,
    normalised per window, and (b) tomorrow's calendar + weather features
    (holiday, rain, lockdown flags...). Can therefore only improve on naive where
    the data supports it."""
    def __init__(self, n_future_feat, hidden=48):
        super().__init__()
        self.gru = nn.GRU(3, hidden, batch_first=True)
        self.fut = nn.Sequential(nn.Linear(n_future_feat + 3, 64), nn.GELU())
        self.out = nn.Sequential(nn.Linear(hidden + 64, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)

    def forward(self, past, past_hour, fut_feat):
        # past: (B,168) log counts ending at the forecast origin; fut_feat: (B,24,F)
        mu = past.mean(1, keepdim=True)
        ang = 2 * torch.pi * past_hour / 24
        x = torch.stack([past - mu, torch.sin(ang), torch.cos(ang)], -1)
        _, h = self.gru(x[:, ::2])  # 2-hourly stride keeps the CPU cost low
        naive = past[:, :24]
        hz = torch.arange(24, dtype=past.dtype, device=past.device) * 2 * torch.pi / 24
        hz = torch.stack([torch.sin(hz), torch.cos(hz)], -1).expand(past.shape[0], -1, -1)
        f = torch.cat([fut_feat, hz, (naive - mu).unsqueeze(-1)], -1)
        h = h[-1].unsqueeze(1).expand(-1, 24, -1)
        return naive + self.out(torch.cat([h, self.fut(f)], -1)).squeeze(-1)
