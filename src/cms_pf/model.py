from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, hidden, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.net(x).flatten(1)


class TemporalFusion(nn.Module):
    def __init__(self, hidden: int, descriptor_dim: int = 4, pathway: str = "fusion"):
        super().__init__()
        if pathway not in {"fusion", "image", "descriptor"}:
            raise ValueError(f"Unknown pathway: {pathway}")
        self.pathway = pathway
        self.image = ImageEncoder(hidden)
        self.time_embed = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.desc = nn.Sequential(
            nn.Linear(descriptor_dim, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, hidden)
        )
        self.score = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.out = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU())

    def forward(self, fields, descriptors, times, mask=None):
        b, t, _, h, w = fields.shape
        time_z = self.time_embed(times[..., None])
        desc_z = self.desc(descriptors)
        if self.pathway == "descriptor":
            joint = desc_z + time_z
        else:
            image_z = self.image(fields.reshape(b * t, 1, h, w)).reshape(b, t, -1)
            joint = image_z + time_z
        logits = self.score(joint).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=1)
        primary_context = (weights[..., None] * joint).sum(1)
        if self.pathway == "fusion":
            secondary_context = (weights[..., None] * desc_z).sum(1)
        else:
            secondary_context = torch.zeros_like(primary_context)
        return self.out(torch.cat([primary_context, secondary_context], dim=-1))


class GaussianMixturePosterior(nn.Module):
    def __init__(self, hidden: int = 96, parameter_dim: int = 2, components: int = 4, pathway: str = "fusion"):
        super().__init__()
        self.encoder = TemporalFusion(hidden, pathway=pathway)
        self.parameter_dim = parameter_dim
        self.components = components
        per_component = parameter_dim * 2
        self.head = nn.Linear(hidden, components * per_component + components)

    def forward(self, fields, descriptors, times, mask=None):
        context = self.encoder(fields, descriptors, times, mask)
        raw = self.head(context)
        k, d = self.components, self.parameter_dim
        logits = raw[:, :k]
        rest = raw[:, k:].reshape(-1, k, 2 * d)
        mean = rest[..., :d]
        log_scale = rest[..., d:].clamp(-5.0, 3.0)
        return logits, mean, log_scale

    def nll(self, target, fields, descriptors, times, mask=None):
        logits, mean, log_scale = self(fields, descriptors, times, mask)
        z = (target[:, None, :] - mean) / log_scale.exp()
        log_component = -0.5 * (z**2 + 2 * log_scale + math.log(2 * math.pi)).sum(-1)
        return -torch.logsumexp(F.log_softmax(logits, -1) + log_component, -1).mean()

    @torch.no_grad()
    def moments(self, fields, descriptors, times, mask=None):
        logits, mean, log_scale = self(fields, descriptors, times, mask)
        weight = torch.softmax(logits, -1)
        mixture_mean = (weight[..., None] * mean).sum(1)
        second = (weight[..., None] * (log_scale.exp() ** 2 + mean**2)).sum(1)
        variance = (second - mixture_mean**2).clamp_min(1e-8)
        return mixture_mean, variance
