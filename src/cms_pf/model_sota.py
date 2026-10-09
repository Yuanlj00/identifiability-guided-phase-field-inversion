from __future__ import annotations

import math
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x):
        return torch.nn.functional.silu(x + self.net(x))


class MultiScaleMorphologyEncoder(nn.Module):
    """Image encoder retaining local interfaces and long-range domain geometry."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(6, 24), nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 48), nn.SiLU(),
        )
        self.local = ResidualBlock(48, 1)
        self.meso = ResidualBlock(48, 2)
        self.long = ResidualBlock(48, 4)
        self.project = nn.Sequential(
            nn.Conv2d(144, 96, 1), nn.SiLU(), nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(), nn.Linear(96 * 4, out_dim), nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        z = self.stem(x)
        return self.project(torch.cat([self.local(z), self.meso(z), self.long(z)], dim=1))


class PhysicsSummary(nn.Module):
    """Differentiable field statistics tied to interfaces and gradient energy."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.register_buffer("sobel_x", torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        ).reshape(1, 1, 3, 3) / 8.0)
        self.register_buffer("sobel_y", torch.tensor(
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        ).reshape(1, 1, 3, 3) / 8.0)
        self.register_buffer("laplace", torch.tensor(
            [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]
        ).reshape(1, 1, 3, 3))
        self.net = nn.Sequential(nn.Linear(7, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, x):
        gx = torch.nn.functional.conv2d(x, self.sobel_x, padding=1)
        gy = torch.nn.functional.conv2d(x, self.sobel_y, padding=1)
        lap = torch.nn.functional.conv2d(x, self.laplace, padding=1)
        flat = x.flatten(1)
        stats = torch.stack([
            flat.mean(1), flat.std(1), flat.abs().mean(1),
            (flat ** 2).mean(1), (flat ** 4).mean(1),
            (gx.square() + gy.square()).mean((1, 2, 3)), lap.abs().mean((1, 2, 3)),
        ], dim=1)
        return self.net(stats)


class PortableGRUCell(nn.Module):
    """GRU cell expressed with primitive ops supported by Ascend and CUDA/CPU."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, 3 * hidden_dim)
        self.hidden_projection = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden):
        input_reset, input_update, input_new = self.input_projection(x).chunk(3, dim=-1)
        hidden_reset, hidden_update, hidden_new = self.hidden_projection(hidden).chunk(3, dim=-1)
        reset = torch.sigmoid(input_reset + hidden_reset)
        update = torch.sigmoid(input_update + hidden_update)
        candidate = torch.tanh(input_new + reset * hidden_new)
        return (1.0 - update) * candidate + update * hidden


class PortableStackedGRU(nn.Module):
    """Unrolled stacked GRU avoiding the unsupported Ascend DynamicGRUV2 op."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.cells = nn.ModuleList([
            PortableGRUCell(input_dim if layer == 0 else hidden_dim, hidden_dim)
            for layer in range(num_layers)
        ])
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def forward(self, sequence):
        batch, steps, _ = sequence.shape
        layer_input = sequence
        final_hidden = []
        for layer_index, cell in enumerate(self.cells):
            hidden = sequence.new_zeros((batch, self.hidden_dim))
            outputs = []
            for step in range(steps):
                hidden = cell(layer_input[:, step], hidden)
                outputs.append(hidden)
            layer_input = torch.stack(outputs, dim=1)
            final_hidden.append(hidden)
            if layer_index + 1 < len(self.cells) and self.dropout:
                layer_input = torch.nn.functional.dropout(
                    layer_input, p=self.dropout, training=self.training
                )
        return layer_input, torch.stack(final_hidden, dim=0)


class PhysicsTemporalEncoder(nn.Module):
    def __init__(self, hidden: int = 160, descriptor_dim: int = 4,
                 pathway: str = "fusion", temporal: str = "transformer"):
        super().__init__()
        if pathway not in {"fusion", "gated", "image", "descriptor"}:
            raise ValueError(pathway)
        if temporal not in {"transformer", "deepset", "gru", "tcn"}:
            raise ValueError(temporal)
        self.pathway, self.temporal = pathway, temporal
        image_dim, side_dim = 128, 32
        self.image = MultiScaleMorphologyEncoder(image_dim)
        self.physics = PhysicsSummary(side_dim)
        self.descriptor = nn.Sequential(
            nn.Linear(descriptor_dim, side_dim), nn.SiLU(), nn.Linear(side_dim, side_dim)
        )
        self.time = nn.Sequential(nn.Linear(3, side_dim), nn.SiLU(), nn.Linear(side_dim, side_dim))
        if pathway in {"fusion", "gated"}:
            input_dim = image_dim + 2 * side_dim
        elif pathway == "image":
            input_dim = image_dim
        else:
            input_dim = side_dim
        self.token = nn.Sequential(nn.Linear(input_dim + side_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        if pathway == "gated":
            self.fusion_gate = nn.Sequential(
                nn.Linear(2 * side_dim, image_dim), nn.SiLU(), nn.Linear(image_dim, image_dim), nn.Sigmoid()
            )
        if temporal == "transformer":
            layer = nn.TransformerEncoderLayer(
                hidden, nhead=5, dim_feedforward=hidden * 3, dropout=0.08,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.sequence = nn.TransformerEncoder(layer, num_layers=3)
        elif temporal == "gru":
            self.sequence = PortableStackedGRU(hidden, hidden, num_layers=2, dropout=0.08)
        elif temporal == "tcn":
            self.sequence = nn.Sequential(
                nn.Conv1d(hidden, hidden, 3, padding=1), nn.SiLU(),
                nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2), nn.SiLU(),
                nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4), nn.SiLU(),
            )
        else:
            self.sequence = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.pool = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.out = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden))

    def encode_frames(self, fields, descriptors, times):
        b, t, _, h, w = fields.shape
        flat = fields.reshape(b * t, 1, h, w)
        time_features = torch.stack([times, torch.sin(math.pi * times), torch.cos(math.pi * times)], -1)
        parts = []
        time_z = self.time(time_features)
        image_z = None
        if self.pathway != "descriptor":
            image_z = self.image(flat).reshape(b, t, -1)
        if self.pathway in {"fusion", "gated"}:
            physics_z = self.physics(flat).reshape(b, t, -1)
            descriptor_z = self.descriptor(descriptors)
            if self.pathway == "gated":
                gate = self.fusion_gate(torch.cat([descriptor_z, time_z], dim=-1))
                image_z = image_z * (0.25 + 1.5 * gate)
            parts.extend([image_z, physics_z, descriptor_z])
        elif self.pathway == "image":
            parts.append(image_z)
        elif self.pathway == "descriptor":
            parts.append(self.descriptor(descriptors))
        parts.append(time_z)
        return self.token(torch.cat(parts, dim=-1))

    def aggregate(self, token, mask=None):
        b, t, _ = token.shape
        if mask is None:
            mask = torch.ones((b, t), dtype=torch.bool, device=token.device)
        if self.temporal == "transformer":
            encoded = self.sequence(token, src_key_padding_mask=~mask)
        elif self.temporal == "gru":
            encoded = self.sequence(token * mask[..., None])[0]
        elif self.temporal == "tcn":
            encoded = self.sequence((token * mask[..., None]).transpose(1, 2)).transpose(1, 2)
        else:
            encoded = self.sequence(token)
        logits = self.pool(encoded).squeeze(-1).masked_fill(~mask, -1e4)
        weight = torch.softmax(logits, dim=1)
        attentive = (weight[..., None] * encoded).sum(1)
        denom = mask.sum(1, keepdim=True).clamp_min(1)
        mean = (encoded * mask[..., None]).sum(1) / denom
        return self.out(torch.cat([attentive, mean], dim=-1))

    def forward(self, fields, descriptors, times, mask=None):
        return self.aggregate(self.encode_frames(fields, descriptors, times), mask)


class ConditionalAffineLayer(nn.Module):
    def __init__(self, context_dim: int, transform_index: int, hidden: int = 128):
        super().__init__()
        self.transform_index = transform_index
        self.condition_index = 1 - transform_index
        self.net = nn.Sequential(
            nn.Linear(context_dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2),
        )

    def transform_parameters(self, value, context):
        raw = self.net(torch.cat([value, context], dim=-1))
        shift = raw[..., :1]
        log_scale = 1.8 * torch.tanh(raw[..., 1:2] / 1.8)
        return shift, log_scale

    def forward(self, y, context):
        cond = y[..., self.condition_index:self.condition_index + 1]
        shift, log_scale = self.transform_parameters(cond, context)
        z = y.clone()
        idx = self.transform_index
        z[..., idx:idx + 1] = (y[..., idx:idx + 1] - shift) * torch.exp(-log_scale)
        return z, -log_scale.squeeze(-1)

    def inverse(self, z, context):
        cond = z[..., self.condition_index:self.condition_index + 1]
        shift, log_scale = self.transform_parameters(cond, context)
        y = z.clone()
        idx = self.transform_index
        y[..., idx:idx + 1] = z[..., idx:idx + 1] * torch.exp(log_scale) + shift
        return y


class ConditionalFlowPosterior(nn.Module):
    def __init__(self, hidden: int = 160, parameter_dim: int = 2, flow_layers: int = 10,
                 pathway: str = "fusion", temporal: str = "transformer"):
        super().__init__()
        if parameter_dim != 2:
            raise ValueError("The alternating conditional flow is defined for two identifiable groups")
        self.encoder = PhysicsTemporalEncoder(hidden, pathway=pathway, temporal=temporal)
        self.layers = nn.ModuleList([
            ConditionalAffineLayer(hidden, i % 2, min(160, hidden)) for i in range(flow_layers)
        ])
        self.parameter_dim = parameter_dim
        self.pathway, self.temporal = pathway, temporal

    def context(self, fields, descriptors, times, mask=None):
        return self.encoder(fields, descriptors, times, mask)

    def log_prob_from_context(self, target, context):
        z, logdet = target, torch.zeros(target.shape[0], device=target.device)
        for layer in self.layers:
            z, delta = layer(z, context)
            logdet = logdet + delta
        base = -0.5 * (z.square() + math.log(2 * math.pi)).sum(-1)
        return base + logdet

    def nll(self, target, fields, descriptors, times, mask=None):
        return -self.log_prob_from_context(target, self.context(fields, descriptors, times, mask)).mean()

    def sample_from_context(self, context, n_samples: int):
        b, d = context.shape[0], self.parameter_dim
        z = torch.randn((b, n_samples, d), device=context.device, dtype=context.dtype)
        expanded = context[:, None, :].expand(-1, n_samples, -1)
        y = z
        for layer in reversed(self.layers):
            y = layer.inverse(y, expanded)
        return y

    @torch.no_grad()
    def sample(self, fields, descriptors, times, mask=None, n_samples: int = 512):
        return self.sample_from_context(self.context(fields, descriptors, times, mask), n_samples)

    @torch.no_grad()
    def moments(self, fields, descriptors, times, mask=None, n_samples: int = 512):
        sample = self.sample(fields, descriptors, times, mask, n_samples)
        return sample.mean(1), sample.var(1, unbiased=False)
