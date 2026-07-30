import torch
import torch.nn as nn
import torch.nn.functional as F


class PEOA(nn.Module):
    """Parameter-Efficient Online Adapter described in Eqs. (1)-(6)."""

    def __init__(self, in_dim, hidden_dim=64, memory_slots=64, top_k=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_slots = memory_slots
        self.top_k = top_k

        self.norm = nn.LayerNorm(in_dim)
        self.norm_scale = nn.Parameter(torch.full((in_dim,), 1e-6))
        self.identity_scale = nn.Parameter(torch.ones(in_dim))
        self.project_down = nn.Linear(in_dim, hidden_dim)

        self.depthwise_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    groups=hidden_dim,
                )
                for kernel_size in (3, 5, 7)
            ]
        )
        self.pointwise = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.memory_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.project_up = nn.Linear(hidden_dim, in_dim)

        self.register_buffer("memory_bank", torch.zeros(memory_slots, hidden_dim))
        self.register_buffer("memory_ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("memory_filled", torch.zeros((), dtype=torch.long))
        self._last_latent = None
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # Keep the initial residual path equal to the frozen source model.
        # The first update learns the up-projection; subsequent updates also
        # propagate into the remaining PEOA layers.
        nn.init.zeros_(self.project_up.weight)
        nn.init.zeros_(self.project_up.bias)

    def _retrieve_memory(self, latent):
        filled = int(self.memory_filled.item())
        if filled == 0:
            return latent

        query = F.normalize(self.memory_projection(latent), dim=-1)
        memory = self.memory_bank[:filled]
        similarities = query @ F.normalize(memory, dim=-1).transpose(0, 1)
        k = min(self.top_k, filled)
        indices = similarities.topk(k=k, dim=-1).indices
        retrieved = memory[indices].mean(dim=1)
        return latent + retrieved

    @torch.no_grad()
    def commit_memory(self):
        """Store one latent representation per test sample after inference."""
        if self._last_latent is None:
            return
        for vector in self._last_latent:
            pointer = int(self.memory_ptr.item())
            self.memory_bank[pointer].copy_(vector)
            self.memory_ptr.fill_((pointer + 1) % self.memory_slots)
            self.memory_filled.fill_(min(int(self.memory_filled.item()) + 1, self.memory_slots))
        self._last_latent = None

    def forward(self, x, hw_shapes=None):
        identity = x
        normalized = self.norm(x) * self.norm_scale + x * self.identity_scale
        reduced = self.project_down(normalized)

        batch, tokens, channels = reduced.shape
        height, width = hw_shapes
        if tokens != height * width:
            raise ValueError(f"Expected {height * width} tokens, received {tokens}")

        feature_map = reduced.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        feature_map = torch.stack(
            [convolution(feature_map) for convolution in self.depthwise_convs], dim=0
        ).mean(dim=0)

        latent = F.adaptive_avg_pool2d(feature_map, output_size=1).flatten(1)
        enhanced = self._retrieve_memory(latent)
        feature_map = feature_map + enhanced[:, :, None, None]
        feature_map = F.relu(feature_map + self.pointwise(feature_map))

        reduced = feature_map.permute(0, 2, 3, 1).reshape(batch, tokens, channels)
        self._last_latent = latent.detach()
        return identity + self.project_up(reduced)


# Backward-compatible name used throughout the released code.
bba = PEOA
