from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


class MambaBlock(nn.Module):
    """Minimal Mamba block for classification-focused prototypes."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(
                self.d_inner, 1
            )
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _selective_scan(self, u: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = u.shape
        proj = self.x_proj(u)
        dt_raw, b_raw, c_raw = torch.split(
            proj, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_raw))

        A = -torch.exp(self.A_log).to(dtype=u.dtype)
        D = self.D.to(dtype=u.dtype)
        state = torch.zeros(
            batch_size,
            self.d_inner,
            self.d_state,
            device=u.device,
            dtype=u.dtype,
        )
        outputs: list[torch.Tensor] = []
        for t in range(seq_len):
            dt_t = dt[:, t, :]
            u_t = u[:, t, :]
            b_t = b_raw[:, t, :]
            c_t = c_raw[:, t, :]

            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            dB_u = dt_t.unsqueeze(-1) * b_t.unsqueeze(1) * u_t.unsqueeze(-1)
            state = dA * state + dB_u
            y_t = torch.sum(state * c_t.unsqueeze(1), dim=-1) + D.unsqueeze(0) * u_t
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[..., : residual.shape[1]]
        x = F.silu(x.transpose(1, 2))
        y = self._selective_scan(x)
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)
        return residual + y


class MambaEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)

