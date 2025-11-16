from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .move_index import POLICY_SIZE, PLANES_PER_SQUARE
from .encoding import NUM_FEATURE_PLANES


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        return self.relu(out)


class TinyPolicyResNet(nn.Module):
    """
    Minimal policy-only ResNet.
    Input: [B, C, 8, 8]
    Output: logits [B, 4672] corresponding to 73 planes x 8 x 8.
    """

    def __init__(self, in_channels: int = NUM_FEATURE_PLANES, channels: int = 64, blocks: int = 4, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResidualBlock(channels, dropout=dropout) for _ in range(blocks)])
        self.trunk_dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        # Policy head that outputs 73 planes
        policy_layers = [
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            policy_layers.append(nn.Dropout2d(dropout))
        policy_layers.append(nn.Conv2d(32, PLANES_PER_SQUARE, kernel_size=1, bias=True))
        self.policy_head = nn.Sequential(*policy_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.trunk(x)
        x = self.trunk_dropout(x)
        p = self.policy_head(x)  # [B, 73, 8, 8]
        return p.flatten(start_dim=1)  # [B, 4672]


def make_model() -> TinyPolicyResNet:
    return TinyPolicyResNet()


class PolicyOnlyResNet(nn.Module):
    """Policy-only ResNet.

    Stem: 3x3 conv -> BN -> ReLU
    Trunk: n_blocks residual blocks (Conv-BN-ReLU x2 + skip)
    Policy head: 1x1 conv -> BN -> ReLU -> 1x1 conv to 73 planes; flatten to 4672 logits
    """

    def __init__(self, in_channels: int = NUM_FEATURE_PLANES, width: int = 64, n_blocks: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.width = width
        self.n_blocks = n_blocks

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResidualBlock(width, dropout=dropout) for _ in range(n_blocks)])
        # Dropout after trunk, before heads
        self.trunk_dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        policy_layers = [
            nn.Conv2d(width, width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            policy_layers.append(nn.Dropout2d(dropout))
        policy_layers.append(nn.Conv2d(width, PLANES_PER_SQUARE, kernel_size=1, bias=True))
        self.policy_head = nn.Sequential(*policy_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.trunk(x)
        x = self.trunk_dropout(x)
        p = self.policy_head(x)
        return p.flatten(start_dim=1)

    # Backward-compatibility shim to match PolicyValueResNet API
    def forward_policy_only(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class PolicyValueResNet(nn.Module):
    """
    Policy+Value ResNet sharing stem + trunk.

    Defaults: in_channels=NUM_FEATURE_PLANES, width=64, n_blocks=8

    Stem: Conv2d(in_channels→width) + BN + ReLU
    Trunk: n_blocks × ResidualBlock(width)
    Policy head: 1×1 Conv (width→width) → BN → ReLU → 1×1 Conv (width→73) → flatten to [B,4672]
    Value head: 1×1 Conv (width→32) → BN → ReLU → AdaptiveAvgPool2d(1) → Flatten → Linear(32→1) → Tanh → [B,1]
    """

    def __init__(self, in_channels: int = NUM_FEATURE_PLANES, width: int = 64, n_blocks: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.width = width
        self.n_blocks = n_blocks

        # Shared stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )

        # Shared trunk
        self.trunk = nn.Sequential(*[ResidualBlock(width, dropout=dropout) for _ in range(n_blocks)])
        # Dropout after trunk, before heads
        self.trunk_dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        # Policy head
        policy_layers = [
            nn.Conv2d(width, width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            policy_layers.append(nn.Dropout2d(dropout))
        policy_layers.append(nn.Conv2d(width, PLANES_PER_SQUARE, kernel_size=1, bias=True))
        self.policy_head = nn.Sequential(*policy_layers)

        # Value head
        value_layers = [
            nn.Conv2d(width, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            value_layers.append(nn.Dropout2d(dropout))
        value_layers.append(nn.AdaptiveAvgPool2d(1))
        self.value_head = nn.Sequential(*value_layers)
        self.value_fc = nn.Linear(32, 1)
        self.value_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.value_act = nn.Tanh()

    def _shared_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.trunk(x)
        return x

    def forward_policy_only(self, x: torch.Tensor) -> torch.Tensor:
        """Return only policy logits [B, POLICY_SIZE] for compatibility."""
        feat = self._shared_features(x)
        p = self.policy_head(feat)
        return p.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self._shared_features(x)
        feat = self.trunk_dropout(feat)
        # Policy
        p = self.policy_head(feat).flatten(start_dim=1)
        # Value
        v = self.value_head(feat)  # [B, 32, 1, 1]
        v = torch.flatten(v, start_dim=1)  # [B, 32]
        v = self.value_dropout(v)
        v = self.value_fc(v)  # [B, 1]
        v = self.value_act(v)  # [-1, 1]
        return p, v


def make_model(policy_only: bool = False, **kwargs) -> nn.Module:
    """Factory returning a policy-only or policy+value model.

    Args:
        policy_only: if True, returns PolicyOnlyResNet; otherwise PolicyValueResNet.
        **kwargs: forwarded to the chosen model class.
    """
    if policy_only:
        return PolicyOnlyResNet(**kwargs)
    return PolicyValueResNet(**kwargs)


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def quick_shape_test():
    """Quick shape check for the PolicyOnlyResNet."""
    m = PolicyOnlyResNet()
    x = torch.zeros((2, m.in_channels, 8, 8))
    y = m(x)
    print("input:", tuple(x.shape), "output:", tuple(y.shape), "params:", count_params(m))
