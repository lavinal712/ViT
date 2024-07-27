import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViTBlock(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        dropout=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=1000,
        pool="cls",
        dropout=0.1,
    ):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2

        self.patch_embedding = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads) for _ in range(depth)
        ])

        self.pool = pool
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        B, N, D = x.shape
        cls_tokens = self.cls_token.repeat(B, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :(N + 1)]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.mlp_head(x)
        return x
