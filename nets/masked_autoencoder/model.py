import torch as t
import numpy as np
from torch import nn
from timm.models.vision_transformer import Block, PatchEmbed


""" Split this in half so half of it is sin, the other side is cos """


def create_sin_cos_pos_embed(num_patches, embed_dim):
    half_dim = embed_dim // 2
    dim_range = np.arange(half_dim)
    omega = 1 / 10000**(dim_range / half_dim)
    half_patch = np.arange(num_patches)

    domain = np.einsum('l,m -> lm', half_patch, omega)

    sin_embed = np.sin(domain)
    cos_embed = np.cos(domain)

    embed = np.concatenate([sin_embed, cos_embed], axis=1)

    return embed


class MaskedAutoEncoder(nn.Module):
    def __init__(self,
                 img_size=224,
                 mask_ratio=0.75,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=512,
                 encoder_depth=4,
                 decoder_embed_dim=512,
                 decoder_depth=4,
                 num_heads=16,
                 device="cpu"
                 ) -> None:
        super().__init__()

        self.device = device
        self.num_patches = int(img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio

        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim=embed_dim)

        self.pos_embed = nn.Parameter(
            t.zeros(1, self.num_patches, embed_dim), requires_grad=False)

        self.encoder_blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, qkv_bias=True) for _ in range(encoder_depth)])

        self.decoder_blocks = nn.ModuleList(
            [Block(decoder_embed_dim, num_heads, qkv_bias=True)
             for _ in range(decoder_depth)]
        )

        self.init_weights()

    def init_weights(self):
        self.pos_embed.data.copy_(t.from_numpy(create_sin_cos_pos_embed(
            self.num_patches, self.embed_dim)))

    def mask_input(self, x):
        N, P, E = x.shape
        num_visible = int(P * (1 - self.mask_ratio))

        t.rand(N, P, device=self.device)
        ranking = t.argsort(t, axis=1)
        un_mask = t.argsort(ranking, axis=1)

        mask = ranking[:, :num_visible]

        mask_x = t.gather(x, 1, mask)

    def encode(self, x):
        x = self.patch_embed(x)
        x += self.pos_embed

    def forward(self, x):
        x = self.patch_embed(x)

        return x
