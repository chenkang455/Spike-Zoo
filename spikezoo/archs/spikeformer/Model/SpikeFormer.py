'''
Ref: To implenment SpikeFormer, we referred to the code of ”segefomer-pytorch” published on github
     (link: https://github.com/lucidrains/segformer-pytorch.git)
'''
from math import sqrt
from functools import partial
import torch
from torch import nn, einsum
from einops import rearrange, reduce

# helpers

def exists(val):
    return val is not None

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

LayerNorm = partial(nn.InstanceNorm2d, affine = True)

# classes

class DsConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias),
            nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        # return self.fn(x)
        return self.fn(self.norm(x))

class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        reduction_ratio
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride = reduction_ratio, bias = False)
        self.to_out = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.GELU()
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        return self.to_out(out)

class MixFeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        expansion_factor
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            DsConv2d(hidden_dim, hidden_dim, 3, padding = 1),
            # nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)

class MiT(nn.Module):
    def __init__(
        self,
        *,
        channels,
        dims,
        heads,
        ff_expansion,
        reduction_ratio,
        num_layers
    ):
        super().__init__()
        stage_kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))
        channels_mod = channels/16
        dims = (channels_mod, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.stages = nn.ModuleList([])

        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio in zip(dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):
            get_overlap_patches = nn.Unfold(kernel, stride = stride, padding = padding)
            # if count == 0:
                # overlap_patch_embed = nn.Conv2d(int((dim_in * kernel ** 2)/16), dim_out, 1)
            # else:
            overlap_patch_embed = nn.Conv2d(int(dim_in * kernel ** 2), dim_out, 1)
            # count+=1

            layers = nn.ModuleList([])

            for _ in range(num_layers):
                layers.append(nn.ModuleList([
                    PreNorm(dim_out, EfficientSelfAttention(dim = dim_out, heads = heads, reduction_ratio = reduction_ratio)),
                    PreNorm(dim_out, MixFeedForward(dim = dim_out, expansion_factor = ff_expansion)),
                ]))

            self.stages.append(nn.ModuleList([
                get_overlap_patches,
                overlap_patch_embed,
                layers
            ]))

    def forward(
        self,
        x,
        return_layer_outputs = False
    ):
        h, w = x.shape[-2:]
        # h = int(h/4)
        # w = int(w/4)
        # print(h)

        layer_outputs = []
        for (get_overlap_patches, overlap_embed, layers) in self.stages:
            x = get_overlap_patches(x)
            # print('aaa')
            # print(x.shape)
            num_patches = int(x.shape[-1])


            # num_patches = int(x.shape[-1]/16)
            # print(num_patches)
            ratio = int(sqrt((h * w) / num_patches))
            # print(ratio)
            x = rearrange(x, 'b c (h w) -> b c h w', h = h // ratio)
            # print(x.shape)
            x = x.type(torch.cuda.FloatTensor)

            x = overlap_embed(x)
            for (attn, ff) in layers:
                x = attn(x) + x
                x = ff(x) + x

            layer_outputs.append(x)

        ret = x if not return_layer_outputs else layer_outputs
        return ret

class SpikeFormer(nn.Module):
    def __init__(
        self,
        inputDim=64,
        dims = (32, 64, 160, 256),
        heads = (1, 2, 5, 8),
        ff_expansion = (8, 8, 4, 4),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = 2,
        channels =64,
        decoder_dim = 256,
        out_channel = 1
    ):
        super().__init__()
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.mit = MiT(
            channels = channels,
            dims = dims,
            heads = heads,
            ff_expansion = ff_expansion,
            reduction_ratio = reduction_ratio,
            num_layers = num_layers
        )
        self.channel_transform = nn.Sequential(
            nn.Conv2d(inputDim, 64, 3, 1, 1),
            nn.GELU()
        )

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),
            nn.PixelShuffle(2 ** i),
            nn.GELU(),
        ) for i, dim in enumerate(dims)])

        self.to_restore = nn.Sequential(
            nn.Conv2d(256+64+16+4, decoder_dim, 1),
            nn.GELU(),
            nn.Conv2d(decoder_dim, out_channel, 1),
        )
        self.fournew = nn.PixelShuffle(4)



    def forward(self, x):
        x = self.channel_transform(x)
        x = self.fournew(x)
        layer_outputs = self.mit(x, return_layer_outputs = True)

        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]

        fused = torch.cat(fused, dim = 1)

        return self.to_restore(fused)
