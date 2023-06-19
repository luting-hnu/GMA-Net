import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads  # 8*64 = 512
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)  # 将dim维度映射到inner_dim维度*3 (1,65,1024)->(1,65,1563)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # 对tensor张量分块×:1 197 1024  qkv最后是一个元组, tuple，长度是3，每个元素形状:1 197 1024
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)  # 分头 分到多个子空间

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # q乘k的转置 除以 根号 dk

        attn = self.attend(dots)  # Softmax

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')  # 形状变换
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):  # (dim=1024, depth=6, heads=16, dim_head=64, mlp_dim=2048, dropout=0.1)
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),  # 多头注意力部分  到底是先执行Norm，还是先执行多头注意力？
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))                            # MLP 前馈神经网络部分
            ]))
    def forward(self, x):  # (1,65,1024)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'mean', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)  # image_height=256, image_width=256
        patch_height, patch_width = pair(patch_size)  # patch_height=32, patch_width=32

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)  # 64
        patch_dim = channels * patch_height * patch_width  # 3072 = 3 * 32 * 32  即展平之后的维度
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )
        """------------将self.to_patch_embedding分成两个步骤开始---------------"""
        self.to_patch_embedding1 = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)
        self.to_patch_embedding2 = nn.Linear(patch_dim, dim)
        """------------将self.to_patch_embedding分成两个步骤结束---------------"""

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 位置编码，输入维度是patch数+1
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 随机生成cls token的初始化参数
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)  # (dim=1024, depth=6, heads=16, dim_head=64, mlp_dim=2048, dropout=0.1)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),  # 在batch上应用层归一化    channel方向做归一化，算CHW的均值
            nn.Linear(dim, num_classes)  # 全连接层进行分类
        )

    def forward(self, img):  # (1,3,256,256)
        # x = self.to_patch_embedding(img)  # (1,64,1024)  # 将self.to_patch_embedding分成两个步骤
        x = self.to_patch_embedding1(img)  # (1,64,3072)
        x = self.to_patch_embedding2(x)    # (1,64,1024)

        b, n, _ = x.shape  # b:1 n:64 _:1024

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)  # (1,1,1024)  # 将 cls token复制batch份
        x = torch.cat((cls_tokens, x), dim=1)  # (1,65,1024)  # 将token embedding和patch embedding进行拼接
        x += self.pos_embedding[:, :(n + 1)]  # (1,65,1024)   # 拼接之后 每个token（包括cls token）加上对应位置编码
        x = self.dropout(x)  # (1,65,1024)

        x = self.transformer(x)  # (1,65,1024)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]  # (1,1024)

        x = self.to_latent(x)  # (1,1024)
        return self.mlp_head(x)


if __name__ == '__main__':

    # v = ViT(
    #     image_size=30,
    #     patch_size=3,
    #     num_classes=9,
    #     dim=1024,
    #     depth=6,
    #     heads=16,
    #     mlp_dim=2048,
    #     pool='cls',
    #     channels=103,
    #     dim_head=64,
    #     dropout=0.1,
    #     emb_dropout=0.1
    # )
    #
    # img = torch.randn(64, 103, 30, 30)
    #
    # preds = v(img)  # (1, 1000)
    # print(preds.shape)
    # print("Finish")
    v = ViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )

    img = torch.randn(1, 3, 256, 256)

    preds = v(img)  # (1, 1000)
    print(preds.shape)
    print("finish")

