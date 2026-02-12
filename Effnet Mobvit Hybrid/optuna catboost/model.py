import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper function for 1x1 convolution
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.Hardswish()
    )

# Helper function for 3x3 convolution
def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.Hardswish()
    )

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: t.reshape(t.shape[0], t.shape[1], self.heads, t.shape[2] // self.heads).transpose(1, 2), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(out.shape[0], out.shape[1], self.inner_dim)
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x

class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernal_size, patch_size, mlp_dim, dropout = 0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernal_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = nn.Sequential(*[TransformerBlock(dim, 4, 8, mlp_dim, dropout) for _ in range(depth)])

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernal_size)
    
    def forward(self, x):
        y = x.clone()

        # Local global feature interaction
        x = self.conv1(x)
        x = self.conv2(x)
        
        shape = x.shape # (B, dim, H_feat, W_feat)
        B, D, H, W = shape

        # Calculate padding
        pad_h = (self.ph - (H % self.ph)) % self.ph
        pad_w = (self.pw - (W % self.pw)) % self.pw

        # Apply padding if necessary
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
            # Update H and W after padding
            H = H + pad_h
            W = W + pad_w

        h_split, w_split = H // self.ph, W // self.pw

        # Reshape to (B * h_split * w_split, ph * pw, dim)
        x = x.reshape(B, D, h_split, self.ph, w_split, self.pw)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous() # (B, h_split, w_split, ph, pw, dim)
        x = x.view(B * h_split * w_split, self.ph * self.pw, D) # (N_patches_total, pixels_per_patch, dim)

        x = self.transformer(x)

        # Reshape back to (B, dim, H_feat, W_feat)
        x = x.view(B, h_split, w_split, self.ph, self.pw, D)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous() # (B, dim, h_split, ph, w_split, pw)
        x = x.view(B, D, H, W)

        # Remove padding if it was applied
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :shape[2], :shape[3]] # slice back to original H, W before padding

        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x

class MobileViT(nn.Module):
    def __init__(self, image_size, num_classes, dims, channels, depths, patch_size=2, mlp_dim_ratio=2):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        
        ih, iw = image_size, image_size # Assuming square images
        ph, pw = patch_size, patch_size

        self.conv1 = conv_nxn_bn(3, channels[0], kernal_size=3, stride=2)

        self.stem = nn.ModuleList([])
        in_channel = channels[0]

        # MV2 blocks
        self.stem.append(MobileNetV2Block(in_channel, channels[1], stride=1, expansion=4))
        in_channel = channels[1]
        self.stem.append(MobileNetV2Block(in_channel, channels[2], stride=2, expansion=4))
        in_channel = channels[2]
        self.stem.append(MobileNetV2Block(in_channel, channels[3], stride=1, expansion=4))
        in_channel = channels[3]
        self.stem.append(MobileNetV2Block(in_channel, channels[4], stride=2, expansion=4))
        in_channel = channels[4] # Output channel after stem

        # MobileViT blocks
        self.mobilevit_blocks = nn.ModuleList([])
        for i in range(len(dims)):
            mlp_dim = dims[i] * mlp_dim_ratio
            self.mobilevit_blocks.append(MobileViTBlock(dims[i], depths[i], in_channel, 3, (ph, pw), mlp_dim))
            
            # Update in_channel for the next block
            # This assumes that the MobileViTBlock's output channel is the 'channel' parameter passed to it.
            # And the MobileViTBlock's conv4 outputs 'channel' number of channels.
            in_channel = in_channel # The MobileViTBlock maintains the 'channel' size

            if i < len(dims) - 1:
                next_channel_idx = i + 5 
                self.mobilevit_blocks.append(MobileNetV2Block(in_channel, channels[next_channel_idx], stride=2, expansion=4))
                in_channel = channels[next_channel_idx] 

        self.conv2 = conv_1x1_bn(in_channel, dims[-1])

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.stem:
            x = block(x)
        
        for block in self.mobilevit_blocks:
            x = block(x)

        x = self.conv2(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x


# For MobileNetV2Block (used in MobileViT stem and downsampling)
class MobileNetV2Block(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expansion))
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

if __name__ == '__main__':
    img_size = 224
    num_classes = 8  # Based on the 8 categories in Kvasir-V2 dataset
    
    # Example MobileViT-XXS configuration from the original paper
    dims = [96, 192, 384]  # Dimension of the transformer blocks
    channels = [16, 32, 64, 128, 160, 192, 256, 320, 384, 512] # Channels for conv layers
    depths = [2, 4, 3] # Number of transformer blocks in each MobileViT block

    model = MobileViT(image_size=img_size, num_classes=num_classes, dims=dims, channels=channels, depths=depths)
    
    # Test with a dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size) # Batch size 1, 3 channels, 224x224 image
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")

    # Verify the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")
