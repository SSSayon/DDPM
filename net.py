import torch
import torch.nn as nn

# reference: https://github.com/TeaPearce/Conditional_Diffusion_MNIST/blob/main/script.py
class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool=False, device='cuda'
    ) -> None:
        super(ResidualConvBlock, self).__init__()
        '''
        stndard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, device=device),
            nn.BatchNorm2d(out_channels, device=device),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, device=device),
            nn.BatchNorm2d(out_channels, device=device),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414213562
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels, device='cuda') -> None:
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        2x downscale
        '''
        layers = [
            ResidualConvBlock(in_channels, out_channels, device=device),
            nn.MaxPool2d(2),
        ]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x) -> torch.Tensor:
        return self.model(x)

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels, device='cuda'):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        2x upscale
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2, device=device),
            ResidualConvBlock(out_channels, out_channels, device=device),
            ResidualConvBlock(out_channels, out_channels, device=device),
        ]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, skip) -> torch.Tensor:
        x = torch.cat((x, skip), dim=1)
        x = self.model(x)
        return x

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim, device='cuda') -> None:
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim, device=device),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim, device=device),
        ]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x) -> torch.Tensor:
        x = x.view(-1, self.input_dim)
        return self.model(x)

class DDPM(nn.Module):
    def __init__(self, in_channels, n_feat=128) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        divide_groups = 8
        assert n_feat % divide_groups == 0, "n_feat should be divided by {}".format(divide_groups)
        super(DDPM, self).__init__()
        self.device = device
        self.in_channels = in_channels
        self.n_feat = n_feat
        
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True, device=device)
        
        self.down1 = UnetDown(n_feat, n_feat, device=device)
        self.down2 = UnetDown(n_feat, 2*n_feat, device=device)
        
        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        
        self.timeembed1 = EmbedFC(1, 2*n_feat, device=device)
        self.timeembed2 = EmbedFC(1, 1*n_feat, device=device)
        
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2*n_feat, 2*n_feat, 7, 7, device=device),
            nn.GroupNorm(divide_groups, 2*n_feat, device=device),
            nn.ReLU(),
        )
        self.up1 = UnetUp(4*n_feat, n_feat, device=device)
        self.up2 = UnetUp(2*n_feat, n_feat, device=device)
        
        self.out = nn.Sequential(
            nn.Conv2d(2*n_feat, n_feat, 3, 1, 1, device=device),
            nn.GroupNorm(divide_groups, n_feat, device=device),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1, device=device),
        )
    
    def forward(self, x: torch.Tensor, t: float) -> torch.Tensor:
        t = torch.Tensor(t).to(self.device).float()
        # x: [batch_size, 1, 28, 28]
        
        x = self.init_conv(x)
        # x: [batch_size, n_feat, 28, 28]
        
        down1 = self.down1(x)
        # down1: [batch_size, n_feat, 14, 14]
        
        down2 = self.down2(down1)
        # down2: [batch_size, 2*n_feat, 7, 7]
        
        hiddenvec = self.to_vec(down2)
        # hiddenvec: [batch_size, 2*n_feat, 1, 1]
        
        # embed time step
        temb1 = self.timeembed1(t).view(-1, 2*self.n_feat, 1, 1)
        # temb1: [batch_size, 2*n_feat, 1, 1]
        
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        # temb1: [batch_size, n_feat, 1, 1]
        
        up1 = self.up0(hiddenvec)
        # up1: [batch_size, 2*n_feat, 7, 7]
        
        up2 = self.up1(up1 + temb1, down2)
        # up2: [batch_size, n_feat, 14, 14]
        
        up3 = self.up2(up2 + temb2, down1)
        # up3: [batch_size, n_feat, 28, 28]
        
        out = self.out(torch.cat((up3, x), dim=1))
        # out: [batch_size, 1, 28, 28]

        return out

class Conditional_DDPM(nn.Module):
    pass
