
#generator for the gan
class Generator(nn.Module):

  
    def __init__(self, z_dim=128, img_channels=3, base_channels=64):
        super().__init__()
        self.init_proj = nn.ConvTranspose2d(z_dim, base_channels * 16, 4, 1, 0)

        self.block1 =ResBlockG(base_channels * 16, base_channels * 8)
        self.block2 = ResBlockG(base_channels * 8, base_channels * 4)

      
        self.attn = SelfAttention(base_channels * 4)
      
        self.block3 = ResBlockG(base_channels * 4, base_channels * 2)
      #nois injection
        self.noise =NoiseInjection(base_channels * 2)
      
        self.block4 = ResBlockG(base_channels * 2, base_channels)
        self.output =nn.Sequential(
          
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
          
            nn.Conv2d(base_channels, img_channels,3, 1,1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.init_proj(z.view(z.size(0), z.size(1), 1,1))
      
        out = self.block1(out)
        out = self.block2(out)
      
        out =self.attn(out)
        out = self.block3(out)
      
        out =self.noise(out)
        out = self.block4(out)
      
        return self.output(out)

#discriminator
class Discriminator(nn.Module):

  
    def __init__(self, img_channels=3, base_channels=64):
        super().__init__()
        self.block1 = ResBlockD(img_channels, base_channels)
        self.block2 = ResBlockD(base_channels, base_channels *2)
      
        self.attn = SelfAttention(base_channels *2)
        self.block3 = ResBlockD(base_channels *2, base_channels *4)
        self.block4 = ResBlockD(base_channels *4, base_channels * 8)
      
        self.stddev = MiniBatchStdDev()
        self.final_conv = spectral_norm(nn.Conv2d(base_channels * 8+ 1,1, 4))

    def forward(self,x):
        out = self.block1(x)
        out = self.block2(out)

      
        out = self.attn(out)
        out = self.block3(out)
      
        out = self.block4(out)
        out = self.stddev(out)
        out = self.final_conv(out)
      
        return out.view(-1)
