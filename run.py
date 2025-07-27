import torch
# from model import Generator, Discriminator
# from train import train_gan

z_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G=Generator(z_dim=z_dim).to(device)
D=Discriminator().to(device)
g_opt=torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
d_opt=torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))
print("Starting GAN Training...!!!")

train_gan(epochs=100)
