import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

z_dim = 128
G = Generator(z_dim=z_dim).to(device)
D = Discriminator().to(device)

g_opt = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
d_opt = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))

def train_gan(epochs=50, critic_iters=5, lambda_gp=10):
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for real_imgs, _ in loop:
            real_imgs = real_imgs.to(device)
            B = real_imgs.size(0)

            # training the discriminator
            for _ in range(critic_iters):
                z = torch.randn(B, z_dim).to(device)
                fake_imgs = G(z).detach()
                real_scores = D(real_imgs)
                fake_scores = D(fake_imgs)
                gp = gradient_penalty(D, real_imgs, fake_imgs, device)
                d_loss = fake_scores.mean() - real_scores.mean() + lambda_gp * gp

                D.zero_grad()
                d_loss.backward()
                d_opt.step()

            # training the generator
            z = torch.randn(B, z_dim).to(device)
            fake_imgs = G(z)
            g_loss = -D(fake_imgs).mean()

            G.zero_grad()
            g_loss.backward()
            g_opt.step()

            loop.set_postfix({
                "D_loss": d_loss.item(),
                "G_loss": g_loss.item()
            })
