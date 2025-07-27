import torch.autograd as autograd

# gradient penalty
def gradient_penalty(D, real_data, fake_data, device="cuda"):
    B = real_data.size(0)
    alpha = torch.rand(B,1,1,1,device=device)
  
    interpolated = alpha * real_data + (1-alpha)*fake_data
    interpolated.requires_grad_(True)

    prob_interpolated = D(interpolated)
    gradients = autograd.grad(
        outputs=prob_interpolated,
      
        inputs=interpolated,
        grad_outputs=torch.ones_like(prob_interpolated),
        create_graph=True,
      
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(B,-1)
    gp = ((gradients.norm(2, dim=1) -1) ** 2).mean()
    return gp
