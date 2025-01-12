import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# forward diffusion (using the nice property)
def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


# Loss function for denoising
def p_losses(denoise_model, x_start, t, cond, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=noise)
    predicted_noise = denoise_model(x_noisy, t, cond)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


# Position embeddings
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
# class CrossAttention(nn.Module):
#     """
#     Cross-attention block based on MultiheadAttention:
#     - x is your 'query'
#     - cond is your 'key' and 'value'
#     """
#     def __init__(self, embed_dim, num_heads=4):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)

#     def forward(self, x, context):
#         # x:  (B, Nx, embed_dim)
#         # context: (B, Ny, embed_dim)
#         # We do cross-attention: queries = x, keys=values=context
#         out, _ = self.attn(query=x, key=context, value=context)
#         return out
    
# class DenoiseNN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, n_layers, n_cond, d_cond, num_heads=4):
#         super(DenoiseNN, self).__init__()
#         self.n_layers = n_layers
#         self.n_cond = n_cond

#         # Project the condition to d_cond
#         self.cond_mlp = nn.Sequential(
#             nn.Linear(n_cond, d_cond),
#             nn.ReLU(),
#             nn.Linear(d_cond, hidden_dim),
#         )

#         # # Additional projection so that cond can match 'hidden_dim' for attention
#         # self.cond_proj = nn.Linear(d_cond, hidden_dim)

#         self.time_mlp = nn.Sequential(
#             SinusoidalPositionEmbeddings(hidden_dim),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, hidden_dim),
#         )

#         # Our MLP layers are mostly the same
#         # but we will incorporate cross-attn in between them.
#         mlp_layers = []
#         mlp_layers.append(nn.Linear(input_dim, hidden_dim))
#         for i in range(n_layers-2):
#             mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
#         mlp_layers.append(nn.Linear(hidden_dim, input_dim))
#         self.mlp = nn.ModuleList(mlp_layers)

#         # Batch norm layers
#         bn_layers = [nn.BatchNorm1d(hidden_dim) for _ in range(n_layers-1)]
#         self.bn = nn.ModuleList(bn_layers)

#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()

#         # Cross-attention block
#         self.cross_attn = CrossAttention(embed_dim=hidden_dim, num_heads=num_heads)

#     def forward(self, x, t, cond):
#         """
#         x:   (B, input_dim)    <- e.g. your noisy latent
#         t:   (B,)              <- timesteps
#         cond:(B, n_cond)       <- conditioning stats
#         """
#         # Process condition
#         cond = cond.reshape(-1, self.n_cond)
#         cond = torch.nan_to_num(cond, nan=-100.0)
#         cond_feat = self.cond_mlp(cond)               # shape: (B, d_cond)
#         # cond_feat = self.cond_proj(cond_feat)         # shape: (B, hidden_dim)

#         # Process time embedding
#         t_emb = self.time_mlp(t)  # shape: (B, hidden_dim)

#         # We'll do an MLP pass, injecting cross-attention somewhere in the middle
#         # Start with x -> hidden_dim
#         x_h = self.mlp[0](x)      # shape: (B, hidden_dim)

#         # For each hidden layer except the last one
#         for i in range(self.n_layers - 1):
#             # Add time embedding
#             x_h = x_h + t_emb

#             # BatchNorm expects (B, hidden_dim) but cross-attn needs (B, seq_len, hidden_dim).
#             # We'll do BN first, then cross-attn
#             x_h = self.bn[i](x_h) # shape: (B, hidden_dim)
#             x_h = self.relu(x_h)

#             # Reshape x_h and cond for attention: (B, 1, hidden_dim)
#             x_3d = x_h.unsqueeze(1)
#             cond_3d = cond_feat.unsqueeze(1)

#             # Cross-attend: queries=x_3d, keys=cond_3d
#             # shape out: (B, 1, hidden_dim)
#             x_attn = self.cross_attn(x_3d, cond_3d)

#             # Flatten back to (B, hidden_dim)
#             x_h = x_attn.squeeze(1)

#             # If not at the last hidden -> next MLP
#             if i < (self.n_layers - 2):
#                 x_h = self.mlp[i+1](x_h)

#         # Finally, apply the last linear that outputs the predicted noise
#         # (We already used i up to n_layers-2, so the last layer is index n_layers-1)
#         x_out = self.mlp[self.n_layers - 1](x_h)

#         return x_out



# Denoise model
class DenoiseNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_cond, d_cond):
        super(DenoiseNN, self).__init__()
        self.n_layers = n_layers
        self.n_cond = n_cond
        self.cond_mlp = nn.Sequential(
            nn.Linear(n_cond, d_cond),
            nn.ReLU(),
            nn.Linear(d_cond, d_cond),
        )

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        mlp_layers = [nn.Linear(input_dim+d_cond, hidden_dim)] + [nn.Linear(hidden_dim+d_cond, hidden_dim) for i in range(n_layers-2)]
        mlp_layers.append(nn.Linear(hidden_dim, input_dim))
        self.mlp = nn.ModuleList(mlp_layers)

        bn_layers = [nn.BatchNorm1d(hidden_dim) for i in range(n_layers-1)]
        self.bn = nn.ModuleList(bn_layers)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, t, cond):
        cond = torch.reshape(cond, (-1, self.n_cond))
        cond = torch.nan_to_num(cond, nan=-100.0)
        cond = self.cond_mlp(cond)
        t = self.time_mlp(t)
        for i in range(self.n_layers-1):
            x = torch.cat((x, cond), dim=1)
            x = self.relu(self.mlp[i](x))+t
            x = self.bn[i](x)
        x = self.mlp[self.n_layers-1](x)
        return x


@torch.no_grad()
def p_sample(model, x, t, cond, t_index, betas):
    # define alphas

    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, cond) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, cond, timesteps, betas, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in reversed(range(0, timesteps)):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), cond, i, betas)
        imgs.append(img)
        #imgs.append(img.cpu().numpy())
    return imgs



@torch.no_grad()
def sample(model, cond, latent_dim, timesteps, betas, batch_size):
    return p_sample_loop(model, cond, timesteps, betas, shape=(batch_size, latent_dim))
