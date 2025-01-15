import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# A helper for time embeddings, just like in diffusion code
# ---------------------------------------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        time: shape (B,) in [0,1], for instance
        returns: shape (B, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

# ---------------------------------------------------------
# FlowNN: velocity network v_theta(z, t, cond)
# ---------------------------------------------------------
class FlowNN(nn.Module):
    def __init__(self, latent_dim, n_cond,cond_dim, hidden_dim=128, time_emb_dim=64):
        """
        latent_dim: dimension of z
        cond_dim:   dimension of the condition vector
        hidden_dim: MLP hidden dimension
        time_emb_dim: dimension used for sinusoidal time embedding
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim   = cond_dim

        # Project the condition to d_cond
        self.cond_mlp = nn.Sequential(
            nn.Linear(n_cond, cond_dim),
            nn.ReLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Time embeddings
        self.time_emb = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Velocity MLP:
        # Input dimension: z + cond + time
        # Output dimension: velocity in R^latent_dim
        in_dim = latent_dim + cond_dim + hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z, t, cond):
        """
        z:    (B, latent_dim)
        t:    (B,) in [0,1]
        cond: (B, cond_dim)  (e.g., graph stats )
        returns: velocity of shape (B, latent_dim)
        """

        # process condition
        c_feat = self.cond_mlp(cond)            # (B, hidden_dim)

        # process time
        t_embed = self.time_emb(t)              # (B, time_emb_dim)
        t_feat  = self.time_mlp(t_embed)        # (B, hidden_dim)

        # concat all features
        x = torch.cat([z, c_feat, t_feat], dim=1)  # (B, latent_dim + hidden_dim + hidden_dim)

        # produce velocity
        v = self.mlp(x)  # (B, latent_dim)
        return v

# ---------------------------------------------------------
# Flow Matching Loss
# ---------------------------------------------------------
def flow_loss(flow_model, z0, z1, cond, loss_type="l2"):
    """
    z0:   (B, latent_dim)  random sample from N(0,I) base
    z1:   (B, latent_dim)  target latents from encoder
    cond: (B, cond_dim)
    returns: the flow matching loss for a random t in [0,1]
    """
    device = z0.device
    B = z0.size(0)

    # 1) sample time t in [0,1]
    t = torch.rand(B, device=device)

    # 2) linearly interpolate z(t) = z0 + t*(z1 - z0)
    z_t = z0 + t.unsqueeze(1) * (z1 - z0)  # shape (B, latent_dim)

    # 3) target velocity = d/dt z(t) = z1 - z0
    v_target = (z1 - z0)  # shape (B, latent_dim)

    # 4) predicted velocity
    v_pred = flow_model(z_t, t, cond)

    # 5) loss
    if loss_type == "l2":
        return F.mse_loss(v_pred, v_target)
    elif loss_type == "l1":
        return F.l1_loss(v_pred, v_target)
    elif loss_type == "huber":
        return F.smooth_l1_loss(v_pred, v_target)
    else:
        raise NotImplementedError(f"Unknown loss_type={loss_type}")

# ---------------------------------------------------------
# Sampling: integrate from t=0 to t=1
# ---------------------------------------------------------
@torch.no_grad()
def sample_flow(flow_model, cond, batch_size, latent_dim, n_steps=10):
    """
    Conditionally sample from the flow by numerically integrating
    dz/dt = v_teta(z, t, cond).

    For simplicity, we do a naive Euler integrator with n_steps.
    """
    device = next(flow_model.parameters()).device

    # 1) sample z(0) from N(0,I)
    z = torch.randn(batch_size, latent_dim, device=device)

    # 2) naive Euler integration in t
    #    step size = 1/n_steps
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.ones(batch_size, device=device) * (i * dt)  # current time
        v = flow_model(z, t, cond)  # velocity
        z = z + dt * v              # Euler step

    # now z is an approximation of z(1)
    return z
