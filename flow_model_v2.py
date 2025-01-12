import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint

# ----------------------------------------------------------------------
# 1) Sinusoidal Embeddings for time, same as your diffusion code
# ----------------------------------------------------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        time: (B,) in [0,1]
        returns: (B, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

# ----------------------------------------------------------------------
# 2) FlowNN: the neural network for our conditional OT velocity field
# ----------------------------------------------------------------------
class FlowNN(nn.Module):
    """
    v_theta(z, t, cond):
      z:    (B, latent_dim)
      t:    (B,) in [0,1]
      cond: (B, n_cond)
    outputs velocity in R^(latent_dim)
    """
    def __init__(self,
                 latent_dim,
                 n_cond,
                 cond_dim,
                 hidden_dim=128,
                 time_emb_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim   = cond_dim

        # MLP: condition embedding from (n_cond-> cond_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(n_cond, cond_dim),
            nn.ReLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        # time embeddings
        self.time_emb = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # main velocity MLP: input = (z + c_feat + t_feat)
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
        t:    (B,)
        cond: (B, n_cond)
        returns: velocity in R^(latent_dim)
        """
        

        # embed condition
        c_feat = self.cond_mlp(cond)  # (B, cond_dim)

        # embed time
        t_embed = self.time_emb(t)                 # (B, time_emb_dim)
        t_feat  = self.time_mlp(t_embed).squeeze(1)# (B, hidden_dim)

        # print("z.shape:", z.shape)
        # print("t_feat.shape:", t_feat.shape)
        # print("c_feat.shape:", c_feat.shape)
        
        # concat
        x = torch.cat([z, c_feat, t_feat], dim=1)
        return self.mlp(x)  # velocity in R^(latent_dim)
# ----------------------------------------------------------------------
# 3) OTFlowMatching: implements psi_t for an OT line-based approach
#    plus a .loss(...) method 
# ----------------------------------------------------------------------
class OTFlowMatching:
    """
    We implement the line-based conditional flow:
      psi_t(z0, z1, t) = (1 - (1 - sig_min)*t)*z0 + t*z1
    derivative: d/dt psi_t = z1 - (1 - sig_min)*z0
    """
    def __init__(self, sig_min=0.001):
        self.sig_min = sig_min
        self.eps = 1e-5

    def psi_t(self, z0, z1, t):
        """
        z0, z1: (B, latent_dim)
        t:      (B, latent_dim) or (B,1) broadcast
        returns: (B, latent_dim)
        """
        # (1 - (1-sig_min)*t)*z0 + t*z1
        return (1 - (1 - self.sig_min)*t) * z0 + t * z1

    def loss(self, vf_model: nn.Module, z1: torch.Tensor, cond: torch.Tensor):
        """
        We'll sample z0 from N(0,I), pick random t in [0,1],
        define z_t = psi_t(z0,z1), and match v_theta to d/dt(psi_t).
        """
        device = z1.device
        B, dim = z1.shape

        # sample random t in [0,1], with a tiny offset approach
        # to ensure we don't pick exactly t=1
        t_rand = (torch.rand(1, device=device)
                + torch.arange(B, device=device)/B) % (1 - self.eps)
        # t_rand.shape: (B,)

        # for broadcasting inside psi_t, we make t_col = shape (B,1)
        t_col = t_rand.view(B, 1)

        # sample z0 from Gaussian
        z0 = torch.randn_like(z1)
        

        # define z(t)
        z_t = self.psi_t(z0, z1, t_col)

        # predicted velocity
        # cond is (B, n_cond)
        v_pred = vf_model(z_t, t_rand, cond)

        # true velocity: d/dt psi_t
        # derivative: z1 - (1-sig_min)*z0
        d_psi = z1 - (1 - self.sig_min)*z0

        # loss
        return F.mse_loss(v_pred, d_psi)

# ----------------------------------------------------------------------
# 4) CondVF: a wrapper that can do ODE-based "decode" from t=0->1
#    or "encode" from t=1->0
# ----------------------------------------------------------------------
class CondVF(nn.Module):
    """
    This class takes a velocity net (FlowNN) and
    does an ODE solve from t=0->1 (decode) or t=1->0 (encode).
    We'll do a 'decode' method for sampling new latents.
    """
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net  # e.g. an instance of FlowNN

    def forward(self, z, t, cond):
        """
        We unify the signature to mimic the notebook:
        t: scalar or (B,)
        z: (B,latent_dim)
        cond: (B,n_cond)
        return: velocity in R^(latent_dim)
        """
        return self.net(z, t, cond)

    def odefunc(self, t_scalar, z_flat, cond):
        """
        This function is used by odeint. 
        - z_flat: shape (B*latent_dim,)
        - we unflatten to (B,latent_dim)
        - broadcast t_scalar -> (B,) 
        - cond is (B, n_cond)
        - output velocity, flattened
        """
        B_latent = z_flat.shape[0]
        # shape check: we assume B_latent = B*latent_dim
        # so we guess B = cond.shape[0]
        B = cond.shape[0]
        latent_dim = B_latent // B

        z_current = z_flat.view(B, latent_dim)
        t_batch   = z_current.new_full((B,), t_scalar)
        v = self.net(z_current, t_batch, cond)  # (B, latent_dim)
        return v.view(-1)  # flattened

    @torch.no_grad()
    def decode(self, cond, batch_size, latent_dim, n_steps=50, method='rk4'):
        """
        Integrate from t=0 to t=1. 
        1) sample z0 ~ N(0,I)
        2) solve dz/dt = v_\theta(z,t)
        3) return z(1)
        """
        device = next(self.net.parameters()).device
        z0 = torch.randn(batch_size, latent_dim, device=device)
        z0_flat = z0.view(-1)

        t_span = torch.linspace(0., 1., n_steps, device=device)

        # define a partial to pass cond
        def func(t_s, z_flat):
            return self.odefunc(t_s, z_flat, cond)

        z_traj = odeint(func, z0_flat, t_span, method=method)
        z1_flat = z_traj[-1]  # shape (B*latent_dim,)
        return z1_flat.view(batch_size, latent_dim)

    @torch.no_grad()
    def encode(self, z1, cond, n_steps=50, method='rk4'):
        """
        Integrate from t=1 -> 0. 
        If you want 'z0' ~ N(0,I)', you do the inverse ODE.
        """
        device = next(self.net.parameters()).device
        B, latent_dim = z1.shape
        z1_flat = z1.view(-1)
        t_span = torch.linspace(1., 0., n_steps, device=device)

        def func(t_s, z_flat):
            return self.odefunc(t_s, z_flat, cond)

        z_traj = odeint(func, z1_flat, t_span, method=method)
        z0_flat = z_traj[-1]
        return z0_flat.view(B, latent_dim)
