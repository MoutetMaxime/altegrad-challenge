import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool,GATConv

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes, n_cond=7):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        mlp_layers = [nn.Linear(latent_dim + n_cond, hidden_dim)] + [
            nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers - 2)
        ]
        mlp_layers.append(nn.Linear(hidden_dim, 2 * n_nodes * (n_nodes - 1) // 2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, data):
        stats = data.stats
        x = torch.cat((x, stats), axis=1)

        for i in range(self.n_layers - 1):
            x = self.relu(self.mlp[i](x))

        x = self.mlp[self.n_layers - 1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=1, hard=True)[:, :, 0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:, idx[0], idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        return adj
    
class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            ))                        
        for layer in range(n_layers-1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            )) 

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out


# ----------------------------------------------------------------------
# GAT-based Encoder (new)
# ----------------------------------------------------------------------
class GAT(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        n_layers,
        dropout=0.2,
        heads=4
    ):
        """
        A multi-layer GAT encoder that outputs a hidden representation,
        from which we'll derive mu/logvar in the VAE class.
        """
        super().__init__()
        self.dropout = dropout
        self.n_layers = n_layers

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # First GAT layer
        self.convs.append(
            GATConv(
                in_channels=input_dim,
                out_channels=hidden_dim,
                heads=heads,
                concat=False  # output dim = hidden_dim (not hidden_dim * heads)
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Subsequent GAT layers
        for _ in range(n_layers - 1):
            self.convs.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    concat=False
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Final BatchNorm -> We'll apply a linear layer for mu/logvar outside
        self.bn_final = nn.BatchNorm1d(hidden_dim)

    def forward(self, data):
        # Node features, edges, and batch
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Stack of GAT layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = F.leaky_relu(x, negative_slope=0.2)
            x = bn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling to get graph-level embedding
        x = global_add_pool(x, batch)
        x = self.bn_final(x)
        return x


# ----------------------------------------------------------------------
# Variational AutoEncoder (only the encoder changed to GAT)
# ----------------------------------------------------------------------
# class VariationalAutoEncoder(nn.Module):
#     def __init__(
#         self,
#         input_dim,
#         hidden_dim_enc,
#         hidden_dim_dec,
#         latent_dim,
#         n_layers_enc,
#         n_layers_dec,
#         n_max_nodes
#     ):
#         super(VariationalAutoEncoder, self).__init__()
#         self.n_max_nodes = n_max_nodes
#         self.input_dim = input_dim

#         # Encoder replaced with GAT
#         self.encoder = GAT(
#             input_dim=input_dim,
#             hidden_dim=hidden_dim_enc,
#             latent_dim=latent_dim,  # Not used directly in GAT, but we keep for signature
#             n_layers=n_layers_enc,
#             dropout=0.2,
#             heads=4
#         )

#         # Two separate heads for mu / logvar
#         self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
#         self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)

#         # Same decoder as before
#         self.decoder = Decoder(
#             latent_dim=latent_dim,
#             hidden_dim=hidden_dim_dec,
#             n_layers=n_layers_dec,
#             n_nodes=n_max_nodes
#         )

#     def forward(self, data):
#         # Encode
#         x_enc = self.encoder(data)

#         mu = self.fc_mu(x_enc)
#         logvar = self.fc_logvar(x_enc)

#         # Reparameterize -> latent code
#         z = self.reparameterize(mu, logvar)
#         # Decode adjacency
#         adj = self.decoder(z)
#         return adj

#     def encode(self, data):
#         # For external usage (e.g. in diffusion)
#         x_enc = self.encoder(data)
#         mu = self.fc_mu(x_enc)
#         logvar = self.fc_logvar(x_enc)
#         z = self.reparameterize(mu, logvar)
#         return z

#     def reparameterize(self, mu, logvar, eps_scale=1.0):
#         # Sample z from N(mu, sigma)
#         if self.training:
#             std = (0.5 * logvar).exp_()
#             eps = torch.randn_like(std) * eps_scale
#             return eps.mul(std).add_(mu)
#         else:
#             return mu

#     def decode(self, mu, logvar):
#         # If you want to decode from mu & logvar directly
#         z = self.reparameterize(mu, logvar)
#         return self.decoder(z)

#     def decode_mu(self, mu):
#         # Optionally decode directly from mu
#         return self.decoder(mu)

#     def loss_function(self, data, beta=0.05):
#         # 1) Encode -> mu, logvar
#         x_enc = self.encoder(data)
#         mu = self.fc_mu(x_enc)
#         logvar = self.fc_logvar(x_enc)

#         # 2) Sample latent
#         z = self.reparameterize(mu, logvar)

#         # 3) Decode adjacency
#         adj = self.decoder(z)

#         # Reconstruction loss (L1)
#         recon = F.l1_loss(adj, data.A, reduction='mean')

#         # KL divergence
#         kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#         # Weighted total loss
#         loss = recon + beta * kld
#         return loss, recon, kld

#Variational Autoencoder
# ----------------------------------------------------------------------
# Variational AutoEncoder (more weight on the recon in the loss and the concat of data.stats in the decoder)
# ----------------------------------------------------------------------
class VariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim_enc,
        hidden_dim_dec,
        latent_dim,
        n_layers_enc,
        n_layers_dec,
        n_max_nodes,
    ):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.encoder = GIN(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)

    def forward(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g, data)
        return adj

    def encode(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        return x_g

    def reparameterize(self, mu, logvar, eps_scale=1.0):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu, logvar, data):
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g, data)
        return adj

    def decode_mu(self, mu, data):
        adj = self.decoder(mu, data)
        return adj

    def loss_function(self, data, beta):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g, data)

        recon = F.l1_loss(adj, data.A, reduction="mean")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta * kld

        return loss, recon, kld
