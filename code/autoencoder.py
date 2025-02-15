import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINConv, global_add_pool

# Cross-Attention Layer
class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2):
        super(CrossAttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Layers pour Q, K, V pour la cross-attention
        self.query_layer = nn.Linear(hidden_dim, hidden_dim)
        self.key_layer = nn.Linear(hidden_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, hidden_dim)

        # Layer de sortie
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

        # Dropout pour régularisation
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, graph_latents, text_latents):
        # Calcul des Q, K, V pour le mécanisme de cross-attention
        Q = self.query_layer(graph_latents)  # Requêtes venant du graphe
        K = self.key_layer(text_latents)  # Clés venant du texte
        V = self.value_layer(text_latents)  # Valeurs venant du texte

        # Calcul des scores d'attention
        attn_scores = (
            torch.matmul(Q, K.transpose(-2, -1)) / self.hidden_dim**0.5
        )  # Attention
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # Calcul des valeurs pondérées par l'attention
        attended_values = torch.matmul(attn_probs, V)

        # Passer les valeurs attentives par la couche de sortie
        output = self.output_layer(attended_values)
        return output
        

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

class AutoRegressiveDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super(AutoRegressiveDecoder, self).__init__()

        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.latent_dim = latent_dim

        # Créer un décodeur sans encodeur
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=4,
            dim_feedforward=hidden_dim,
        )

        self.transformer_decoder = nn.TransformerDecoder(
            self.transformer_decoder_layer, num_layers=n_layers
        )

        # Couches linéaires pour prédire les arêtes
        self.fc_out = nn.Linear(
            latent_dim, self.n_nodes
        )  # Prédire une probabilité d'arête
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x est la représentation latente des nœuds du graphe
        # Initialisation du tensor de "sequences" vide pour le décodeur Transformer
        seq_input = torch.zeros(
            x.size(0), self.n_nodes, self.latent_dim, device=x.device
        )

        # Transformer pour générer les arêtes une par une
        output = self.transformer_decoder(
            tgt=x.unsqueeze(1).repeat(1, self.n_nodes, 1), memory=seq_input
        )  # [batch_size, n_nodes, latent_dim]

        # Prédiction des probabilités d'arêtes pour chaque paire de nœuds
        adj_prob = self.fc_out(output)  # [batch_size, n_nodes, n_nodes]
        # Symétrisation de la matrice d'adjacence
        adj_prob = torch.triu(
            adj_prob, diagonal=1
        )  # Garder que la partie supérieure de la matrice

        # Appliquer la sigmoid pour obtenir des probabilités
        adj_prob = self.sigmoid(adj_prob)

        adj = (adj_prob + adj_prob.transpose(1, 2)) / 2
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
    

# GAT Model
class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, heads=4, dropout=0.2):
        super().__init__()
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        # Première couche GAT
        self.convs.append(GATConv(in_channels=input_dim, 
                                  out_channels=hidden_dim // heads, 
                                  heads=heads, 
                                  concat=True))  # concat=True pour concaténer les têtes d'attention

        # Couches intermédiaires
        for _ in range(n_layers - 1):
            self.convs.append(GATConv(in_channels=hidden_dim, 
                                      out_channels=hidden_dim // heads, 
                                      heads=heads, 
                                      concat=True))
        
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        for conv in self.convs:
            x = conv(x, edge_index)  # Passage dans chaque couche GAT
            x = F.leaky_relu(x, 0.2)  # Fonction d'activation
            x = F.dropout(x, self.dropout, training=self.training)

        # Pooling global
        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out


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

        ### GAT ENCODER
        # self.encoder = GAT(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        ###

        ### GIN ENCODER
        self.encoder = GIN(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        ###

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
