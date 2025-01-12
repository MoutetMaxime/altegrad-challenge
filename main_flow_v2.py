import argparse
import ast
import csv
import os
import pickle
import random
import shutil
from datetime import datetime

import networkx as nx
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from autoencoder import VariationalAutoEncoder
from denoise_model import DenoiseNN, p_losses, sample
from torch import Tensor
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from utils import construct_nx_from_adj, linear_beta_schedule, preprocess_dataset

np.random.seed(13)

"""
Parses command line arguments for configuring the NeuralGraphGenerator model. This includes
settings for learning rates, architecture dimensions, training epochs, dropout rates, and 
parameters specific to the autoencoder (VGAE) and diffusion-based denoising model components.

Returns:
    argparse.Namespace: Parsed arguments as attributes for easy configuration of the model.
"""

# Argument parser
parser = argparse.ArgumentParser(description='NeuralGraphGenerator')

# Argument parser for configuring the NeuralGraphGenerator model
parser = argparse.ArgumentParser(description='Configuration for the NeuralGraphGenerator model')

# Learning rate for the optimizer
parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for the optimizer, typically a small float value (default: 0.001)")

# Dropout rate
parser.add_argument('--dropout', type=float, default=0.0, help="Dropout rate (fraction of nodes to drop) to prevent overfitting (default: 0.0)")

# Batch size for training
parser.add_argument('--batch-size', type=int, default=256, help="Batch size for training, controlling the number of samples per gradient update (default: 256)")

# Number of epochs for the autoencoder training
parser.add_argument('--epochs-autoencoder', type=int, default=200, help="Number of training epochs for the autoencoder (default: 200)")

# Hidden dimension size for the encoder network
parser.add_argument('--hidden-dim-encoder', type=int, default=64, help="Hidden dimension size for encoder layers (default: 64)")

# Hidden dimension size for the decoder network
parser.add_argument('--hidden-dim-decoder', type=int, default=256, help="Hidden dimension size for decoder layers (default: 256)")

# Dimensionality of the latent space
parser.add_argument('--latent-dim', type=int, default=64, help="Dimensionality of the latent space in the autoencoder (default: 32)")

# Maximum number of nodes of graphs
parser.add_argument('--n-max-nodes', type=int, default=50, help="Possible maximum number of nodes in graphs (default: 50)")

# Number of layers in the encoder network
parser.add_argument('--n-layers-encoder', type=int, default=2, help="Number of layers in the encoder network (default: 2)")

# Number of layers in the decoder network
parser.add_argument('--n-layers-decoder', type=int, default=3, help="Number of layers in the decoder network (default: 3)")

# Dimensionality of spectral embeddings for graph structure representation
parser.add_argument('--spectral-emb-dim', type=int, default=10, help="Dimensionality of spectral embeddings for representing graph structures (default: 10)")

# Number of training epochs for the denoising model
parser.add_argument('--epochs-denoise', type=int, default=200, help="Number of training epochs for the denoising model (default: 100)")

# Number of timesteps in the diffusion
parser.add_argument('--timesteps', type=int, default=1000, help="Number of timesteps for the diffusion (default: 500)")

# Hidden dimension size for the denoising model
parser.add_argument('--hidden-dim-denoise', type=int, default=512, help="Hidden dimension size for denoising model layers (default: 512)")

# Number of layers in the denoising model
parser.add_argument('--n-layers_denoise', type=int, default=3, help="Number of layers in the denoising model (default: 3)")

# Flag to toggle training of the autoencoder (VGAE)
parser.add_argument('--train-autoencoder', action='store_false', default=True, help="Flag to enable/disable autoencoder (VGAE) training (default: enabled)")

# Flag to toggle training of the diffusion-based denoising model
parser.add_argument('--train-denoiser', action='store_true', default=True, help="Flag to enable/disable denoiser training (default: enabled)")

# Dimensionality of conditioning vectors for conditional generation
parser.add_argument('--dim-condition', type=int, default=128, help="Dimensionality of conditioning vectors for conditional generation (default: 128)")

# Number of conditions used in conditional vector (number of properties)
parser.add_argument('--n-condition', type=int, default=7, help="Number of distinct condition properties used in conditional vector (default: 7)")

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# preprocess train data, validation data and test data. Only once for the first time that you run the code. Then the appropriate .pt files will be saved and loaded.
trainset = preprocess_dataset("train", args.n_max_nodes, args.spectral_emb_dim)
validset = preprocess_dataset("valid", args.n_max_nodes, args.spectral_emb_dim)
testset = preprocess_dataset("test", args.n_max_nodes, args.spectral_emb_dim)



# initialize data loaders
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)


# initialize VGAE model
autoencoder = VariationalAutoEncoder(args.spectral_emb_dim+1, args.hidden_dim_encoder, args.hidden_dim_decoder, args.latent_dim, args.n_layers_encoder, args.n_layers_decoder, args.n_max_nodes).to(device)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

# KL-annealing
warmup_epochs = int(0.5 * args.epochs_autoencoder)  # 30% of total epochs
beta_final = 1e-4                                   # you can adjust

args.train_autoencoder = False
# Train VGAE model
if args.train_autoencoder:
    best_val_loss = np.inf
    for epoch in range(1, args.epochs_autoencoder+1):
        # ---- Compute current beta for this epoch ----
        if epoch <= warmup_epochs:
            # linearly scale up from 0 to beta_final
            beta_value = (epoch / warmup_epochs) * beta_final
        else:
            beta_value = beta_final
        # ---------------------------------------------
        autoencoder.train()
        train_loss_all = 0
        train_count = 0
        train_loss_all_recon = 0
        train_loss_all_kld = 0
        cnt_train=0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss, recon, kld  = autoencoder.loss_function(data,beta_value)
            train_loss_all_recon += recon.item()
            train_loss_all_kld += kld.item()
            cnt_train+=1
            loss.backward()
            train_loss_all += loss.item()
            train_count += torch.max(data.batch)+1
            optimizer.step()

        autoencoder.eval()
        val_loss_all = 0
        val_count = 0
        cnt_val = 0
        val_loss_all_recon = 0
        val_loss_all_kld = 0

        with torch.no_grad() :
            for data in val_loader:
                data = data.to(device)
                loss, recon, kld  = autoencoder.loss_function(data,beta_value)
                val_loss_all_recon += recon.item()
                val_loss_all_kld += kld.item()
                val_loss_all += loss.item()
                cnt_val+=1
                val_count += torch.max(data.batch)+1

        if epoch % 1 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('{} Epoch: {:04d}, Train Loss: {:.5f}, Train Reconstruction Loss: {:.2f}, Train KLD Loss: {:.2f}, Val Loss: {:.5f}, Val Reconstruction Loss: {:.2f}, Val KLD Loss: {:.2f}'.format(dt_t,epoch, train_loss_all/cnt_train, train_loss_all_recon/cnt_train, train_loss_all_kld/cnt_train, val_loss_all/cnt_val, val_loss_all_recon/cnt_val, val_loss_all_kld/cnt_val))
            
        scheduler.step()

        if best_val_loss >= val_loss_all:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': autoencoder.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, 'autoencoder.pth.tar')
else:
    checkpoint = torch.load('autoencoder.pth.tar')
    autoencoder.load_state_dict(checkpoint['state_dict'])

autoencoder.eval()


from flow_model_v2 import FlowNN, OTFlowMatching, CondVF

# Instantiate the velocity net
net = FlowNN(
    latent_dim=args.latent_dim,
    n_cond=args.n_condition,
    cond_dim=args.dim_condition,
    hidden_dim=args.hidden_dim_denoise,
    time_emb_dim=128
).to(device)

# We wrap it in CondVF if we want ODE-based decode/encode
flow_model = CondVF(net)


ot_flow = OTFlowMatching(sig_min=0.001)

optimizer = torch.optim.Adam(flow_model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

best_val_loss = float('inf')
args.train_denoiser = True


# Train flow model
if args.train_denoiser:
    for epoch in range(1, args.epochs_denoise + 1):
        flow_model.train()
        train_loss_all = 0
        train_count = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            # 1) get latents from AE
            z1 = autoencoder.encode(data)  # shape (B, latent_dim)
            # 2) cond = data.stats
            cond = data.stats  # shape (B, n_condition)

            # 3) OT flow matching loss
            loss = ot_flow.loss(flow_model, z1, cond)
            loss.backward()
            optimizer.step()

            train_loss_all += loss.item() * z1.size(0)
            train_count += z1.size(0)

        # val loop
        flow_model.eval()
        val_loss_all = 0
        val_count = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                z1 = autoencoder.encode(data)
                cond = data.stats
                loss = ot_flow.loss(flow_model, z1, cond)
                val_loss_all += loss.item() * z1.size(0)
                val_count += z1.size(0)

        train_loss_avg = train_loss_all / train_count
        val_loss_avg   = val_loss_all / val_count
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Train Loss={train_loss_avg:.4f}, Val Loss={val_loss_avg:.4f}")

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save({'state_dict': flow_model.state_dict()}, "flow_model.pth.tar")
else:
    # Load existing flow model
    ckpt = torch.load("flow_model.pth.tar", map_location=device)
    flow_model.load_state_dict(ckpt['state_dict'])
    flow_model.eval()

#del train_loader, val_loader



### output.csv used for kaggle : 
# Save to a CSV file
with open("output_flow.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["graph_id", "edge_list"])

    for k, data in enumerate(tqdm(test_loader, desc='Processing test set')):
        data = data.to(device)
        stat = data.stats         # shape (B, n_condition)
        bs   = stat.size(0)
        graph_ids = data.filename

        # 1) sample latents by decode(...) from t=0->1
        z_sample = flow_model.decode(
            cond=stat,
            batch_size=bs,
            latent_dim=args.latent_dim,
            n_steps=50, 
            method='rk4'
        )

        # 2) decode adjacency
        adj = autoencoder.decode_mu(z_sample, data)

        # 3) build edges, write CSV
        for i in range(bs):
            Gs_generated = construct_nx_from_adj(adj[i].detach().cpu().numpy())
            graph_id = graph_ids[i]
            edge_list_text = ", ".join([f"({u}, {v})" for u, v in Gs_generated.edges()])
            writer.writerow([graph_id, edge_list_text])