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
from utils import (
    construct_nx_from_adj,
    create_training_folder,
    linear_beta_schedule,
    preprocess_dataset,
)
from warmup_scheduler import WarmupReduceLROnPlateau
import torch.nn.utils as nn_utils

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
parser.add_argument('--epochs-denoise', type=int, default=100, help="Number of training epochs for the denoising model (default: 100)")

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

# Dimensionality of prompt embeddings
parser.add_argument('--dim-prompt', type=int, default=384, help="Dimensionality of prompt embeddings (default: 384)")

parser.add_argument('--promp-encoding-model', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help="Prompt encoding model (default: sentence-transformers/all-MiniLM-L6-v2)")

parser.add_argument("--metrics-name", type=str, default="metrics.csv", help="Name of the metrics file (default: metrics.csv)")
parser.add_argument("--denoiser-metrics-name", type=str, default="denoiser_metrics.csv", help="Name of the denoiser metrics file (default: denoiser_metrics.csv)")
parser.add_argument("--training-description", type=str, default="This training session focuses on VGAE and Denoiser models with specific hyperparameters and data preprocessing methods.", help="Description of the training session (default: This training session focuses on VGAE and Denoiser models with specific hyperparameters and data preprocessing methods.)")

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# preprocess train data, validation data and test data. Only once for the first time that you run the code. Then the appropriate .pt files will be saved and loaded.
trainset = preprocess_dataset("train", args.n_max_nodes, args.spectral_emb_dim, args.promp_encoding_model)
validset = preprocess_dataset("valid", args.n_max_nodes, args.spectral_emb_dim, args.promp_encoding_model)
testset = preprocess_dataset("test", args.n_max_nodes, args.spectral_emb_dim, args.promp_encoding_model)

# initialize data loaders
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)


# initialize VGAE model
autoencoder = VariationalAutoEncoder(args.spectral_emb_dim+1, args.hidden_dim_encoder, args.hidden_dim_decoder, args.latent_dim, args.n_layers_encoder, args.n_layers_decoder, args.n_max_nodes).to(device)

# optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=args.lr, betas=(0.9, 0.999))


# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
### Scheduler ###
# decrease learning rate by a factor of 0.5 if the train reconstruction loss does not decrease for 10 epochs
# add a warm-up phase for the first 100 epochs
reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
warmup_scheduler = WarmupReduceLROnPlateau(
    optimizer=optimizer,
    scheduler=reduce_on_plateau,
    warmup_steps=100,
    initial_lr=1e-5,
    target_lr=1e-3
)


# Fichier pour sauvegarder les métriques
metrics_file = args.metrics_name
denoiser_metrics_file = args.denoiser_metrics_name
training_description = args.training_description

# Créer le dossier pour l'entraînement
training_folder = create_training_folder(training_description, args)

# Fichiers de métriques dans le dossier
vgae_metrics_file = os.path.join(training_folder, metrics_file)
denoiser_metrics_file = os.path.join(training_folder, denoiser_metrics_file)


# Sauvegarde des métriques dans un fichier CSV
with open(vgae_metrics_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Learning Rate", "Train Loss", "Train Recon Loss", "Train KLD Loss", 
                     "Val Loss", "Val Recon Loss", "Val KLD Loss"])

with open(denoiser_metrics_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Learning Rate", "Train Loss", "Val Loss"])

# KL-annealing
warmup_epochs = int(0.5 * args.epochs_autoencoder)  # 30% of total epochs
beta_final = 1e-4                                   # you can adjust

args.train_autoencoder = True
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

        # Logging et sauvegarde des métriques
        current_lr = warmup_scheduler.get_last_lr()[-1]
        train_loss_avg = train_loss_all / cnt_train
        val_loss_avg = val_loss_all / cnt_val
        train_recon_avg = train_loss_all_recon / cnt_train
        train_kld_avg = train_loss_all_kld / cnt_train
        val_recon_avg = val_loss_all_recon / cnt_val
        val_kld_avg = val_loss_all_kld / cnt_val

        if epoch % 1 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('{} Epoch: {:04d}, lr: {:.5f}, Train Loss: {:.5f}, Train Recon Loss: {:.3f}, Train KLD Loss: {:.2f}, Val Loss: {:.5f}, Val Recon Loss: {:.3f}, Val KLD Loss: {:.2f}'.format(
                dt_t, epoch, current_lr, train_loss_avg, train_recon_avg, train_kld_avg, val_loss_avg, val_recon_avg, val_kld_avg))
        
        # Écriture dans le fichier CSV
        with open(vgae_metrics_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, current_lr, train_loss_avg, train_recon_avg, train_kld_avg, 
                             val_loss_avg, val_recon_avg, val_kld_avg])
            
        warmup_scheduler.step(train_loss_all_recon/cnt_train)

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



# define beta schedule
betas = linear_beta_schedule(timesteps=args.timesteps)

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

# initialize denoising model
denoise_model = DenoiseNN(input_dim=args.latent_dim, hidden_dim=args.hidden_dim_denoise, n_layers=args.n_layers_denoise, n_cond=args.n_condition, d_cond=args.dim_condition, d_prompt=args.dim_prompt).to(device)
optimizer = torch.optim.Adam(denoise_model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

args.train_denoiser = True


# Train denoising model
if args.train_denoiser:
    best_val_loss = np.inf
    for epoch in range(1, args.epochs_denoise+1):
        denoise_model.train()
        train_loss_all = 0
        train_count = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            x_g = autoencoder.encode(data)
            t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
            conds = [data.stats, data.prompt]
            loss = p_losses(denoise_model, x_g, t, conds, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")
            loss.backward()
            train_loss_all += x_g.size(0) * loss.item()
            train_count += x_g.size(0)
            # ---- GRADIENT CLIPPING ----
            nn_utils.clip_grad_norm_(denoise_model.parameters(), max_norm=1.0)
            optimizer.step()
        denoise_model.eval()
        val_loss_all = 0
        val_count = 0
        for data in val_loader:
            data = data.to(device)
            x_g = autoencoder.encode(data)
            t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
            conds = [data.stats, data.prompt]
            loss = p_losses(denoise_model, x_g, t, conds, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")
            val_loss_all += x_g.size(0) * loss.item()
            val_count += x_g.size(0)

        # Calcul des moyennes des pertes
        current_lr = scheduler.get_last_lr()[-1]
        train_loss_avg = train_loss_all / cnt_train
        val_loss_avg = val_loss_all / cnt_val

        # Affichage des logs
        if epoch % 5 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('{} Epoch: {:04d}, lr: {:.5f}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(
                dt_t, epoch, current_lr, train_loss_avg, val_loss_avg))
        
        # Enregistrer les métriques dans le fichier CSV
        with open(denoiser_metrics_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, current_lr, train_loss_avg, val_loss_avg])

        scheduler.step()

        if best_val_loss >= val_loss_all:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': denoise_model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, 'denoise_model.pth.tar')
else:
    checkpoint = torch.load('denoise_model.pth.tar')
    denoise_model.load_state_dict(checkpoint['state_dict'])

denoise_model.eval()

### output.csv used for kaggle : 
# Save to a CSV file
with open("output.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(["graph_id", "edge_list"])
    for k, data in enumerate(tqdm(test_loader, desc='Processing test set',)):
        data = data.to(device)
        
        stat = data.stats
        bs = stat.size(0)

        graph_ids = data.filename

        conds = [data.stats, data.prompt]
        samples = sample(denoise_model, conds, latent_dim=args.latent_dim, timesteps=args.timesteps, betas=betas, batch_size=bs)
        x_sample = samples[-1]
        adj = autoencoder.decode_mu(x_sample,data)
        stat_d = torch.reshape(stat, (-1, args.n_condition))


        for i in range(stat.size(0)):
            stat_x = stat_d[i]

            Gs_generated = construct_nx_from_adj(adj[i,:,:].detach().cpu().numpy())
            stat_x = stat_x.detach().cpu().numpy()

            # Define a graph ID
            graph_id = graph_ids[i]

            # Convert the edge list to a single string
            edge_list_text = ", ".join([f"({u}, {v})" for u, v in Gs_generated.edges()])           
            # Write the graph ID and the full edge list as a single row
            writer.writerow([graph_id, edge_list_text])