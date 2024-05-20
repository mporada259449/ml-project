# Define the AutoEncoder model
import torch
from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 14),
            nn.ReLU(),
            nn.Linear(14, 7),
            nn.ReLU(),
            nn.Linear(7, 3)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(3, 7),
            nn.ReLU(),
            nn.Linear(7, 14),
            nn.ReLU(),
            nn.Linear(14, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class SparseAutoEncoder(nn.Module):
    def __init__(self, input_dim, sparsity_target=0.05, sparsity_weight=0.3):
        super(SparseAutoEncoder, self).__init__()

        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 14),
            nn.ReLU(),
            nn.Linear(14, 7),
            nn.ReLU(),
            nn.Linear(7, 3)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(3, 7),
            nn.ReLU(),
            nn.Linear(7, 14),
            nn.ReLU(),
            nn.Linear(14, input_dim),
            nn.Sigmoid()
        )

        # Initialize weights to prevent NaNs
        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_normal_(layer.weight)
            layer.bias.data.fill_(0.01)
    
    def kl_divergence(self, p, q, e = 1e-10):
        q = torch.clamp(q, min=1e-10, max=1.0)
        return p * torch.log((p+e) / (q+e)) + (1 - p) * torch.log((1 - (p+e)) / (1 - (q+e)))
    
    def sparse_loss(self, criterion, decoded, data, encoded):
        mse_loss = criterion(decoded, data)
        
        mean_activation = torch.mean(encoded, dim=0)
        
        kl_div = self.kl_divergence(self.sparsity_target, mean_activation)
        penalty = torch.sum(kl_div)  

        return mse_loss + self.sparsity_weight * penalty
    
    # Define the Variational AutoEncoder model
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 14),
            nn.ReLU(),
            nn.Linear(14, 7),
            nn.ReLU()
        )
        # Latent space distribution parameters
        self.fc_mu = nn.Linear(7, 3)
        self.fc_logvar = nn.Linear(7, 3)

        self.decoder = nn.Sequential(
            nn.Linear(3,7),
            nn.ReLU(),
            nn.Linear(7, 14),
            nn.ReLU(),
            nn.Linear(14, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std
    
    def forward(self, x):

        encoded = self.encoder(x)

        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)

        decoded = self.decoder(z)

        return decoded, mu, logvar

    def vae_loss(self, criterion, decoded, data, mu, logvar):

        MSE = criterion(decoded, data)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE + KLD