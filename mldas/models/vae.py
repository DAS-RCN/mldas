# Externals
import torch
from torch.nn import functional as F

class ShallowVAE(torch.nn.Module):

  def __init__(self, m, n, b):
    super(ShallowVAE, self).__init__()
    self.N = m * n
    self.fc1 = torch.nn.Linear(self.N, 250)
    self.fc21 = torch.nn.Linear(250, b)
    self.fc22 = torch.nn.Linear(250, b)
    self.fc3 = torch.nn.Linear(b, 250)
    self.fc4 = torch.nn.Linear(250, self.N)

  def encoder(self, x):
    h1 = F.relu(self.fc1(x))
    return self.fc21(h1), self.fc22(h1)

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

  def decoder(self, z):
    h3 = F.relu(self.fc3(z))
    return self.fc4(h3)

  def forward(self, x):
    mu, logvar = self.encoder(x)
    z = self.reparameterize(mu, logvar)
    return z, self.decoder(z), mu, logvar

def get_model(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ShallowVAE(**kwargs)
