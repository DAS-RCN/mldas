# Externals
import torch

class MLP(torch.nn.Module):
    def __init__(self, n_layer):
        super(MLP, self).__init__()
        layers = []
        for ii in range(len(n_layer)-2):
            layers.append(torch.nn.Linear(n_layer[ii], n_layer[ii+1]))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=0.2))
        layers.append(torch.nn.Linear(n_layer[-2],n_layer[-1]))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.layers(x)
        return x

def get_model(**kwargs):
    """
    Constructs a ResNet model.
    """
    return MLP(**kwargs)
