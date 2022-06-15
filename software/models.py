import torch 
import torch.nn as nn

#Visual Rep Dim (i3d): 400
#IMU Dim: 36

class Generator(nn.Module):
  def __init__(self, i3d_dim=400, z_dim=10, hidden_dim=500, imu_dim=36):
    super(Generator, self).__init__()
    self.main = nn.Sequential(nn.Linear(z_dim + i3d_dim, hidden_dim),
                              nn.LeakyReLU(),
                              nn.Linear(hidden_dim, imu_dim))
    
  def forward(self, z, v, test=False):
    if test == False:
      input = torch.cat([z, v], 1)
    else:
      input = torch.cat([z, v], 0)
    output = self.main(input)
    return output


class Discriminator(nn.Module):
  def __init__(self, imu_dim=36, hidden_dim=500, n_classes=18):
    super(Discriminator, self).__init__()
    self.main = nn.Sequential(nn.Linear(imu_dim, 200),
                              nn.ReLU(),
                              nn.Linear(200, 350),
                              nn.ReLU(),
                              nn.Linear(350, hidden_dim),
                              nn.ReLU())
    
    #Discriminator branch (real/fake)
    self.D_gan = nn.Linear(hidden_dim, 1)

    #Classifier branch (cls loss on classes)
    self.D_cls = nn.Linear(hidden_dim, n_classes)

  def forward(self, input):
    hidden = self.main(input)
    return self.D_gan(hidden), self.D_cls(hidden)

  

#Weight initilization
import torch.nn.init as init
def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.xavier_normal(m.weight.data)
        init.constant(m.bias, 0.0)


def reset_grad(nets):
    for net in nets:
        net.zero_grad()