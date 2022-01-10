import torch
from torch import nn

class NerfMLP(nn.Module):
  def __init__(self, num_layers, num_hidden):
    super(NerfMLP, self).__init__()
    self.linear_relu_stack_0 = nn.Sequential(
      nn.Linear(30, num_hidden),
      nn.ReLU(inplace=True),
      nn.Linear(num_hidden, num_hidden),
      nn.ReLU(inplace=True),
      nn.Linear(num_hidden, num_hidden),
      nn.ReLU(inplace=True),
      nn.Linear(num_hidden, num_hidden),
      nn.ReLU(inplace=True),
    )
    self.linear_relu_stack_1 = nn.Sequential(
      nn.Linear(30 + num_hidden, num_hidden),
      nn.ReLU(inplace=True),
      nn.Linear(num_hidden, num_hidden),
      nn.ReLU(inplace=True),
      nn.Linear(num_hidden, num_hidden),
      nn.ReLU(inplace=True),
      nn.Linear(num_hidden, 1 + 128)
    )
    self.view_dependent = nn.Sequential(
      nn.Linear(128 + 3, 3),
      nn.Sigmoid()
    )

  def positional_encoding(self, pos_query):
    L = 5
    max_scale = 2
    encoded = torch.Tensor(size=(pos_query.shape[0], 3, L * 2)).to(device)
    for i in range(L):
      encoded[:,:,i * 2] = torch.sin(np.exp2(i - max_scale) * math.pi * pos_query)
      encoded[:,:,i * 2 + 1] = torch.cos(np.exp2(i - max_scale) * math.pi * pos_query)
    return encoded.reshape((-1, 3 * L * 2))

  def query(self, pos_query, dir_query):
    pos_query = self.positional_encoding(pos_query.reshape((-1,3)))
    output0 = self.linear_relu_stack_0(pos_query)
    output1 = torch.cat((output0, pos_query), dim=-1)
    feature_density = self.linear_relu_stack_1(output1)
    density = feature_density[:,0]
    dir_query = dir_query.reshape((-1,3))
    feature_dir = torch.cat((F.relu(feature_density[:,1:]), dir_query), dim=-1)
    color = self.view_dependent(feature_dir)
    return density, color

  def forward(self, x, random_regularization_std=0.0):
    # x [batch, (pos, dir)]
    batch_size = x.shape[0]
    samples = 64

    near = 2.0
    far = 6.0

    pos_query = torch.Tensor(size=(samples, x.shape[0], 3)).to(device)
    view_dir = torch.Tensor(size=(samples, x.shape[0], 3)).to(device)
    offsets = torch.rand(size=(samples, x.shape[0])).to(device)
    dists = torch.Tensor(size=(samples, x.shape[0])).to(device)
    norms = torch.norm(x[:,3:6], dim=-1)
    for i in range(samples):
      z_dist = near + (i + offsets[i]) * (far - near) / samples
      view_dir[i] = x[:,3:6] / norms[:,None]
      pos_query[i] = x[:,0:3] + x[:,3:6] * z_dist[:,None]
      dists[i] = z_dist * norms
    dists[:samples-1] = dists[1:] - dists[:samples-1]
    dists[-1] = 1e10

    # Query
    density, color = self.query(pos_query, view_dir)
    density = density.reshape(samples, batch_size)
    if random_regularization_std > 0.0:
      density = density + torch.randn(density.shape).to(device) * random_regularization_std
    color = color.reshape(samples, batch_size, 3)
    # Convert density to alpha
    alpha = 1.0 - torch.exp(-F.relu(density) * dists)
    # Composite
    weight = alpha * torch.cumprod(1.0 - alpha + 1e-10, dim=0)

    color = color * weight[:,:,None]
    output = color.sum(dim=0)

    return output
