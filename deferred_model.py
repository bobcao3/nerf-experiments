from random import sample
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

import taichi as ti

from math_utils import ray_aabb_intersection

device = 'cuda'

@ti.kernel
def gen_samples(x : ti.types.ndarray(element_dim=1),
                pos_query : ti.types.ndarray(element_dim=1),
                view_dirs : ti.types.ndarray(element_dim=1),
                dists : ti.types.ndarray(),
                n_samples : ti.i32, batch_size : ti.i32):
  for i in range(batch_size):
    vec = x[i]
    ray_origin = ti.Vector([vec[0], vec[1], vec[2]])
    ray_dir = ti.Vector([vec[3], vec[4], vec[5]]).normalized()
    isect, near, far = ray_aabb_intersection(ti.Vector([-1.5, -1.5, -1.5]), ti.Vector([1.5, 1.5, 1.5]), ray_origin, ray_dir)
    if not isect:
      near = 2.0
      far = 6.0
    view_dirs[i] = -ray_dir
    for j in range(n_samples):
      d = near + (far - near) / ti.cast(n_samples, ti.f32) * (ti.cast(j, ti.f32) + ti.random())
      pos_query[j, i] = ray_origin + ray_dir * d
      dists[j, i] = d
    for j in range(n_samples - 1):
      dists[j, i] = dists[j + 1, i] - dists[j, i]
    dists[n_samples - 1, i] = 1e10    

class DeferredNerf(nn.Module):
  def __init__(self, num_layers, num_hidden):
    super(DeferredNerf, self).__init__()
    self.density_diffuse_feature_net = nn.Sequential(
      nn.Linear(30, num_hidden),
      nn.ReLU(inplace=True),
      nn.Linear(num_hidden, num_hidden),
      nn.ReLU(inplace=True),
      nn.Linear(num_hidden, num_hidden),
      nn.ReLU(inplace=True),
      nn.Linear(num_hidden, num_hidden),
      nn.ReLU(inplace=True),
      nn.Linear(num_hidden, num_hidden),
      nn.ReLU(inplace=True),
      nn.Linear(num_hidden, num_hidden),
      nn.ReLU(inplace=True),
      nn.Linear(num_hidden, 1 + 3 + 4)
    )
    self.specular_net = nn.Sequential(
      nn.Linear(4 + 3, 16),
      nn.ReLU(inplace=True),
      nn.Linear(16, 3),
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

  def query(self, pos_query):
    pos_query = self.positional_encoding(pos_query.reshape((-1,3)))
    density_feature = self.density_diffuse_feature_net(pos_query)
    density = density_feature[:,0]
    diffuse = torch.sigmoid(density_feature[:,1:4])
    feature = torch.sigmoid(density_feature[:,4:])
    return density, diffuse, feature
  
  def composite(self, density, color, feature, dir_query, dists, samples, batch_size):
    density = density.reshape(samples, batch_size)
    color = color.reshape(samples, batch_size, 3)
    feature = feature.reshape(samples, batch_size, 4)
    # Convert density to alpha
    alpha = 1.0 - torch.exp(-F.relu(density) * dists)
    # Composite
    weight = alpha * torch.cumprod(1.0 - alpha + 1e-10, dim=0)

    diffuse = (color * weight[:,:,None]).sum(dim=0)
    feature = (feature * weight[:,:,None]).sum(dim=0)

    feature_dir = torch.cat((feature, dir_query), dim=-1)

    specular = self.specular_net(feature_dir)

    return diffuse + specular, diffuse

  def forward(self, x):
    # x [batch, (pos, dir)]
    batch_size = x.shape[0]
    samples = 512

    pos_query = torch.Tensor(size=(samples, x.shape[0], 3)).to(device)
    view_dir = torch.Tensor(size=(x.shape[0], 3)).to(device)
    dists = torch.Tensor(size=(samples, x.shape[0])).to(device)

    gen_samples(x, pos_query, view_dir, dists, samples, batch_size)
    ti.sync()
    torch.cuda.synchronize(device=None)

    # Query fine model
    density, color, feature = self.query(pos_query)
    output = self.composite(density, color, feature, view_dir, dists, samples, batch_size)

    return output
