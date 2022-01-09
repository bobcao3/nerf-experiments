from numpy.core.numeric import indices
from taichi.lang.ops import mod
from torch import nn
import torch.nn.functional as F
import numpy as np
import taichi as ti
import json
import math
import torch
from torch.nn.modules import module
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

ti.init(arch=ti.vulkan)
torch.backends.cuda.matmul.allow_tf32 = True

def loss_fn(X, Y):
  L = (X - Y) * (X - Y)
  return L.sum()
  # return F.mse_loss(X, Y)

set_name = "nerf_synthetic"
scene_name = "lego"
image_w = 800.0
image_h = 800.0

mlp_layers = 8
mlp_hidden = 256
learning_rate = 5e-4
epochs = 1000
batch_size = 4096
optimizer_fn = torch.optim.Adam

def load_desc_from_json(filename):
  f = open(filename, "r")
  content = f.read()
  decoded = json.loads(content)
  print(len(decoded["frames"]), "images from", filename)
  print("=", len(decoded["frames"]) * image_w * image_h, "samples")
  return decoded

# Assume Z is up?

@ti.func
def get_arg(dir_x : ti.f32, dir_y : ti.f32, dir_z : ti.f32):
  theta = ti.atan2(dir_y, dir_x)
  phi = ti.atan2(dir_z, ti.sqrt(dir_x * dir_x + dir_y * dir_y))
  return theta, phi

camera_mtx = ti.Matrix.field(4, 4, dtype=ti.f32, shape=())

@ti.func
def normalize(v):
  return v / ti.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

@ti.kernel
def image_to_data(
  img : ti.template(),
  input : ti.template(),
  output : ti.template(),
  fov_w : ti.f32,
  fov_h : ti.f32):
  for i,j in input_image:
    znear = 0.0
    uv = ti.Vector([i / image_w, j / image_h])
    uv = uv * 2.0 - 1.0
    l = ti.tan(fov_w * 0.5 * math.pi / 180.0)
    t = ti.tan(fov_h * 0.5 * math.pi / 180.0)
    uv.x *= l
    uv.y *= t
    view_dir = normalize(ti.Vector([uv.x, uv.y, -1.0, 0.0]))
    world_dir = camera_mtx[()] @ view_dir
    world_pos = camera_mtx[()] @ ti.Vector([0.0, 0.0, znear, 1.0])
    theta, phi = get_arg(world_dir.x, world_dir.y, world_dir.z)
    input[ti.cast(j * image_w + i, dtype=ti.i32)] = ti.Vector([world_pos.x, world_pos.y, world_pos.z, theta, phi])
    output[ti.cast(j * image_w + i, dtype=ti.i32)] = ti.Vector([img[i, j].x, img[i, j].y, img[i, j].z]) / 255.0
    # output[ti.cast(j * image_w + i, dtype=ti.i32)] = ti.Vector([img[i, j].w, img[i, j].w, img[i, j].w]) / 255.0

input_image = ti.Vector.field((4), dtype=ti.f32, shape=(int(image_w), int(image_h)))
input_data = ti.Vector.field((5), dtype=ti.f32, shape=(int(image_w * image_h)))
output_data = ti.Vector.field((3), dtype=ti.f32, shape=(int(image_w * image_h)))

def generate_data(desc, i):
  img = desc["frames"][i]
  file_name = set_name + "/" + scene_name + "/" + img["file_path"] + ".png"
  # print("loading", file_name)
  input_image.from_numpy(ti.imread(file_name))
  camera_mtx.from_numpy(np.array(img["transform_matrix"]))
  image_to_data(input_image, input_data, output_data, float(desc["camera_angle_x"]), float(desc["camera_angle_x"]))

desc = load_desc_from_json(set_name + "/" + scene_name + "/transforms_train.json")
desc_test = load_desc_from_json(set_name + "/" + scene_name + "/transforms_test.json")

device = "cuda"

class NerfMLP(nn.Module):
  def __init__(self, num_layers, num_hidden):
    super(NerfMLP, self).__init__()
    self.linear_relu_stack_0 = nn.Sequential(
      nn.Linear(60, num_hidden),
      nn.ReLU(inplace=True),
      nn.Linear(num_hidden, num_hidden),
      nn.ReLU(inplace=True),
      nn.Linear(num_hidden, num_hidden),
      nn.ReLU(inplace=True),
      nn.Linear(num_hidden, num_hidden),
      nn.ReLU(inplace=True),
    )
    self.linear_relu_stack_1 = nn.Sequential(
      nn.Linear(60 + num_hidden, num_hidden),
      nn.ReLU(inplace=True),
      nn.Linear(num_hidden, num_hidden),
      nn.ReLU(inplace=True),
      nn.Linear(num_hidden, num_hidden),
      nn.ReLU(inplace=True),
      nn.Linear(num_hidden, 1 + 128),
      nn.ReLU(inplace=True),
    )
    self.view_dependent = nn.Sequential(
      nn.Linear(128 + 2, 3),
      nn.Sigmoid()
    )

  def positional_encoding(self, pos_query):
    L = 10
    encoded = torch.Tensor(size=(pos_query.shape[0], 3, L * 2)).to(device)
    for i in range(L):
      encoded[:,:,i * 2] = torch.sin(np.exp2(i) * math.pi * pos_query)
      encoded[:,:,i * 2 + 1] = torch.cos(np.exp2(i) * math.pi * pos_query)
    return encoded.reshape((-1, 3 * L * 2))

  def query(self, pos_query, dir_query):
    pos_query = self.positional_encoding(pos_query.reshape((-1,3)))
    output0 = self.linear_relu_stack_0(pos_query)
    output1 = torch.cat((output0, pos_query), dim=-1)
    feature_density = self.linear_relu_stack_1(output1)
    density = feature_density[:,0]
    dir_query = dir_query.reshape((-1,2))
    feature_dir = torch.cat((feature_density[:,1:], dir_query), dim=-1)
    color = self.view_dependent(feature_dir)
    return density, color

  def forward(self, x):
    # x [batch, (pos, dir)]
    delta = 0.05
    batch_size = x.shape[0]
    samples = 128
    dir_x = torch.cos(x[:, 3]) * torch.cos(x[:, 4])
    dir_y = torch.sin(x[:, 3]) * torch.cos(x[:, 4])
    dir_z = torch.sin(x[:, 4])
    dir = torch.stack((dir_x, dir_y, dir_z), dim=-1)

    # print("dir", dir[0,:])
    # print("origin", x[0,0:3])

    pos_query = torch.Tensor(size=(samples, x.shape[0], 3)).to(device)
    view_dir = torch.Tensor(size=(samples, x.shape[0], 2)).to(device)
    offsets = torch.rand(size=(x.shape[0],)).to(device)
    for i in range(samples):
      z_dist = (i + offsets) * delta
      pos_query[i] = x[:,0:3] + dir * z_dist[:,None]
      view_dir[i] = x[:,3:5]

    # print("pos_query", pos_query[:,0])
    # print("view_dir", view_dir[:,0])

    # Query
    density, color = self.query(pos_query, view_dir)
    density = density.reshape(samples, batch_size)
    color = color.reshape(samples, batch_size, 3)
    # Convert density to alpha
    alpha = 1.0 - torch.exp(-density * delta)
    # Composite
    weight = alpha * torch.cumprod(1.0 - alpha + 1e-10, dim=0)

    # print("color", color[:,0,:])

    color = color * weight[:,:,None]
    output = color.sum(dim=0)

    # print("density", density[:,0])
    # print("weight", weight[:,0])

    return output

model = NerfMLP(num_layers=mlp_layers, num_hidden=mlp_hidden).to(device)
print(model)

# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         torch.nn.init.xavier_uniform_(m.weight, gain=2.0)
#         m.bias.data.fill_(0.01)

# model.apply(init_weights)

optimizer = optimizer_fn(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()

writer = SummaryWriter()
gui = ti.GUI(res=800, show_gui=False)


# train loop
iter = 0

X = []
Y = []
for i in range(len(desc["frames"])):
  print("load img", i)
  generate_data(desc, i)
  X.append(input_data.to_torch().to(device))
  Y.append(output_data.to_torch().to(device))
X = torch.vstack(X)
Y = torch.vstack(Y)

indices = torch.randperm(X.shape[0])
indices = torch.split(indices, batch_size)

for e in range(epochs):
  frame_loss = 0.0

  for b in range(len(indices)):
    with torch.cuda.amp.autocast():
      pred = model(X[indices[b]])
      loss = loss_fn(pred, Y[indices[b]])
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    frame_loss += loss.item()
    print(e, i, b, "train loss=", loss.item())

    writer.add_scalar('Loss/train', loss.item(), iter)
    iter += 1

  with torch.no_grad():
    test_loss = 0.0
    for i in range(10):
      generate_data(desc_test, i)
      X_test = input_data.to_torch().to(device)
      Y_test = output_data.to_torch().to(device)
      X_test = torch.split(X_test, batch_size)
      Y_test = torch.split(Y_test, batch_size)
      img_pred = []

      for b in range(len(X_test)):
        with torch.cuda.amp.autocast():
          pred = model(X_test[b])
          loss = loss_fn(pred, Y_test[b])
        img_pred.append(pred)

        test_loss += loss.item()

      img_pred = torch.vstack(img_pred)

      gui.set_image(img_pred.cpu().detach().numpy().reshape(800, 800, 3))
      gui.show("output_e" + str(e) + "_i" + str(i) + ".png")
      writer.add_scalar('Loss/test', test_loss, e)

    print("test loss=", test_loss / len(desc_test["frames"]))

    torch.save(model, "model_" + str(e) + ".pth")

