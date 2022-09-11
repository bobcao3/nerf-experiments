from numpy.core.numeric import indices
from taichi.lang.ops import mod
from torch import nn
from torch.functional import norm
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
from deferred_model import DeferredNerf

ti.init(arch=ti.cuda)
torch.backends.cuda.matmul.allow_tf32 = True

def loss_fn(X, Y):
  L = (X - Y) * (X - Y)
  return L.sum()
  # return F.mse_loss(X, Y)

set_name = "nerf_synthetic"
scene_name = "lego"
downscale = 2
image_w = 800.0 / downscale
image_h = 800.0 / downscale

mlp_layers = 8
mlp_hidden = 64
learning_rate = 1e-3
iterations = 300000
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

camera_mtx = ti.Vector.field(3, dtype=ti.f32, shape=(3))

@ti.func
def normalize(v):
  return v / ti.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

@ti.func
def dot(a, b):
  return a.x * b.x + a.y * b.y + a.z * b.z

@ti.kernel
def image_to_data(
  input_img : ti.template(),
  scaled_image : ti.template(),
  input : ti.template(),
  output : ti.template(),
  fov_w : ti.f32,
  fov_h : ti.f32,
  world_pos_x : ti.f32,
  world_pos_y : ti.f32,
  world_pos_z : ti.f32):
  for i,j in scaled_image:
    scaled_image[i, j] = ti.Vector([0.0, 0.0, 0.0, 0.0])
  for i,j in input_img:
    scaled_image[i // downscale, j // downscale] += input_img[i, j] / (downscale * downscale * 255)
  for i,j in scaled_image:
    uv = ti.Vector([(i + 0.5) / image_w, (j + 0.5) / image_h])
    uv = uv * 2.0 - 1.0
    l = ti.tan(fov_w * 0.5)
    t = ti.tan(fov_h * 0.5)
    uv.x *= l
    uv.y *= t
    view_dir = ti.Vector([uv.x, uv.y, -1.0])
    world_dir = ti.Vector([
      dot(camera_mtx[0], view_dir),
      dot(camera_mtx[1], view_dir),
      dot(camera_mtx[2], view_dir)])
    input[ti.cast(i * image_h + j, dtype=ti.i32)] = ti.Vector([world_pos_x, world_pos_y, world_pos_z, world_dir.x, world_dir.y, world_dir.z])
    output[ti.cast(i * image_h + j, dtype=ti.i32)] = ti.Vector([scaled_image[i, j].x, scaled_image[i, j].y, scaled_image[i, j].z])

input_image = ti.Vector.field((4), dtype=ti.f32, shape=(int(image_w) * downscale, int(image_h) * downscale))
input_data = ti.Vector.field((6), dtype=ti.f32, shape=(int(image_w * image_h)))
output_data = ti.Vector.field((3), dtype=ti.f32, shape=(int(image_w * image_h)))
scaled_image = ti.Vector.field((4), dtype=ti.f32, shape=(int(image_w), int(image_h)))

def generate_data(desc, i):
  img = desc["frames"][i]
  file_name = set_name + "/" + scene_name + "/" + img["file_path"] + ".png"
  # print("loading", file_name)
  npimg = ti.imread(file_name)
  input_image.from_numpy(npimg)
  mtx = np.array(img["transform_matrix"])
  camera_mtx.from_numpy(mtx[:3,:3])
  ray_o = mtx[:3,-1]
  ti.sync()
  image_to_data(input_image, scaled_image, input_data, output_data, float(desc["camera_angle_x"]), float(desc["camera_angle_x"]), ray_o[0], ray_o[1], ray_o[2])

desc = load_desc_from_json(set_name + "/" + scene_name + "/transforms_train.json")
desc_test = load_desc_from_json(set_name + "/" + scene_name + "/transforms_test.json")

device = "cuda"

model = DeferredNerf(num_layers=mlp_layers, num_hidden=mlp_hidden).to(device)
print(model)

optimizer = optimizer_fn(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()

# train loop
iter = 0

X = []
Y = []
for i in range(len(desc["frames"])):
  print("load img", i)
  generate_data(desc, i)
  ti.sync()
  X.append(input_data.to_torch().to(device).reshape(-1,6))
  Y.append(output_data.to_torch().to(device).reshape(-1,3))
X = torch.vstack(X)
Y = torch.vstack(Y)

ti.imwrite(input_image, "input_full_sample.png")
ti.imwrite(scaled_image, "input_sample.png")

torch.save(X, "input_samples.th")
torch.save(Y, "output_samples.th")

# exit(0)

writer = SummaryWriter()

indices = torch.randperm(X.shape[0])
indices = torch.split(indices, batch_size)

test_indicies = torch.randperm(len(desc_test["frames"]))

for iter in range(iterations):
  accum_loss = 0.0
  
  b = np.random.randint(0, len(indices))
  Xbatch = X[indices[b]]
  Ybatch = Y[indices[b]]

  with torch.cuda.amp.autocast():
    pred, diffuse = model(Xbatch)
    loss = loss_fn(pred, Ybatch) * 0.1
    loss += loss_fn(diffuse, Ybatch)
  
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()
  accum_loss += loss.item()

  if iter % 10 == 9:
    print(iter, b, "train loss=", accum_loss / 10)
    writer.add_scalar('Loss/train', accum_loss / 10, iter)
    accum_loss = 0.0

  if iter % 1000 == 0:
    with torch.no_grad():
      test_loss = 0.0
      for i in np.array(test_indicies[:10]):
        generate_data(desc_test, i)
        ti.sync()
        X_test = input_data.to_torch().to(device).reshape(-1,6)
        Y_test = output_data.to_torch().to(device).reshape(-1,3)

        Xbatch = X_test.split(batch_size)
        Ybatch = Y_test.split(batch_size)

        img_pred = []

        for b in range(len(Xbatch)):
          with torch.cuda.amp.autocast():
            pred, _ = model(Xbatch[b])
            loss = loss_fn(pred, Ybatch[b])
            img_pred.append(pred)
            test_loss += loss.item()

        img_pred = torch.vstack(img_pred)
        img_pred = img_pred.cpu().detach().numpy()
        img_pred = img_pred.reshape((int(image_w), int(image_h), 3))

        if i == test_indicies[0]:
          ti.imwrite(img_pred, "output_iter" + str(iter) + "_r" + str(i) + ".png")
      
      writer.add_scalar('Loss/test', test_loss / 10.0, iter / 1000.0)
      print("test loss=", test_loss / 10.0)

  if iter % 5000 == 0:
    torch.save(model, "model_" + str(iter) + ".pth")

