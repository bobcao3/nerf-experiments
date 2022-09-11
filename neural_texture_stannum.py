from queue import Empty
import taichi as ti
import numpy as np

from taichi.math import ivec2, vec2

import torch
import torch.nn as nn
import torch.nn.functional as F

from stannum import Tin

ti.init(arch=ti.cuda)

learning_rate = 1e-3
n_iters = 10000

np_img = ti.tools.imread("test.jpg").astype(np.single) / 255.0
width = np_img.shape[0]
height = np_img.shape[1]

print(width, height)

BATCH_SIZE=15000

img = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
img.from_numpy(np_img)

L = 5
max_scale = 2

@ti.data_oriented
class FrequencyEncoding:
    def __init__(self) -> None:
        self.input_positions = ti.Vector.field(2, dtype=ti.f32, shape=(BATCH_SIZE), needs_grad=False)
        self.encoded_positions = ti.field(dtype=ti.f32, shape=(BATCH_SIZE, 2 * L * 2), needs_grad=False)

    @ti.kernel
    def frequency_encoding(self):
        for i in self.input_positions:
            p = self.input_positions[i]
            for axis in ti.static(range(2)):
                for l in range(L):
                    x = (2.0 ** (l - max_scale)) * np.pi * p[axis]
                    self.encoded_positions[i, (axis * L + l) * 2] = ti.sin(x)
                    self.encoded_positions[i, (axis * L + l) * 2 + 1] = ti.cos(x)

@ti.data_oriented
class DenseGridEncoding:
    def __init__(self, scale = 32) -> None:
        self.input_positions = ti.Vector.field(2, dtype=ti.f32, shape=(BATCH_SIZE), needs_grad=False)
        self.n_features = 16
        self.grid = ti.field(dtype=ti.f32, shape=(width // scale, height // scale, self.n_features), needs_grad=True)
        self.encoded_positions = ti.field(dtype=ti.f32, shape=(BATCH_SIZE, self.n_features + 2), needs_grad=True)
        self.scale = scale

    @ti.kernel
    def initialize(self):
        for I in ti.grouped(self.grid):
            self.grid[I] = ti.random() * 2.0 - 1.0

    @ti.kernel
    def encoding(self):
        for i, j in ti.ndrange(self.input_positions.shape[0], self.n_features):
            p = self.input_positions[i]
            uv = p / self.scale
            iuv = ti.cast(ti.floor(uv), ti.i32)
            fuv = ti.math.fract(uv)
            c00 = self.grid[iuv, j]
            c01 = self.grid[iuv + ivec2(0, 1), j]
            c10 = self.grid[iuv + ivec2(1, 0), j]
            c11 = self.grid[iuv + ivec2(1, 1), j]
            c0 = c00 * (1.0 - fuv[0]) + c10 * fuv[0]
            c1 = c01 * (1.0 - fuv[0]) + c11 * fuv[0]
            c = c0 * (1.0 - fuv[1]) + c1 * fuv[1]
            self.encoded_positions[i, j] = c
        for i in ti.ndrange(self.input_positions.shape[0]):
            p = self.input_positions[i]
            uv = p / self.scale
            iuv = ti.cast(ti.floor(uv), ti.i32)
            fuv = ti.math.fract(uv)
            self.encoded_positions[i, self.n_features] = fuv.x
            self.encoded_positions[i, self.n_features + 1] = fuv.y

torch_device = torch.device("cuda:0")

class MLP(nn.Module):
    def __init__(self, encoding=None):
        super(MLP, self).__init__()
        layers = []
        input_size = 2
        output_size = 3
        hidden_size = 256
        n_layers = 8
        if encoding == "frequency":
            input_size = 2 * L * 2
            self.freq_encoding = FrequencyEncoding()
            self.encoding = Tin(self.freq_encoding, device=torch_device) \
                .register_kernel(self.freq_encoding.frequency_encoding) \
                .register_input_field(self.freq_encoding.input_positions) \
                .register_output_field(self.freq_encoding.encoded_positions) \
                .finish()
        elif encoding == "dense_grid":
            hidden_size = 64
            n_layers = 6
            self.grid_encoding = DenseGridEncoding()
            input_size = self.grid_encoding.n_features + 2
            self.encoding = Tin(self.grid_encoding, device=torch_device) \
                .register_kernel(self.grid_encoding.encoding) \
                .register_input_field(self.grid_encoding.input_positions) \
                .register_internal_field(self.grid_encoding.grid, needs_grad=True) \
                .register_output_field(self.grid_encoding.encoded_positions, needs_grad=True) \
                .finish()
            self.grid_encoding.initialize()
        else:
            self.encoding = None
        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size, bias=False))
                layers.append(nn.ReLU(inplace=True))
            elif i == n_layers - 1:
                layers.append(nn.Linear(hidden_size, output_size, bias=False))
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.Linear(hidden_size, hidden_size, bias=False))
                layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        if self.encoding == None:
            return self.mlp(x)
        encoded = self.encoding(x)
        return self.mlp(encoded)

input_positions = torch.Tensor(BATCH_SIZE, 2).to(torch_device)
output_colors = torch.Tensor(BATCH_SIZE, 3).to(torch_device)

model = MLP(encoding="dense_grid").to(torch_device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()
loss_fn = torch.nn.L1Loss().to(torch_device)

@ti.kernel
def fill_batch_train(input_positions : ti.types.ndarray(element_dim=1),
                     output_colors : ti.types.ndarray(element_dim=1)):
    for i in range(BATCH_SIZE):
        rand_i = ti.random()
        rand_j = ti.random()
        input_positions[i] = ti.Vector([rand_i, rand_j])
        uv = ti.Vector([rand_i, rand_j]) * ti.Vector([width, height])
        iuv = ti.cast(uv, ti.i32)
        color = img[iuv]
        output_colors[i] = color

width_scaled = width // 8
height_scaled = height // 8

rendered = ti.Vector.field(4, dtype=ti.f32, shape=(width_scaled, height_scaled))

@ti.kernel
def fill_batch_test(base : ti.i32, input_positions : ti.types.ndarray(element_dim=1)):
    for i in range(BATCH_SIZE):
        ii = i + base
        iuv = ti.Vector([ii % width_scaled, ii // width_scaled])
        uv = ti.cast(iuv, ti.f32)
        input_positions[i] = uv / ti.Vector([width_scaled, height_scaled])

@ti.kernel
def paint_batch_test(base : ti.i32, output : ti.types.ndarray(element_dim=1)):
    for i in range(BATCH_SIZE):
        ii = i + base
        iuv = ti.Vector([ii % width_scaled, ii // width_scaled])
        rendered[iuv] = ti.Vector([output[i].r, output[i].g, output[i].b, 1.0])

window = ti.ui.Window("test", (width_scaled, height_scaled))
canvas = window.get_canvas()

for iter in range(100000):
    fill_batch_train(input_positions, output_colors)
    
    with torch.cuda.amp.autocast():
        pred = model(input_positions)
        loss = loss_fn(pred, output_colors)

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if iter % 10 == 0:
        print(loss.item())

    if iter % 50 == 0:
        i = 0
        while i < (width_scaled * height_scaled):
            fill_batch_test(i, input_positions)
            pred = model(input_positions)
            paint_batch_test(i, pred)
            i += BATCH_SIZE
        canvas.set_image(rendered)
        window.show()
