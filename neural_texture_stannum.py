from weakref import ref
import taichi as ti
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from taichi.math import ivec2, vec2
# from msssim.pytorch_msssim import MS_SSIM

import torch
import torch.nn as nn
import torch.nn.functional as F

from stannum import Tin

ti.init(arch=ti.cuda)

learning_rate = 1e-2
n_iters = 10000

np_img = ti.tools.imread("test.jpg").astype(np.single) / 255.0
width = np_img.shape[0]
height = np_img.shape[1]

print(width, height)

BATCH_SIZE=30000

img = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
img.from_numpy(np_img)

L = 8
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
    def __init__(self, scale = 8) -> None:
        self.input_positions = ti.Vector.field(2, dtype=ti.f32, shape=(BATCH_SIZE), needs_grad=False)
        self.n_features = 16
        self.grid = ti.field(dtype=ti.f16, shape=(width // scale, height // scale, self.n_features), needs_grad=True)
        self.encoded_positions = ti.field(dtype=ti.f32, shape=(BATCH_SIZE, self.n_features), needs_grad=True)
        self.scale = scale

    @ti.kernel
    def initialize(self):
        for I in ti.grouped(self.grid):
            self.grid[I] = (ti.random() * 2.0 - 1.0) * ti.sqrt(6) / ti.sqrt(self.n_features)

    @ti.kernel
    def encoding(self):
        for i, j in ti.ndrange(self.input_positions.shape[0], self.n_features):
            p = self.input_positions[i]
            uv = p * ti.Vector([width, height]) / self.scale
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

@ti.data_oriented
class MultiResGridEncoding:
    def __init__(self) -> None:
        min_scale = max(1, int(np.exp2(np.floor(np.log2(min(width, height)) - L * 0.5 - 1.0))))
        self.input_positions = ti.Vector.field(2, dtype=ti.f32, shape=(BATCH_SIZE), needs_grad=False)
        self.grids = []
        self.n_features = 0
        for i in range(L):
            scale = min_scale * np.exp2(i * 0.5)
            print(scale)
            self.grids.append(ti.Vector.field(2, dtype=ti.f32, shape=(int(np.ceil(width / scale)), int(np.ceil(height / scale))), needs_grad=True))
            self.n_features += 2
        self.encoded_positions = ti.field(dtype=ti.f32, shape=(BATCH_SIZE, self.n_features), needs_grad=True)
        self.min_scale = min_scale

    @ti.kernel
    def initialize(self):
        for l in ti.static(range(L)):
            for I in ti.grouped(self.grids[l]):
                self.grids[l][I] = (ti.Vector([ti.random(), ti.random()]) * 2.0 - 1.0) * ti.sqrt(6) / ti.sqrt(self.n_features)

    @ti.kernel
    def encoding(self):
        for i in self.input_positions:
            p = self.input_positions[i]
            for l in ti.static(range(L)):
                scale = self.min_scale * (2.0 ** (l * 0.5))
                uv = p * ti.Vector([width, height]) / scale
                iuv = ti.cast(ti.floor(uv), ti.i32)
                fuv = ti.math.fract(uv)
                c00 = self.grids[l][iuv]
                c01 = self.grids[l][iuv + ivec2(0, 1)]
                c10 = self.grids[l][iuv + ivec2(1, 0)]
                c11 = self.grids[l][iuv + ivec2(1, 1)]
                c0 = c00 * (1.0 - fuv[0]) + c10 * fuv[0]
                c1 = c01 * (1.0 - fuv[0]) + c11 * fuv[0]
                c = c0 * (1.0 - fuv[1]) + c1 * fuv[1]
                self.encoded_positions[i, l * 4 + 0] = c.x
                self.encoded_positions[i, l * 4 + 1] = c.y

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
                .register_input_field(self.freq_encoding.input_positions, needs_grad=False) \
                .register_output_field(self.freq_encoding.encoded_positions, needs_grad=False) \
                .finish()
        elif encoding == "dense_grid":
            hidden_size = 256
            n_layers = 8
            self.grid_encoding = DenseGridEncoding()
            input_size = self.grid_encoding.n_features
            self.encoding = Tin(self.grid_encoding, device=torch_device) \
                .register_kernel(self.grid_encoding.encoding) \
                .register_input_field(self.grid_encoding.input_positions, needs_grad=False) \
                .register_internal_field(self.grid_encoding.grid, needs_grad=True) \
                .register_output_field(self.grid_encoding.encoded_positions, needs_grad=True) \
                .finish()
            self.grid_encoding.initialize()
        elif encoding == "multires_grid":
            hidden_size = 256
            n_layers = 6
            self.grid_encoding = MultiResGridEncoding()
            input_size = self.grid_encoding.n_features
            self.encoding = Tin(self.grid_encoding, device=torch_device) \
                .register_kernel(self.grid_encoding.encoding) \
                .register_input_field(self.grid_encoding.input_positions, needs_grad=False) \
                .register_output_field(self.grid_encoding.encoded_positions, needs_grad=True)
            for l in range(L):
                self.encoding = self.encoding.register_internal_field(self.grid_encoding.grids[l], needs_grad=True)
            self.encoding = self.encoding.finish()
            self.grid_encoding.initialize()
        else:
            self.encoding = None
        npars = 0
        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size, bias=False))
                layers.append(nn.ReLU(inplace=True))
                npars += input_size * hidden_size
            elif i == n_layers - 1:
                layers.append(nn.Linear(hidden_size, output_size, bias=False))
                layers.append(nn.Sigmoid())
                npars += hidden_size * output_size
            else:
                layers.append(nn.Linear(hidden_size, hidden_size, bias=False))
                layers.append(nn.ReLU(inplace=True))
                npars += hidden_size * hidden_size
        self.mlp = nn.Sequential(*layers)
        print(self)
        print(f"Number of parameters: {npars}")

    def forward(self, x):
        if self.encoding == None:
            return self.mlp(x)
        encoded = self.encoding(x)
        return self.mlp(encoded)

input_positions = torch.Tensor(BATCH_SIZE, 2).to(torch_device)
output_colors = torch.Tensor(BATCH_SIZE, 3).to(torch_device)

model = MLP(encoding="multires_grid").to(torch_device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-12, weight_decay=1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()
loss_fn = torch.nn.L1Loss().to(torch_device)

refine = False

@ti.kernel
def fill_batch_train(input_positions : ti.types.ndarray(element_dim=1),
                     output_colors : ti.types.ndarray(element_dim=1),
                     refine : ti.template()):
    base = ti.Vector([0.0, 0.0])
    window = ti.Vector([1.0, 1.0])
    if ti.static(refine):
        window = ti.Vector([0.4, 0.4])
        base = ti.Vector([ti.random(), ti.random()]) * (1.0 - window)
    for i in range(BATCH_SIZE):
        uv = base + input_positions[i] * window
        input_positions[i] = uv
        iuv = ti.cast(ti.floor(uv * ti.Vector([width, height])), ti.i32)
        output_colors[i] = img[iuv]

width_scaled = width // 4
height_scaled = height // 4

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
gui = window.get_gui()

writer = SummaryWriter()

loss_smooth_0 = 0.0
loss_smooth_1 = 0.0

soboleng = torch.quasirandom.SobolEngine(dimension=2)

for iter in range(100000):
    input_positions = soboleng.draw(BATCH_SIZE).to(torch_device)
    fill_batch_train(input_positions, output_colors, refine)
    
    with torch.cuda.amp.autocast():
        pred = model(input_positions)
        loss = loss_fn(pred, output_colors)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    writer.add_scalar('Loss/train', loss.item(), iter)

    if iter % 50 == 0:
        i = 0
        while i < (width_scaled * height_scaled):
            fill_batch_test(i, input_positions)
            pred = model(input_positions)
            paint_batch_test(i, pred)
            i += BATCH_SIZE
    
    loss_smooth_0 = loss_smooth_0 * 0.9 + loss.item() * 0.1
    loss_smooth_1 = max(loss_smooth_0, loss_smooth_1 * 0.999 + loss.item() * 0.001)
    
    if iter % 5 == 0:
        learning_rate = 10.0 ** gui.slider_float("learning_rate (log)", np.log10(learning_rate), -10.0, 1.0)
        for g in optimizer.param_groups:
            g['lr'] = learning_rate
        canvas.set_image(rendered)
        refine = gui.checkbox("refine", refine)
        gui.text(f"Iteration {iter}")
        gui.text(f"loss smooth 0 = {loss_smooth_0}")
        gui.text(f"loss smooth 1 = {loss_smooth_1}")
        window.show()
