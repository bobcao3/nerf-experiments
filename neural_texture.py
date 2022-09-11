import taichi as ti
import numpy as np

ti.init(arch=ti.vulkan)

real = ti.f32
scalar = lambda: ti.field(dtype=real)
learning_rate = 1e-2
n_iters = 10000

@ti.data_oriented
class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        for w in self.params:
            self._step(w)

    @ti.kernel
    def _step(self, w: ti.template()):
        for I in ti.grouped(w):
            w[I] -= min(max(w.grad[I], -20.0), 20.0) * self.lr

    def zero_grad(self):
        for w in self.params:
            w.grad.fill(0.0)

@ti.func
def sigmoid(x):
    return 1.0 / (1.0 + ti.exp(-x))

@ti.func
def relu(x):
    return max(x, 0.0)

@ti.data_oriented
class Linear:
    def __init__(self,
                 batch_size,
                 n_input,
                 n_output,
                 needs_grad=False,
                 activation=None):
        self.batch_size = batch_size
        self.n_input = n_input
        self.n_output = n_output
        self.activation = activation

        self.hidden = scalar()
        self.output = scalar()

        # array of structs
        self.batch_node = ti.root
        self.n_hidden_node = self.batch_node.dense(ti.i, self.n_output)
        self.weights1_node = self.n_hidden_node.dense(ti.j, self.n_input)

        self.batch_node.dense(
            ti.ij,
            (self.batch_size, self.n_output)).place(self.hidden)
        self.batch_node.dense(
            ti.ij,
            (self.batch_size, self.n_output)).place(self.output)

        self.weights1 = scalar()
        self.bias1 = scalar()

        self.weights1_node.place(self.weights1)
        self.n_hidden_node.place(self.bias1)

        if needs_grad:
            ti.root.lazy_grad()

    def parameters(self):
        return [self.weights1, self.bias1]

    @ti.kernel
    def weights_init(self):
        q1 = ti.sqrt(6) / ti.sqrt(self.n_output + self.n_input)
        for i, j in ti.ndrange(self.n_output,  self.n_input):
            self.weights1[i, j] = (ti.random() * 2 - 1) * q1
        for i in range(self.n_output):
            self.bias1[i] = ti.random() * 0.1

    @ti.kernel
    def _forward(self, nn_input: ti.template()):
        for k, i, j in ti.ndrange(self.batch_size, self.n_output, self.n_input):
            weight = self.weights1[i, j]
            x = nn_input[k, j]
            self.hidden[k, i] += weight * x
        for k, i in ti.ndrange(self.batch_size, self.n_output):
            x = self.hidden[k, i]
            b = self.bias1[i]
            out = x + b
            if ti.static(self.activation != None):
                self.output[k, i] = self.activation(out)
            else:
                self.output[k, i] = out

    @ti.kernel
    def clear(self):
        for I in ti.grouped(self.hidden):
            self.hidden[I] = 0.
        for I in ti.grouped(self.output):
            self.output[I] = 0.

    def forward(self, nn_input):
        self._forward(nn_input)

np_img = ti.tools.imread("test.jpg").astype(np.single) / 255.0
width = np_img.shape[0]
height = np_img.shape[1]

print(width, height)

BATCH_SIZE=8192

img = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
img.from_numpy(np_img)

L = 5
max_scale = 2

@ti.func
def frequency_encoding(pos, freqs : ti.template(), i):
  for axis in ti.static(range(2)):
    for l in range(L):
      x = (2.0 ** (l - max_scale)) * np.pi * pos[axis]
      freqs[i, (axis * L + l) * 2] = ti.sin(x)
      freqs[i, (axis * L + l) * 2 + 1] = ti.cos(x)

input_positions = ti.field(dtype=ti.f32, shape=(BATCH_SIZE, 2 * L * 2), needs_grad=False)
output_colors = ti.field(dtype=ti.f32, shape=(BATCH_SIZE, 4), needs_grad=False)

loss = ti.field(float, shape=(), needs_grad=True)

n_layers = 4

linears = []
parameters = []
for i in range(n_layers):
    l = None
    if i == 0:
        l = Linear(batch_size=BATCH_SIZE, n_input=input_positions.shape[1], n_output=128, needs_grad=True, activation=relu)
    elif i == n_layers - 1:
        l = Linear(batch_size=BATCH_SIZE, n_input=128, n_output=3, needs_grad=True, activation=sigmoid)
    else:
        l = Linear(batch_size=BATCH_SIZE, n_input=128, n_output=128, needs_grad=True, activation=relu)
    l.weights_init()
    parameters.extend(l.parameters())
    linears.append(l)

optimizer = SGD(params=parameters, lr=learning_rate)

@ti.kernel
def fill_batch_train():
    for i in range(BATCH_SIZE):
        rand_i = ti.random()
        rand_j = ti.random()
        frequency_encoding(ti.Vector([rand_i, rand_j]) * 2.0 - 1.0, input_positions, i)
        uv = ti.Vector([rand_i, rand_j]) * ti.Vector([width, height])
        iuv = ti.cast(uv, ti.i32)
        color = img[iuv]
        output_colors[i, 0] = color.r
        output_colors[i, 1] = color.g
        output_colors[i, 2] = color.b

width_scaled = width // 8
height_scaled = height // 8

rendered = ti.Vector.field(4, dtype=ti.f32, shape=(width_scaled, height_scaled))

@ti.kernel
def fill_batch_test(base : ti.i32):
    for i in range(BATCH_SIZE):
        ii = i + base
        iuv = ti.Vector([ii % width_scaled, ii // width_scaled])
        uv = ti.cast(iuv, ti.f32)
        frequency_encoding(ti.Vector([uv[0] / width_scaled, uv[1] / height_scaled]) * 2.0 - 1.0, input_positions, i)


@ti.kernel
def paint_batch_test(base : ti.i32, output : ti.template()):
    for i in range(BATCH_SIZE):
        ii = i + base
        iuv = ti.Vector([ii % width_scaled, ii // width_scaled])
        rendered[iuv] = ti.Vector([output[i, 0], output[i, 1], output[i, 2], 1.0])

@ti.kernel
def l1_loss(output: ti.template(), target: ti.template()):
    for i in ti.grouped(output):
        loss[None] += ti.abs(output[i] - target[i]) / BATCH_SIZE

window = ti.ui.Window("test", (width_scaled, height_scaled))
canvas = window.get_canvas()

for iter in range(10000):
    fill_batch_train()
    
    for l in linears:
        l.clear()

    with ti.ad.Tape(loss=loss):
        x = input_positions
        for l in linears:
            l.forward(x)
            x = l.output
        l1_loss(x, output_colors)
    optimizer.step()

    print(loss[None])

    if iter % 100 == 0:
        i = 0
        while i < (width_scaled * height_scaled):
            fill_batch_test(i)
            for l in linears:
                l.clear()
            x = input_positions
            for l in linears:
                l.forward(x)
                x = l.output
            paint_batch_test(i, x)
            i += BATCH_SIZE
        canvas.set_image(rendered)
        window.show()
