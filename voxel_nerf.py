import torch
import taichi as ti
import numpy as np
from math_utils import ray_aabb_intersection

data_t = ti.f32
input_vec_t = ti.types.vector(1 + 3 * 9, data_t) # sigma + 3 * (9 SH coeffs)
pos_vec_t = ti.types.vector(3, data_t) # XYZ
output_vec_t = ti.types.vector(4, data_t) # RGB, alpha

@ti.func
def l2_sh(coeffs, pos):
    return \
        coeffs[0] * ti.sqrt(1.0 / (4.0 * np.pi)) + \
        coeffs[1] * ti.sqrt(3.0 / (4.0 * np.pi)) * pos.x + \
        coeffs[2] * ti.sqrt(3.0 / (4.0 * np.pi)) * pos.y + \
        coeffs[3] * ti.sqrt(3.0 / (4.0 * np.pi)) * pos.z + \
        coeffs[4] * ti.sqrt(15.0 / (4.0 * np.pi)) * pos.x * pos.y + \
        coeffs[5] * ti.sqrt(15.0 / (4.0 * np.pi)) * pos.y * pos.z + \
        coeffs[6] * ti.sqrt(5.0 / (16.0 * np.pi)) * (3.0 * (pos.z ** 2.0) - 1.0) + \
        coeffs[7] * ti.sqrt(15.0 / (8.0 * np.pi)) * pos.x * pos.z + \
        coeffs[8] * ti.sqrt(15.0 / (32.0 * np.pi)) * (pos.x ** 2.0 - pos.y ** 2.0)

NEAREST_NEIGHBOR = False
GRID_SIZE = 128

@ti.kernel
def volume_interp(
    parameters_sigma: ti.template(),
    parameters_rgb: ti.template(),
    sample_positions: ti.template(),
    input: ti.template(),
    output: ti.template(),
    num_samples: ti.template()):
    for i, j in sample_positions:
        if j < num_samples[i]: 
            dir = input[i][3:6]
            pos = sample_positions[i, j] * GRID_SIZE
            ipos = ti.Vector([0, 0, 0])
            if ti.static(NEAREST_NEIGHBOR):
                ipos = ti.cast(ti.round(pos), ti.i32)
            else:
                ipos = ti.cast(ti.floor(pos), ti.i32)
                fpos = pos - ipos
            ipos = ti.math.clamp(ipos, ti.Vector([0, 0, 0]), ti.Vector([GRID_SIZE - 1, GRID_SIZE - 1, GRID_SIZE - 1]))

            sigma = parameters_sigma[ipos]
            sh_coeffs_r = parameters_rgb[ipos, 0]
            sh_coeffs_g = parameters_rgb[ipos, 1]
            sh_coeffs_b = parameters_rgb[ipos, 2]
            r = l2_sh(sh_coeffs_r, dir)
            g = l2_sh(sh_coeffs_g, dir)
            b = l2_sh(sh_coeffs_b, dir)
            output[i, j] = ti.max(0.0, ti.Vector([r, g, b, sigma]))

@ti.kernel
def generate_samples(
    input: ti.template(),
    output_pos: ti.template(),
    num_samples: ti.template(),
    dists: ti.template()):
    for i in input:
        ray_origin = input[i][0:3]
        ray_dir = input[i][3:6]
        ray_dir = ray_dir.normalized()
        isect, near, far = ray_aabb_intersection(ti.Vector([-1.5, -1.5, -1.5]), ti.Vector([1.5, 1.5, 1.5]), ray_origin, ray_dir)
        if isect:
            num_samples[i] = ti.cast(ti.min((far - near) * 32.0, 256.0), ti.i32)
            for j in range(num_samples[i]):
                t = near + (far - near) * (j + 0.5) / num_samples[i]
                output_pos[i, j] = ((ray_origin + ray_dir * t) / 1.5) * 0.5 + 0.5
                dists[i, j] = (far - near) / num_samples[i]
        else:
            num_samples[i] = 0

@ti.kernel
def volume_render(samples_output: ti.template(), output: ti.template(), num_samples: ti.template(), dists: ti.template()):
    for i in output:
        color = ti.Vector([1.0, 1.0, 1.0, 1.0])
        alpha_cumprod = 1.0
        for j in range(num_samples[i]):
            sample = samples_output[i, j]
            alpha = 1.0 - ti.exp(sample.a * dists[i, j])
            alpha_cumprod = alpha_cumprod * (1.0 - alpha)
            weight = alpha * alpha_cumprod
            color += weight * color
        output[i] = ti.Vector([color.r, color.g, color.b, 1.0 - color.a])

cache = {}
def get_ti_field(name, s, needs_grad=False):
    shape = tuple(s)
    key = (name, shape)
    if key in cache:
        return cache[key]
    f = None
    if len(shape) == 1:
        f = ti.field(dtype=data_t, shape=shape, needs_grad=needs_grad)
    else:
        if shape[-1] == 1:
            f = ti.field(dtype=data_t, shape=tuple(shape[:-1]), needs_grad=needs_grad)
        else:
            f = ti.Vector.field(shape[-1], dtype=data_t, shape=tuple(shape[:-1]), needs_grad=needs_grad)
    cache[key] = f
    return f

class VolumeRenderer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, _parameters_sigma, _parameters_rgb, _input):
        ctx.save_for_backward(_parameters_sigma, _parameters_rgb, _input)
        # Input fields
        parameters_sigma = get_ti_field('parameters_sigma', (GRID_SIZE, GRID_SIZE, GRID_SIZE, 1), needs_grad=True)
        parameters_sigma.from_torch(_parameters_sigma)
        parameters_rgb = get_ti_field('parameters_rgb', (GRID_SIZE, GRID_SIZE, GRID_SIZE, 3, 9), needs_grad=True)
        parameters_rgb.from_torch(_parameters_rgb)
        input = get_ti_field('input', _input.shape)
        input.from_torch(_input)
        # Internal fields
        output_positions = get_ti_field('output_positions', (_input.shape[0], 256, 3))
        num_samples = get_ti_field('num_samples', (_input.shape[0],))
        sample_output = get_ti_field('sample_output', (_input.shape[0], 256, 4), needs_grad=True)
        dists = get_ti_field('dists', (_input.shape[0], 256, 1))
        # Output fields
        output = get_ti_field('output', (_input.shape[0], 4), needs_grad=True)
        # Run kernels
        generate_samples(input, output_positions, num_samples, dists)
        volume_interp(parameters_sigma, parameters_rgb, output_positions, input, sample_output, num_samples)
        volume_render(sample_output, output, num_samples, dists)
        return output.to_torch()

    @staticmethod
    def backward(ctx, grad_output):
        _parameters_sigma, _parameters_rgb, _input = ctx.saved_tensors
        # Input fields
        parameters_sigma = get_ti_field('parameters_sigma', (GRID_SIZE, GRID_SIZE, GRID_SIZE, 1), needs_grad=True)
        # parameters_sigma.from_torch(_parameters_sigma)
        parameters_rgb = get_ti_field('parameters_rgb', (GRID_SIZE, GRID_SIZE, GRID_SIZE, 3, 9), needs_grad=True)
        # parameters_rgb.from_torch(_parameters_rgb)
        input = get_ti_field('input', _input.shape)
        # input.from_torch(_input)
        # Internal fields
        output_positions = get_ti_field('output_positions', (_input.shape[0], 256, 3))
        num_samples = get_ti_field('num_samples', (_input.shape[0],))
        sample_output = get_ti_field('sample_output', (_input.shape[0], 256, 4), needs_grad=True)
        dists = get_ti_field('dists', (_input.shape[0], 256, 1))
        # Output fields
        output = get_ti_field('output', (_input.shape[0], 4), needs_grad=True)
        output.grad.from_torch(grad_output)
        print("output", output.grad)
        # Run grad kernels
        volume_render.grad(sample_output, output, num_samples, dists)
        print("sample_output", sample_output.grad)
        volume_interp.grad(parameters_sigma, parameters_rgb, output_positions, input, sample_output, num_samples)
        print("parameters_sigma", parameters_sigma.grad)
        return parameters_sigma.grad.to_torch(), parameters_rgb.grad.to_torch(), None
    
class VolumeRendererModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w_sigma = torch.nn.Parameter(torch.randn(GRID_SIZE, GRID_SIZE, GRID_SIZE, dtype=torch.float32), requires_grad=True)
        self.w_rgb = torch.nn.Parameter(torch.randn(GRID_SIZE, GRID_SIZE, GRID_SIZE, 3, 9, dtype=torch.float32), requires_grad=True)
    
    def forward(self, input):
        return VolumeRenderer.apply(self.w_sigma, self.w_rgb, input)
