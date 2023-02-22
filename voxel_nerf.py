import torch
import taichi as ti
from math_utils import ray_aabb_intersection

data_t = ti.f32
input_vec_t = ti.types.vector(1 + 3 * 9, data_t) # sigma + 3 * (9 SH coeffs)
pos_vec_t = ti.types.vector(3, data_t) # XYZ
output_vec_t = ti.types.vector(4, data_t) # RGB, alpha

@ti.func
def l2_sh(coeffs, pos):
    return coeffs[0] + \
        coeffs[1] * pos[0] + \
        coeffs[2] * pos[1] + \
        coeffs[3] * pos[2] + \
        coeffs[4] * pos[0] * pos[1] + \
        coeffs[5] * pos[0] * pos[2] + \
        coeffs[6] * pos[1] * pos[2] + \
        coeffs[7] * pos[0] * pos[0] + \
        coeffs[8] * pos[1] * pos[1] + \
        coeffs[9] * pos[2] * pos[2]

NEAREST_NEIGHBOR = False

@ti.kernel
def volume_interp(
    grid_size: ti.types.vector(3, ti.i32),
    parameters: ti.template(),
    sample_positions: ti.template(),
    input: ti.template(),
    output: ti.template(),
    num_samples: ti.template()):
    for i, j in sample_positions:
        if j < num_samples[i]: 
            dir = input[i][3:6]
            pos = sample_positions[i, j] * grid_size
            ipos = ti.Vector([0, 0, 0])
            if ti.static(NEAREST_NEIGHBOR):
                ipos = ti.cast(ti.round(pos), ti.i32)
            else:
                ipos = ti.cast(ti.floor(pos), ti.i32)
                fpos = pos - ipos
            ipos = ti.math.clamp(ipos, ti.Vector([0, 0, 0]), grid_size)

            param = parameters[ipos]
            sigma = param[3 * 9]
            sh_coeffs_r = param[0 * 9 : 1 * 9]
            sh_coeffs_g = param[1 * 9 : 2 * 9]
            sh_coeffs_b = param[2 * 9 : 3 * 9]
            r = l2_sh(sh_coeffs_r, dir)
            g = l2_sh(sh_coeffs_g, dir)
            b = l2_sh(sh_coeffs_b, dir)
            output[i, j] = ti.Vector([r, g, b, sigma])

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
        if num_samples[i] > 0:
            for j in range(num_samples[i]):
                sample = samples_output[i, j]
                alpha = 1.0 - ti.exp(sample.a * dists[i, j])
                color = color * alpha + sample * (1.0 - alpha)
        output[i] = color

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
    def forward(ctx, _parameters, _input):
        ctx.save_for_backward(_parameters, _input)
        # Input fields
        parameters = get_ti_field('parameters', _parameters.shape, needs_grad=True)
        parameters.from_torch(_parameters)
        input = get_ti_field('input', _input.shape)
        input.from_torch(_input)
        # Internal fields
        output_positions = get_ti_field('output_positions', (_input.shape[0], 256, 3))
        num_samples = get_ti_field('num_samples', (_input.shape[0],))
        sample_output = get_ti_field('sample_output', (_input.shape[0], 256, 4), needs_grad=True)
        dists = get_ti_field('dists', (_input.shape[0], 256, 1), needs_grad=True)
        # Output fields
        _output = torch.zeros((_input.shape[0], 4), dtype=_input.dtype, device=_input.device, requires_grad=True)
        output = get_ti_field('output', (_input.shape[0], 4), needs_grad=True)
        grid_size = _parameters.shape[0]
        # Run kernels
        generate_samples(input, output_positions, num_samples, dists)
        print("what")
        volume_interp(ti.Vector([grid_size, grid_size, grid_size]), parameters, output_positions, input, sample_output, num_samples)
        print("the")
        volume_render(sample_output, num_samples, output, dists)
        print("doodoo")
        output.to_torch(_output)
        return _output

    @staticmethod
    def backward(ctx, grad_output):
        _parameters, _input = ctx.saved_tensors
        # Input fields
        parameters = get_ti_field('parameters', _parameters.shape, needs_grad=True)
        parameters.from_torch(_parameters)
        input = get_ti_field('input', _input.shape)
        input.from_torch(_input)
        # Internal fields
        output_positions = get_ti_field('output_positions', (_input.shape[0], 256, 3))
        num_samples = get_ti_field('num_samples', (_input.shape[0],))
        sample_output = get_ti_field('sample_output', (_input.shape[0], 256, 4), needs_grad=True)
        # Output fields
        output = get_ti_field('output', (_input.shape[0], 4), needs_grad=True)
        output.grad.from_torch(grad_output)
        grid_size = _parameters.shape[0]
        # Run grad kernels
        volume_render.grad(sample_output, num_samples, output)
        volume_interp.grad(ti.Vector([grid_size, grid_size, grid_size]), parameters, output_positions, input, sample_output)
        _parameters_grad = torch.zeros(_parameters.shape, dtype=_parameters.dtype, device=_parameters.device, requires_grad=True)
        parameters.grad.to_torch(_parameters_grad)
        return _parameters_grad, None, None
    
class VolumeRendererModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(128, 128, 128, 1 + 3 * 9, dtype=torch.float32))
        self.W.requires_grad = True
    
    def forward(self, input):
        return VolumeRenderer.apply(self.W, input)
