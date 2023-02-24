import json
import taichi as ti
import numpy as np
from math_utils import ray_aabb_intersection

ti.init(arch=ti.vulkan, device_memory_GB=8)

data_t = ti.f32
data_vec3_t = ti.types.vector(3, data_t)
data_vec4_t = ti.types.vector(4, data_t)
sh_vec_t = ti.types.vector(9, data_t) # sigma + 3 * (9 SH coeffs)
pos_vec_t = ti.types.vector(3, data_t) # XYZ
output_vec_t = ti.types.vector(4, data_t) # RGB, alpha

GRID_SIZE = 128

set_name = "nerf_synthetic"
scene_name = "lego"
downscale = 2
image_w = 800 // downscale
image_h = 800 // downscale
num_pixels = image_w * image_h

learning_rate_sigma = 0.1
learning_rate_rgb = 0.01
iterations = 300000

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

@ti.kernel
def image_to_data(
  input_img : ti.types.ndarray(dtype=data_vec4_t),
  scaled_image : ti.types.ndarray(dtype=data_vec4_t),
  ray_o : ti.template(),
  ray_dir : ti.template(),
  output : ti.template(),
  fov_w : ti.f32,
  fov_h : ti.f32,
  origin : ti.types.vector(3, ti.f32),
  camera_mtx : ti.types.matrix(3, 3, ti.f32)):
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
    world_dir = camera_mtx @ view_dir
    ray_dir[ti.cast(i * image_h + j, dtype=ti.i32)] = world_dir
    ray_o[ti.cast(i * image_h + j, dtype=ti.i32)] = origin
    output[ti.cast(i * image_h + j, dtype=ti.i32)] = ti.Vector([scaled_image[i, j].x, scaled_image[i, j].y, scaled_image[i, j].z])

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

@ti.func
def trilerp(fout: ti.template(), f: ti.template(), ipos, fpos):
    for I in ti.static(ti.ndrange(2, 2, 2)):
        weight = (1 - I[0]) * (1.0 - fpos.x) + I[0] * fpos.x
        weight *= (1 - I[1]) * (1.0 - fpos.y) + I[1] * fpos.y
        weight *= (1 - I[2]) * (1.0 - fpos.z) + I[2] * fpos.z
        fout += f[ipos + I] * weight

@ti.func
def trilerp_rgb(fout: ti.template(), f: ti.template(), channel, ipos, fpos):
    for I in ti.static(ti.ndrange(2, 2, 2)):
        weight = (1 - I[0]) * (1.0 - fpos.x) + I[0] * fpos.x
        weight *= (1 - I[1]) * (1.0 - fpos.y) + I[1] * fpos.y
        weight *= (1 - I[2]) * (1.0 - fpos.z) + I[2] * fpos.z
        fout += f[ipos + I, channel] * weight

@ti.kernel
def volume_interp(
    parameters_sigma: ti.template(),
    parameters_rgb: ti.template(),
    sample_positions: ti.template(),
    ray_dirs: ti.template(),
    num_samples: ti.template(),
    sample_output: ti.template()):
    for i, j in sample_positions:
        if j < num_samples[i]: 
            dir = ray_dirs[i]
            pos = (sample_positions[i, j] / 3.0 + 0.5) * (GRID_SIZE - 1)
            ipos = ti.cast(ti.floor(pos), ti.i32)
            fpos = pos - ipos
            ipos = ti.math.clamp(ipos, ti.Vector([0, 0, 0]), ti.Vector([GRID_SIZE - 1, GRID_SIZE - 1, GRID_SIZE - 1]))

            sigma = 0.0
            trilerp(sigma, parameters_sigma, ipos, fpos)
            sh_coeffs_r = sh_vec_t(0.0)
            sh_coeffs_g = sh_vec_t(0.0)
            sh_coeffs_b = sh_vec_t(0.0)
            trilerp_rgb(sh_coeffs_r, parameters_rgb, 0, ipos, fpos)
            trilerp_rgb(sh_coeffs_g, parameters_rgb, 1, ipos, fpos)
            trilerp_rgb(sh_coeffs_b, parameters_rgb, 2, ipos, fpos)
            r = l2_sh(sh_coeffs_r, dir)
            g = l2_sh(sh_coeffs_g, dir)
            b = l2_sh(sh_coeffs_b, dir)
            sample_output[i, j] = ti.max(ti.Vector([r, g, b, sigma]), 0.0)

@ti.kernel
def generate_samples(
    ray_origin : ti.template(),
    ray_dirs: ti.template(),
    sample_pos: ti.template(),
    num_samples: ti.template(),
    dists: ti.template()):
    for i in ray_dirs:
        o = ray_origin[i]
        dir = ray_dirs[i].normalized()
        isect, near, far = ray_aabb_intersection(ti.Vector([-1.5, -1.5, -1.5]), ti.Vector([1.5, 1.5, 1.5]), o, dir)
        if isect:
            num_samples[i] = ti.cast(ti.min((far - near) * 32.0, 256.0), ti.i32)
            for j in range(num_samples[i]):
                t = near + (far - near) * (j + 0.5) / num_samples[i]
                sample_pos[i, j] = o + dir * t
                dists[i, j] = (far - near) / num_samples[i]
        else:
            num_samples[i] = 0

@ti.kernel
def volume_render(samples_output: ti.template(), output: ti.template(), alpha_T : ti.template(), num_samples: ti.template(), dists: ti.template()):
    for i in output:
        # color = ti.Vector([1.0, 1.0, 1.0, 1.0])
        n_samples = ti.cast(num_samples[i], ti.i32)
        for j in range(n_samples):
            sample = samples_output[i, j]
            alpha = 1.0 - ti.exp(-sample.a * dists[i, j])
            T_ = 1.0
            if j > 0:
                T_ = alpha_T[i, j - 1]
            w = alpha * T_
            alpha_T[i, j] = (1.0 - alpha) * T_
            output[i] += ti.Vector([sample.r, sample.g, sample.b, 1.0]) * w

@ti.kernel
def nerf_loss_fn(output: ti.template(), target: ti.template(), loss: ti.template()):
    for i in output:
        sq_diff = (output[i].rgb - target[i]) ** 2.0
        o = output[i].a + eps
        # encourage opacity to be either 0 or 1 to avoid floater
        loss[None] += 1e-3 * (-o * ti.log(o))
        loss[None] += sq_diff.x + sq_diff.y + sq_diff.z

@ti.kernel
def randomize_parameters(parameters: ti.template(), scale: ti.f32, mean: ti.f32):
    for i in ti.grouped(parameters):
        parameters[i] += (ti.random() * scale - scale * 0.5) + mean

beta1 = 0.9
beta2 = 0.99
eps=1e-15

@ti.kernel
def update_parameters(parameters: ti.template(), grad_1st_moments: ti.template(), grad_2nd_moments: ti.template(), learning_rate: ti.f32):
    for I in ti.grouped(parameters):
        g = parameters.grad[I]
        m = beta1 * grad_1st_moments[I] + (1.0 - beta1) * g
        v = beta2 * grad_2nd_moments[I] + (1.0 - beta2) * g * g
        grad_1st_moments[I] = m
        grad_2nd_moments[I] = v
        m_hat = m / (1.0 - beta1)
        v_hat = v / (1.0 - beta2)
        parameters[I] -= learning_rate * m_hat / (ti.sqrt(v_hat) + eps)

desc = load_desc_from_json(set_name + "/" + scene_name + "/transforms_train.json")
desc_test = load_desc_from_json(set_name + "/" + scene_name + "/transforms_test.json")

ray_dir = ti.Vector.field(3, data_t, (num_pixels,))
ray_o = ti.Vector.field(3, data_t, (num_pixels,))
target_data = ti.Vector.field(3, data_t, (num_pixels,))
sample_pos = ti.Vector.field(3, dtype=data_t, shape=(num_pixels, 256))
num_samples = ti.field(dtype=ti.i32, shape=num_pixels)
dists = ti.field(dtype=data_t, shape=(num_pixels, 256))

parameters_sigma = ti.field(dtype=data_t, shape=(GRID_SIZE, GRID_SIZE, GRID_SIZE), needs_grad=True)
parameters_rgb = ti.Vector.field(9, dtype=data_t, shape=(GRID_SIZE, GRID_SIZE, GRID_SIZE, 3), needs_grad=True)
alpha_T = ti.field(dtype=data_t, shape=(num_pixels, 256), needs_grad=True)
sample_output = ti.Vector.field(4, dtype=data_t, shape=(num_pixels, 256), needs_grad=True)
output = ti.Vector.field(4, dtype=data_t, shape=num_pixels, needs_grad=True)
loss = ti.field(dtype=data_t, shape=(), needs_grad=True)

grad_1st_moments_sigma = ti.field(dtype=data_t, shape=(GRID_SIZE, GRID_SIZE, GRID_SIZE))
grad_2nd_moments_sigma = ti.field(dtype=data_t, shape=(GRID_SIZE, GRID_SIZE, GRID_SIZE))
grad_1st_moments_rgb = ti.Vector.field(9, dtype=data_t, shape=(GRID_SIZE, GRID_SIZE, GRID_SIZE, 3))
grad_2nd_moments_rgb = ti.Vector.field(9, dtype=data_t, shape=(GRID_SIZE, GRID_SIZE, GRID_SIZE, 3))


def generate_data(desc, i):
  img = desc["frames"][i]
  file_name = set_name + "/" + scene_name + "/" + img["file_path"] + ".png"
  input_image = np.ascontiguousarray(ti.tools.imread(file_name))
  mtx = np.array(img["transform_matrix"])
  scaled_image = ti.Vector.ndarray(4, data_t, (image_w, image_h))
  origin = mtx[:3,-1]
  image_to_data(
     input_image, 
     scaled_image, 
     ray_o,
     ray_dir, 
     target_data, 
     float(desc["camera_angle_x"]), 
     float(desc["camera_angle_x"]),
     ti.Vector(origin, dt=data_t),
     ti.Matrix(mtx, dt=data_t))
  return ray_o

# window = ti.ui.Window("Nerf", (image_w, image_h))
# canvas = window.get_canvas()

randomize_parameters(parameters_sigma, 0.1, 0.05)
randomize_parameters(parameters_rgb, 0.1, 0.0)

def reset():
    ray_dir.fill(0.0)
    target_data.fill(0.0)
    sample_pos.fill(0.0)
    num_samples.fill(0)
    dists.fill(0.0)

    alpha_T.fill(0.0)
    sample_output.fill(0.0)
    output.fill(0.0)
    loss[None] = 0.0

def zero_grad():
    parameters_sigma.grad.fill(0.0)
    parameters_rgb.grad.fill(0.0)
    alpha_T.grad.fill(0.0)
    sample_output.grad.fill(0.0)
    output.grad.fill(0.0)

num_tests = 5
test_indices = np.random.choice(len(desc_test["frames"]), size=(num_tests))
print(test_indices)

X_ray_o = []
X_ray_dir = []
Y = []
for i in range(len(desc["frames"])):
  print("load img", i)
  generate_data(desc, i)
  
  X_ray_o.append(ray_o.to_numpy())
  X_ray_dir.append(ray_dir.to_numpy())
  Y.append(target_data.to_numpy())
X_ray_o = np.vstack(X_ray_o)
X_ray_dir = np.vstack(X_ray_dir)
Y = np.vstack(Y)

arr = np.arange(num_pixels * len(desc["frames"])).reshape((len(desc["frames"]), num_pixels))
indices = np.random.permutation(arr)

epoch = 0
for iter in range(len(indices) * 10):
    # Training
    reset()
    zero_grad()

    b = np.random.randint(0, indices.shape[0])
    Xbatch_o = X_ray_o[indices[b]]
    Xbatch_dir = X_ray_dir[indices[b]]
    Ybatch = Y[indices[b]]

    ray_o.from_numpy(Xbatch_o)
    ray_dir.from_numpy(Xbatch_dir)
    target_data.from_numpy(Ybatch)

    generate_samples(ray_o, ray_dir, sample_pos, num_samples, dists)        
    with ti.ad.Tape(loss):
        volume_interp(parameters_sigma, parameters_rgb, sample_pos, ray_dir, num_samples, sample_output)
        volume_render(sample_output, output, alpha_T, num_samples, dists)
        nerf_loss_fn(output, target_data, loss)
    # print(f"epoch={epoch} iter={i} loss={loss[None]}")
    # print("output grad", output.grad.to_numpy().max())
    # print("sample_output grad", sample_output.grad.to_numpy().max())
    # print("parameters_sigma grad", parameters_sigma.grad.to_numpy().max())
    update_parameters(parameters_sigma, grad_1st_moments_sigma, grad_2nd_moments_sigma, learning_rate_sigma)
    update_parameters(parameters_rgb, grad_1st_moments_rgb, grad_2nd_moments_rgb, learning_rate_rgb)
    print(f"iter={iter} training loss={loss[None]}")

    # Testing
    if iter % 100 == 0:
        print("Testing epoch", epoch)
        average_test_loss = 0.0
        for i in test_indices:
            reset()
            generate_data(desc_test, i)
            generate_samples(ray_o, ray_dir, sample_pos, num_samples, dists)        
            volume_interp(parameters_sigma, parameters_rgb, sample_pos, ray_dir, num_samples, sample_output)
            volume_render(sample_output, output, alpha_T, num_samples, dists)
            nerf_loss_fn(output, target_data, loss)
            average_test_loss += loss[None]
            output_shaped = output.to_numpy().reshape((image_w, image_h, 4))
            ti.tools.imwrite(output_shaped, f"out/test_{i}_epoch_{epoch}.png")
        print("Average test loss", average_test_loss / num_tests)
        epoch += 1

