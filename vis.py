import torch
import open3d as o3d
import numpy as np

X = torch.vstack(torch.load("input_samples.th")).cpu()
Y = torch.vstack(torch.load("output_samples.th")).cpu()

# view_samples = 10000

non_zero_samples_r = Y[:,0].ge(0.5)
non_zero_samples_g = Y[:,1].le(0.01)
non_zero_samples_b = Y[:,2].le(0.01)
non_zero_samples = non_zero_samples_r.logical_and(non_zero_samples_g).logical_and(non_zero_samples_b)

print(non_zero_samples)

X = X[non_zero_samples]
Y = Y[non_zero_samples]

view_samples = Y.shape[0]

indicies = torch.randperm(X.shape[0])[:view_samples]

Xp = X[indicies]
Yp = Y[indicies]

line_start = Xp[:,0:3] + Xp[:,3:] * 2.0
line_end = Xp[:,0:3] + Xp[:,3:] * 6.0

vertices = torch.cat((line_start, line_end), dim=0).detach().numpy()
indicies = np.array(range(view_samples)).reshape((view_samples,1))
indicies = np.hstack((indicies, indicies + view_samples))
colors = Yp.detach().numpy()

print(vertices.shape)
print(indicies.shape)
print(colors.shape)

line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(vertices),
    lines=o3d.utility.Vector2iVector(indicies),
)
line_set.colors = o3d.utility.Vector3dVector(colors)

point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices[:view_samples]))
point_cloud.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([line_set, point_cloud])
