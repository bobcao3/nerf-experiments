import taichi as ti

eps = 1e-4
inf = 1e10

@ti.func
def ray_aabb_intersection(box_min, box_max, o, d):
    near_int = -inf
    far_int = inf

    for i in ti.static(range(3)):
        if d[i] == 0:
            if o[i] < box_min[i] or o[i] > box_max[i]:
                intersect = 0
        else:
            i1 = (box_min[i] - o[i]) / d[i]
            i2 = (box_max[i] - o[i]) / d[i]

            new_far_int = ti.max(i1, i2)
            new_near_int = ti.min(i1, i2)

            far_int = ti.min(new_far_int, far_int)
            near_int = ti.max(new_near_int, near_int)

    intersect = near_int <= far_int
    return intersect, near_int, far_int