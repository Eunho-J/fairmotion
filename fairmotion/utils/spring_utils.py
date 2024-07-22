# https://github.com/orangeduck/Motion-Matching/blob/main/spring.h

PIf = 3.14159265358979323846
LN2f = 0.69314718056
from numpy.core.fromnumeric import repeat
from scipy.spatial.transform import Rotation
import numpy as np


def fast_negexp(x):
    return 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)


def halflife_to_damping(halflife, eps=1e-5):
    return (4.0 * LN2f) / (halflife + eps)


def simple_spring_damper_implicit(
    x,  # vector
    v,  # vector
    x_goal,  # const float
    halflife,  # const float
    dt,  # const float
):
    y = halflife_to_damping(halflife) / 2.0
    j0 = x - x_goal
    j1 = v + j0 * y
    eydt = fast_negexp(y * dt)

    x = eydt * (j0 + j1 * dt) + x_goal
    v = eydt * (v - j1 * y * dt)
    return x, v


def simple_spring_damper_implicit_a(
    x,  # vector
    v,  # vector
    a,  # vector
    v_goal,  # vector
    halflife,  # float
    dt,  # float
):
    y = halflife_to_damping(halflife) / 2.0
    j0 = v - v_goal
    j1 = a + j0 * y
    eydt = fast_negexp(y * dt)

    x = (
        eydt * (((-j1) / (y * y)) + ((-j0 - j1 * dt) / y))
        + (j1 / (y * y))
        + j0 / y
        + v_goal * dt
        + x
    )
    v = eydt * (j0 + j1 * dt) + v_goal
    a = eydt * (a - j1 * y * dt)

    return x, v, a


def simple_spring_damper_implicit_quat(
    x,  # quat&
    v,  # vec3&
    x_goal,  # const quat
    halflife,  # const float
    dt,  # const float
):
    y = halflife_to_damping(halflife) / 2.0
    j0 = Rotation.from_matrix(x @ np.linalg.inv(x_goal)).as_rotvec()
    # j0 = quat_to_scaled_angle_axis(quat_abs(quat_mul(x, quat_inv(x_goal))))
    j1 = v + j0 * y

    eydt = fast_negexp(y * dt)

    x = Rotation.from_rotvec(eydt * (j0 + j1 * dt)).as_matrix() @ x_goal
    # x = quat_mul(quat_from_scaled_angle_axis(eydt*(j0 + j1*dt)), x_goal)
    v = eydt * (v - j1 * y * dt)
    return x, v


def decay_spring_damper_implicit_vector(
    x, v, halflife, dt  # vec3&  # vec3&  # const float  # const float
):
    y = halflife_to_damping(halflife) / 2.0
    j1 = v + x * y  # [3]
    eydt = fast_negexp(y * dt)

    x = eydt * (x + j1 * dt)
    v = eydt * (v - j1 * y * dt)
    return x, v


def inertialize_transition(
    off_x,  # vec3&
    off_v,  # vec3&
    src_x,  # const vec3
    src_v,  # const vec3
    dst_x,  # const vec3
    dst_v,  # const vec3
):
    off_x = (src_x + off_x) - dst_x
    off_v = (src_v + off_v) - dst_v
    return off_x, off_v


def inertialize_update(
    out_x,  # vec3&
    out_v,  # vec3&
    off_x,  # vec3&
    off_v,  # vec3&
    in_x,  # const vec3
    in_v,  # const vec3
    halflife,  # const float
    dt,  # const float
):
    off_x, off_v = decay_spring_damper_implicit_vector(off_x, off_v, halflife, dt)
    out_x = in_x + off_x
    out_v = in_v + off_v
    return out_x, out_v, off_x, off_v


def spring_character_predict_pos(x, v, count, x_goal, halflife, dt):
    px = x[None, ...].repeat(count, 0)
    pv = v[None, ...].repeat(count, 0)

    for i in range(count):
        px[i], pv[i] = simple_spring_damper_implicit(
            px[i], pv[i], x_goal, halflife, (i + 1) * dt
        )

    return px, pv


def spring_character_predict_vel(
    x,  # vector
    v,  # vector
    a,  # vector
    count,  # int
    v_goal,  # vector
    halflife,  # float
    dt,  # float
):
    px = x[None, ...].repeat(count, 0)
    pv = v[None, ...].repeat(count, 0)
    pa = a[None, ...].repeat(count, 0)

    for i in range(count):
        px[i], pv[i], pa[i] = simple_spring_damper_implicit_a(
            px[i], pv[i], pa[i], v_goal, halflife, (i + 1) * dt
        )

    return px, pv, pa


def spring_character_predict_quat(
    x,  # quat
    v,  # vector
    count,  # int
    x_goal,  # quat
    halflife,  # float
    dt,  # float
):
    px = x[None, ...].repeat(count, 0)
    pv = v[None, ...].repeat(count, 0)

    for i in range(count):
        px[i], pv[i] = simple_spring_damper_implicit_quat(
            px[i], pv[i], x_goal, halflife, (i + 1) * dt
        )

    return px, pv
