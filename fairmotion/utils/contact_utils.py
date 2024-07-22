import numpy as np
from scipy.spatial.transform import Rotation
from IPython import embed

from fairmotion.utils.spring_utils import inertialize_update, inertialize_transition

# from fairmotion.core import motion as motion_classes
from fairmotion.ops import motion as motion_ops
from fairmotion.ops import conversions

# from fairmotion.ops.spring import *
import scipy.ndimage as ndimage
from utils.data_utils import safe_normalize_to_one as normalize  # shortcut name alias


def get_foot_indices(
    skel, toe_only=False, skip_if_missing=False, print_if_missing=True
):
    joint_names = [j.name for j in skel.joints]
    lte = "LeftToeBase_End" if "LeftToeBase_End" in joint_names else "LeftToe_End"
    rte = "RightToeBase_End" if "RightToeBase_End" in joint_names else "RightToe_End"
    if not (lte in joint_names) or not (rte in joint_names):
        if print_if_missing:
            print("get_foot_indices: no such feet name")
        if not skip_if_missing:
            embed()
        return

    lt = skel.get_joint(lte).parent_joint
    rt = skel.get_joint(rte).parent_joint
    if toe_only:
        return [skel.get_index_joint(lt), skel.get_index_joint(rt)]
    else:
        return [
            skel.get_index_joint(lt),
            skel.get_index_joint(lte),
            skel.get_index_joint(rt),
            skel.get_index_joint(rte),
        ]


def get_default_fc_thres(skel, foot_indices):
    # p_thres = [2* tpose.get_transform(f_idx, local=False)[1, 3] for f_idx in foot_indices]
    # p_thres = [4.2] * len(foot_indices)
    """assumption: skel is in tpose"""
    """ CAUTION: should call this after motion normalization"""
    # ex) stunt_loco: [left: -16, right: +16], thus max(h,0)+4: [4, 20]
    # before normalize: p_thres:  [4.0, 4.0, 20.0463, 19.04634]    v_thres: [0.4, 0.4, 0.4, 0.4]
    # after normalize: p_thres:  [6.1102, 5.1563, 6.1123, 7.0662]   v_thres: [0.4, 0.4, 0.4, 0.4]

    p_thres = [
        max(skel.get_joint(fi).xform_global[1, 3], 0) + 4.0 for fi in foot_indices
    ]
    # v_thres = [0.18] * len(p_thres)
    v_thres = [0.4] * len(p_thres)
    return p_thres, v_thres


def get_foot_contact(poses, foot_indices, p_thres, v_thres, use_filter=True):

    if (p_thres == None) or (v_thres == None):
        try:
            p_thres_auto, v_thres_auto = get_default_fc_thres(
                poses[0].skel, foot_indices
            )
            if p_thres is None:
                p_thres = p_thres_auto
            if v_thres is None:
                v_thres = v_thres_auto
        except:
            print("get foot contact")
            from IPython import embed

            embed()

    # print("p_thres: ", p_thres, "\tv_thres:", v_thres)
    contacts = np.zeros((len(poses), len(foot_indices)), dtype=bool)
    for frame, pose in enumerate(poses):
        foot_position = np.array(
            [pose.get_transform(f_idx, local=False)[:3, 3] for f_idx in foot_indices]
        )
        ct_p = [fp[1] < p_thres_i for fp, p_thres_i in zip(foot_position, p_thres)]
        if frame > 0:
            prev_pose = poses[frame - 1]
            prev_foot_position = np.array(
                [
                    prev_pose.get_transform(f_idx, local=False)[:3, 3]
                    for f_idx in foot_indices
                ]
            )
            ct_v = [
                np.linalg.norm(fp - prev_fp) * 30.0 / 100.0 < v_thres_i
                for fp, prev_fp, v_thres_i in zip(
                    foot_position, prev_foot_position, v_thres
                )
            ]
            contacts[frame] = [ct_p_i & ct_v_i for ct_p_i, ct_v_i in zip(ct_p, ct_v)]
        else:
            contacts[frame] = ct_p

    if use_filter:
        for ci in range(contacts.shape[1]):
            contacts[:, ci] = ndimage.median_filter(
                contacts[:, ci], size=4, mode="nearest"
            )

    return contacts


def get_foot_contact_ratio(poses, foot_indices, p_thres=None, v_thres=None, raw=True):
    foot_contact_bool = get_foot_contact(poses, foot_indices, p_thres, v_thres)
    n_frames = foot_contact_bool.shape[0]
    if raw:  # do not avg over each side(left, right), just use raw value for all joints
        foot_contact_ratio = foot_contact_bool.astype("float32")
    else:
        n_feet = len(foot_indices)
        foot_contact_ratio = (
            np.sum(foot_contact_bool.reshape(n_frames, int(n_feet // 2), 2), axis=-1)
            / 2.0
        )

    return foot_contact_ratio


def get_all_joint_ground_contact(motion, p_thres=5, v_thres=0.4):
    positions = motion.positions(local=False)
    positions_y = positions[..., 1]
    velocities = np.linalg.norm(positions[1:] - positions[:-1], axis=-1) * 30.0 / 100.0
    contact = np.zeros((motion.num_frames(), motion.skel.num_joints()))
    contact[0] = positions_y[0] < p_thres
    contact[1:] = (positions_y[1:] < p_thres) & (velocities < v_thres)
    return contact


# Rotate a joint to look toward some
# given target position
def ik_look_at(
    bone_rotation,  # quat&
    global_parent_rotation,  # const quat
    global_rotation,  # const quat
    global_position,  # const vec3
    child_position,  # const vec3
    target_position,  # const vec3
    eps=1e-5,  # const float
):
    curr_dir = normalize(child_position - global_position)
    targ_dir = normalize(target_position - global_position)

    if np.abs(1.0 - np.dot(curr_dir, targ_dir) > eps):
        curr_to_target = Rotation.from_rotvec(np.cross(curr_dir, targ_dir))
        bone_rotation = (
            Rotation.inv(global_parent_rotation) * curr_to_target * global_rotation
        )

    return bone_rotation.as_matrix()


def ik_look_at_idx(
    pose,
    parent_id,
    joint_id,
    child_id,
    target_position,  # const vec3
    eps=1e-5,  # const float
):
    parent_global_T = pose.get_transform(parent_id, local=False)
    parent_global_R = Rotation.from_matrix(parent_global_T[:3, :3])

    joint_global_T = pose.get_transform(joint_id, local=False)
    joint_global_R = Rotation.from_matrix(joint_global_T[:3, :3])
    joint_global_p = joint_global_T[:3, 3]

    child_global_T = pose.get_transform(child_id, local=False)
    child_global_p = child_global_T[:3, 3]

    joint_local_R = Rotation.from_matrix(
        pose.get_transform(joint_id, local=True)[:3, :3]
    )

    return ik_look_at(
        joint_local_R,
        parent_global_R,
        joint_global_R,
        joint_global_p,
        child_global_p,
        target_position,
        eps,
    )


# Basic two-joint IK in the style of https://theorangeduck.com/page/simple-two-joint
# Here I add a basic "forward vector" which acts like a kind of pole-vetor
# to control the bending direction
def ik_two_bone(
    bone_joint,  # const vec3
    bone_mid,  # const vec3
    bone_end,  # const vec3
    target,  # const vec3
    fwd,  # const vec3
    bone_joint_gr,  # const quat
    bone_mid_gr,  # const quat
    bone_par_gr,  # const quat
    max_length_buffer,  # const float
):
    from utils.data_utils import (
        safe_normalize_to_one as normalize,
    )  # shortcut name alias

    max_extension = (
        np.linalg.norm(bone_joint - bone_mid)
        + np.linalg.norm(bone_mid - bone_end)
        - max_length_buffer
    )

    target_clamp = target  # [vec3]
    if np.linalg.norm(target - bone_joint) > max_extension:
        target_clamp = bone_joint + max_extension * normalize(target - bone_joint)

    axis_dwn = normalize(bone_end - bone_joint)  # TODO
    fwd = normalize(bone_mid - bone_joint)
    axis_rot = normalize(np.cross(axis_dwn, fwd))

    a = bone_joint
    b = bone_mid
    c = bone_end
    t = target_clamp

    lab = np.linalg.norm(b - a)
    lcb = np.linalg.norm(b - c)
    lat = np.linalg.norm(t - a)

    ac_ab_0 = np.arccos(np.clip(np.dot(normalize(c - a), normalize(b - a)), -1.0, 1.0))
    ba_bc_0 = np.arccos(np.clip(np.dot(normalize(a - b), normalize(c - b)), -1.0, 1.0))

    ac_ab_1 = np.arccos(
        np.clip((lab * lab + lat * lat - lcb * lcb) / (2.0 * lab * lat), -1.0, 1.0)
    )
    ba_bc_1 = np.arccos(
        np.clip((lab * lab + lcb * lcb - lat * lat) / (2.0 * lab * lcb), -1.0, 1.0)
    )

    r0 = Rotation.from_rotvec((ac_ab_1 - ac_ab_0) * axis_rot)
    r1 = Rotation.from_rotvec((ba_bc_1 - ba_bc_0) * axis_rot)

    c_a = normalize(bone_end - bone_joint)
    t_a = normalize(target_clamp - bone_joint)

    ac_at_0 = np.arccos(np.clip(np.dot(c_a, t_a), -1.0, 1.0))
    r2 = Rotation.from_rotvec(ac_at_0 * normalize(np.cross(c_a, t_a)))

    bone_joint_lr = Rotation.inv(bone_par_gr) * (r2 * (r0 * bone_joint_gr))

    bone_mid_lr = Rotation.inv(bone_joint_gr) * (r1 * bone_mid_gr)

    return bone_joint_lr.as_matrix(), bone_mid_lr.as_matrix()


def ik_two_bone_idx(
    pose,
    parent_id,
    joint_id,
    mid_id,
    end_id,
    target,  # const vec3
    fwd,  # const vec3
    max_length_buffer,  # const float
):
    parent_global_T = pose.get_transform(parent_id, local=False)
    parent_global_R = Rotation.from_matrix(parent_global_T[:3, :3])

    joint_global_T = pose.get_transform(joint_id, local=False)
    joint_global_R = Rotation.from_matrix(joint_global_T[:3, :3])
    joint_global_p = joint_global_T[:3, 3]

    mid_global_T = pose.get_transform(mid_id, local=False)
    mid_global_R = Rotation.from_matrix(mid_global_T[:3, :3])
    mid_global_p = mid_global_T[:3, 3]

    end_global_T = pose.get_transform(end_id, local=False)
    end_global_p = end_global_T[:3, 3]

    return ik_two_bone(
        joint_global_p,
        mid_global_p,
        end_global_p,
        target,
        fwd,
        joint_global_R,
        mid_global_R,
        parent_global_R,
        max_length_buffer,
    )


# https:#github.com/orangeduck/Motion-Matching/blob/47f4adbec4f2ac1c282f996f7e02ae97e64772c0/controller.cpp
class Contact:
    # static class variables
    meter_conversion_scale = 100
    ik_max_length_buffer = 0.015 * meter_conversion_scale
    # ik_unlock_radius = 0.10 * meter_conversion_scale
    ik_unlock_radius = 0.10 * meter_conversion_scale
    ik_blending_halflife = 0.1
    # contact_label_thres = 0.6
    contact_label_thres = 0.75

    # self.ik_max_length_buffer = 0.015 * meter_conversion_scale
    # self.ik_unlock_radius = 0.20 * meter_conversion_scale
    # self.ik_blending_halflife = 0.1

    def __init__(self, ik_toe_length=5, ik_foot_height=1) -> None:
        self.ik_enabled = True
        self.ik_toe_length = ik_toe_length
        self.ik_foot_height = ik_foot_height

    def contact_ratio_to_bool_state(self, contact_state):
        if type(contact_state) == bool:
            return contact_state
        else:
            return contact_state > self.contact_label_thres  # float, np.float, ...

    def reset(
        self, input_contact_position, input_contact_velocity, input_contact_state
    ):
        self.contact_state = False  # input_contact_state
        self.contact_lock = False
        self.contact_position = input_contact_position  # vec3&
        self.contact_velocity = input_contact_velocity  # vec3&
        self.contact_point = input_contact_position  # vec3&
        self.contact_target = input_contact_position  # vec3&
        self.contact_offset_position = np.zeros(3)  # vec3&
        self.contact_offset_velocity = np.zeros(3)  # vec3&
        self.contact_log = [self.contact_ratio_to_bool_state(input_contact_state)]

        self.contact_point_log = [self.contact_point]
        self.contact_position_log = [self.contact_position]

    def update(
        self,
        input_contact_position,  # const vec3
        input_contact_state,  # const bool
        dt,  # const float
        eps=1e-8,  # const float
        debug_frame=None,
    ):
        unlock_radius = self.ik_unlock_radius
        foot_height = self.ik_foot_height
        halflife = self.ik_blending_halflife
        input_contact_state = self.contact_ratio_to_bool_state(input_contact_state)

        # First compute the input contact position velocity via finite difference
        input_contact_velocity = (input_contact_position - self.contact_target) / (
            dt + eps
        )  # [3]
        self.contact_target = input_contact_position

        # Update the inertializer to tick forward in time
        (
            self.contact_position,
            self.contact_velocity,
            self.contact_offset_position,
            self.contact_offset_velocity,
        ) = inertialize_update(
            self.contact_position,
            self.contact_velocity,
            self.contact_offset_position,
            self.contact_offset_velocity,
            # If locked we feed the contact point and zero velocity,
            # otherwise we feed the input from the animation
            self.contact_point if self.contact_lock else input_contact_position,
            np.zeros(3) if self.contact_lock else input_contact_velocity,
            halflife,
            dt,
        )

        # If the contact point is too far from the current input position
        # then we need to unlock the contact
        unlock_contact = self.contact_lock and (
            np.linalg.norm(self.contact_point - input_contact_position) > unlock_radius
        )
        if debug_frame is not None:
            # if (not self.contact_state) and input_contact_state:
            #     print(debug_frame, '/', np.linalg.norm(self.contact_point - input_contact_position))
            if unlock_contact:
                print(
                    debug_frame,
                    "/ unlock: ",
                    np.linalg.norm(self.contact_point - input_contact_position),
                )

        # If the contact was previously inactive but is now active we
        # need to transition to the locked contact state
        if (not self.contact_state) and input_contact_state:
            # Contact point is given by the current position of
            # the foot projected onto the ground plus foot height
            self.contact_lock = True
            self.contact_point = self.contact_position
            self.contact_point[1] = (
                self.contact_position[1] + foot_height
            ) / 2.0  # foot_height

            self.contact_offset_position, self.contact_offset_velocity = (
                inertialize_transition(
                    self.contact_offset_position,
                    self.contact_offset_velocity,
                    input_contact_position,
                    input_contact_velocity,
                    self.contact_point,
                    np.zeros(3),
                )
            )
            # self.contact_offset_position = self.contact_offset_position + input_contact_position - self.contact_point
            # self.contact_offset_velocity = self.contact_offset_velocity + input_contact_velocity - np.zeros(3)

            if debug_frame is not None:
                print(
                    debug_frame,
                    "/ lock ",
                    self.contact_point,
                    self.contact_offset_position,
                    self.contact_offset_velocity,
                )

        # Otherwise if we need to unlock or we were previously in
        # contact but are no longer we transition to just taking
        # the input position as-is
        elif (
            self.contact_lock and self.contact_state and (not input_contact_state)
        ) or (unlock_contact):
            self.contact_lock = False

            self.contact_offset_position, self.contact_offset_velocity = (
                inertialize_transition(
                    self.contact_offset_position,
                    self.contact_offset_velocity,
                    self.contact_point,
                    np.zeros(3),
                    input_contact_position,
                    input_contact_velocity,
                )
            )
            # self.contact_offset_position = self.contact_offset_position + self.contact_point - input_contact_position
            # self.contact_offset_velocity = self.contact_offset_velocity + np.zeros(3) - input_contact_velocity

        # if self.contact_state and input_contact_state:
        #     print(self.contact_point, self.contact_position, input_contact_position)

        # Update contact state
        self.contact_state = input_contact_state
        self.contact_log.append(self.contact_state)
        self.contact_point_log.append(self.contact_point)
        self.contact_position_log.append(self.contact_position)


def init_contact(motion, frame, c_idx, foot_height=None):
    motion.contacts = {}
    # assert len(c_idx) == len(state)
    for i, ci in enumerate(c_idx):
        if foot_height is not None:
            contact_i = Contact(ik_foot_height=foot_height[i])
        else:
            contact_i = Contact()

        pose_f = motion.poses[frame]
        ci_f_position = pose_f.get_transform(ci, local=False)[:3, 3]
        ci_fprev_position = (
            motion.poses[frame - 1].get_transform(ci, local=False)[:3, 3]
            if frame != 0
            else ci_f_position
        )
        ci_f_velocity = (ci_f_position - ci_fprev_position) * 30
        contact_i.reset(ci_f_position, ci_f_velocity, False)
        motion.contacts[ci] = contact_i


def foot_cleanup_exact_point(
    pose,
    toe_contact_position,
    toe_chains,
    toe_i,
    ik_toe_vector,
    ik_foot_height,
    ik_max_length_buffer,
):
    assert toe_contact_position is not None
    assert ik_toe_vector is not None
    assert ik_foot_height is not None

    root_i, hip_i, knee_i, heel_i = toe_chains
    """ Perform simple two-joint IK to place heel """
    heel_global_p = pose.get_transform(heel_i, local=False)[:3, 3]
    toe_global_p = pose.get_transform(toe_i, local=False)[:3, 3]
    heel_target = toe_contact_position + heel_global_p - toe_global_p

    knee_global_R = pose.get_transform(knee_i, local=False)[:3, :3]
    knee_global_R_y = knee_global_R @ np.array([0.0, 1.0, 0.0])

    hip_local_rot, knee_local_rot = ik_two_bone_idx(
        pose,
        root_i,
        hip_i,
        knee_i,
        heel_i,
        heel_target,
        knee_global_R_y,
        ik_max_length_buffer,
    )
    pose.set_transform(hip_i, conversions.R2T(hip_local_rot), local=True)
    pose.set_transform(knee_i, conversions.R2T(knee_local_rot), local=True)

    """ Rotate heel so toe is facing toward contact point """
    heel_local_R = ik_look_at_idx(pose, knee_i, heel_i, toe_i, toe_contact_position)
    pose.set_transform(heel_i, conversions.R2T(heel_local_R), local=True)

    """ Rotate toe bone so that the end of the toe does not intersect with the ground """
    heel_global_Rot = Rotation.from_matrix(
        pose.get_transform(heel_i, local=False)[:3, :3]
    )

    toe_global_T = pose.get_transform(toe_i, local=False)
    toe_global_R = toe_global_T[:3, :3]
    toe_global_p = toe_global_T[:3, 3]
    toe_global_Rot = Rotation.from_matrix(toe_global_R)
    # toe_end_curr = toe_global_R @ (np.array([ik_toe_length, 0,0]) + toe_global_p)
    toe_end_curr = toe_global_R @ (ik_toe_vector + toe_global_p)
    toe_end_targ = toe_end_curr
    toe_end_targ[1] = max(toe_end_targ[1], ik_foot_height)

    toe_local_Rot = Rotation.from_matrix(pose.get_transform(toe_i, local=True)[:3, :3])
    toe_local_R = ik_look_at(
        toe_local_Rot,
        heel_global_Rot,
        toe_global_Rot,
        toe_global_p,
        toe_end_curr,
        toe_end_targ,
    )
    pose.set_transform(toe_i, conversions.R2T(toe_local_R), local=True)


def pose_foot_cleanup(
    pose, contacts, fc_info, toe_chains, dt=1 / 30.0, debug_frame=None
):
    for ci, (toe_i, contact) in enumerate(contacts.items()):
        cur_joint_position = pose.get_transform(toe_i, local=False)[:3, 3]
        cur_contact = fc_info[ci]
        contact.update(cur_joint_position, cur_contact, dt=dt, debug_frame=debug_frame)

        """ Ensure contact position never goes through floor """
        contact_position_clamped = contact.contact_position
        contact_position_clamped[1] = max(
            contact_position_clamped[1], contact.ik_foot_height
        )
        toe_vector = (
            pose.skel.get_joint(toe_i).child_joints[0].xform_from_parent_joint[:3, 3]
        )
        # foot_cleanup_exact_point(pose, contact_position_clamped, toe_chains[ci], toe_i, contact.ik_toe_length, contact.ik_foot_height, contact.ik_max_length_buffer)
        foot_cleanup_exact_point(
            pose,
            contact_position_clamped,
            toe_chains[ci],
            toe_i,
            toe_vector,
            contact.ik_foot_height,
            contact.ik_max_length_buffer,
        )


def motion_foot_cleanup(motion, ref_foot_contact):
    toe_chains = []
    for ci, (joint_i, contact) in enumerate(motion.contacts.items()):
        toe_chain_i = []
        joint = motion.skel.joints[joint_i]
        for _ in range(4):
            parent = joint.parent_joint
            parent_idx = motion.skel.get_index_joint(parent.name)
            toe_chain_i.append(parent_idx)
            joint = parent

        toe_chains.append(list(reversed(toe_chain_i)))

    for frame in range(motion.num_frames()):
        pose = motion.poses[frame]
        pose_foot_cleanup(pose, motion.contacts, ref_foot_contact[frame], toe_chains)


def poses_foot_cleanup(poses, contacts, ref_foot_contact):
    skel = poses[0].skel
    toe_chains = []
    for ci, (joint_i, contact) in enumerate(contacts.items()):
        toe_chain_i = []
        joint = skel.joints[joint_i]
        for _ in range(4):
            parent = joint.parent_joint
            parent_idx = skel.get_index_joint(parent.name)
            toe_chain_i.append(parent_idx)
            joint = parent

        toe_chains.append(list(reversed(toe_chain_i)))

    # print(toe_chains)
    for frame, pose in enumerate(poses):
        pose_foot_cleanup(pose, contacts, ref_foot_contact[frame], toe_chains)


# ====================== offline cleanup  ======================
def glue_foot(
    motion, toe_chains, toe_i, mid, start, end, ik_foot_height=1, ik_toe_length=15
):
    bvh_scale = 100
    ik_max_length_buffer = 0.015 * bvh_scale
    # ik_foot_height = 0.01 * bvh_scale
    # ik_toe_length = 0.15 * bvh_scale

    contact_position_clamped = motion.poses[mid].get_transform(toe_i, local=False)[
        :3, 3
    ]
    contact_position_clamped[1] = max(contact_position_clamped[1], ik_foot_height)

    for fi in range(start, end):
        foot_cleanup_exact_point(
            motion.poses[fi],
            contact_position_clamped,
            toe_chains,
            toe_i,
            ik_toe_length,
            ik_foot_height,
            ik_max_length_buffer,
        )


import copy


def foot_smooth_before(
    motion, toe_chains, toe_i, start, mid, d, ik_foot_height=1, ik_toe_length=15
):
    bvh_scale = 100
    ik_max_length_buffer = 0.015 * bvh_scale
    # ik_foot_height = 0.01 * bvh_scale
    # ik_toe_length = 0.15 * bvh_scale
    contact_position_clamped = motion.poses[mid].get_transform(toe_i, local=False)[
        :3, 3
    ]
    contact_position_clamped[1] = max(contact_position_clamped[1], ik_foot_height)

    for i in range(1, d):
        p = motion.poses[start - d + i]
        glue_p = copy.deepcopy(p)
        foot_cleanup_exact_point(
            glue_p,
            contact_position_clamped,
            toe_chains,
            toe_i,
            ik_toe_length,
            ik_foot_height,
            ik_max_length_buffer,
        )

        t = 0.5 + 0.5 * np.cos(np.pi * i / d)
        blend_p = motion_ops.blend(glue_p, p, t)
        motion.poses[start - d + i] = blend_p


def foot_smooth_after(
    motion, toe_chains, toe_i, end, mid, d, ik_foot_height=1, ik_toe_length=15
):
    bvh_scale = 100
    ik_max_length_buffer = 0.015 * bvh_scale
    # ik_foot_height = 0.01 * bvh_scale
    # ik_toe_length = 0.15 * bvh_scale
    contact_position_clamped = motion.poses[mid].get_transform(toe_i, local=False)[
        :3, 3
    ]
    contact_position_clamped[1] = max(contact_position_clamped[1], ik_foot_height)

    for i in range(1, d):
        p = motion.poses[end + i - 1]
        glue_p = copy.deepcopy(p)
        foot_cleanup_exact_point(
            glue_p,
            contact_position_clamped,
            toe_chains,
            toe_i,
            ik_toe_length,
            ik_foot_height,
            ik_max_length_buffer,
        )

        t = 0.5 - 0.5 * np.cos(np.pi * i / d)
        blend_p = motion_ops.blend(glue_p, p, t)
        motion.poses[end + i - 1] = blend_p


def motion_post_foot_cleanup(motion, contacts, toe_i):
    sDuration, eDuration = 5, 5
    margin = 0

    F = contacts.shape[0]
    frameStart = 0
    mid = 0

    joint = motion.skel.joints[toe_i]
    toe_chains = []
    for _ in range(4):
        parent = joint.parent_joint
        parent_idx = motion.skel.get_index_joint(parent.name)
        toe_chains.append(parent_idx)
        joint = parent

    toe_chains = list(reversed(toe_chains))

    for fi in range(1, F + 1):
        conBefore = contacts[fi - 1]
        conCurrent = (fi < F) and contacts[fi]

        if conBefore and (not conCurrent):
            frameEnd = fi - 1
            mid = (frameStart + frameEnd) // 2
            start = min(mid, frameStart + margin)
            end = max(mid, frameEnd + margin)

            print("foot glue: ", start, mid, end)
            glue_foot(motion, toe_chains, toe_i, mid, start, end)
            foot_smooth_before(
                motion, toe_chains, toe_i, start, mid, min(start, sDuration)
            )
            foot_smooth_after(
                motion, toe_chains, toe_i, end, mid, min(F - 1 - end, eDuration)
            )

        elif (not conBefore) and conCurrent:
            frameStart = fi
    # in-place
