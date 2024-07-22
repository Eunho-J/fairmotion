# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import os
from fairmotion.core import motion as motion_classes
from fairmotion.core.velocity import MotionWithVelocity
from fairmotion.utils import constants, utils
from fairmotion.ops import conversions
from fairmotion.utils.contact_utils import get_foot_indices, get_foot_contact_ratio


def load(
    file,
    motion=None,
    scale=1.0,
    load_skel=True,
    load_motion=True,
    load_velocity=False,
    v_up_skel=np.array([0.0, 1.0, 0.0]),
    v_face_skel=np.array([0.0, 0.0, 1.0]),
    v_up_env=np.array([0.0, 1.0, 0.0]),
    ignore_root_skel=True, # SM)
    ee_as_joint=True, # SM)
):
    if not motion:
        motion = motion_classes.Motion()
    words = None
    with open(file, "rb") as f:
        words = [word.decode() for line in f for word in line.split()]
        f.close()
    assert words is not None and len(words) > 0
    motion.name = os.path.normpath(file)
    cnt = 0
    total_depth = 0
    joint_stack = [None, None]
    joint_list = []
    parent_joint_list = []

    if load_skel:
        assert motion.skel is None
        motion.skel = motion_classes.Skeleton(
            v_up=v_up_skel, v_face=v_face_skel, v_up_env=v_up_env,
        )

    if load_skel:
        end_extra_link = False
        while cnt < len(words):
            # joint_prev = joint_stack[-2]
            joint_cur = joint_stack[-1]
            word = words[cnt].lower()
            if word == "root" or word == "joint":
                parent_joint_list.append(joint_cur)
                name = words[cnt + 1]
                joint = motion_classes.Joint(name=name)
                joint_stack.append(joint)
                joint_list.append(joint)
                cnt += 2
            elif word == "offset":
                x, y, z = (
                    float(words[cnt + 1]),
                    float(words[cnt + 2]),
                    float(words[cnt + 3]),
                )
                T1 = conversions.p2T(scale * np.array([x, y, z]))
                joint_cur.xform_from_parent_joint = T1

                # SM)
                if len(joint_list) == 1 and ignore_root_skel:
                    x, y, z= (0.0, 0.0, 0.0)
                T1 = conversions.p2T(scale * np.array([x, y, z]))
                joint_cur.xform_from_parent_joint = T1

                if end_extra_link and (not ee_as_joint): 
                    parent_extra.add_extra_links(parent_extra.name+"_End", T1)
                    end_extra_link = False


                cnt += 4
            elif word == "channels":
                ndofs = int(words[cnt + 1])
                if ndofs == 6:
                    joint_cur.info["type"] = "free"
                elif ndofs == 3:
                    joint_cur.info["type"] = "ball"
                elif ndofs == 1:
                    joint_cur.info["type"] = "revolute"
                else:
                    raise Exception("Undefined")
                joint_cur.info["dof"] = ndofs
                joint_cur.info["bvh_channels"] = []
                for i in range(ndofs):
                    joint_cur.info["bvh_channels"].append(
                        words[cnt + 2 + i].lower()
                    )
                cnt += ndofs + 2
            elif word == "end":
                # SM)
                if ee_as_joint:
                    parent_joint_list.append(joint_cur)
                    joint = motion_classes.Joint(name=joint_cur.name+'_End')
                    joint.info["dof"] = 0
                    joint.info["bvh_channels"] = []
                    joint_stack.append(joint)
                    joint_list.append(joint)

                else:
                    parent_extra = joint_cur 
                    joint_dummy = motion_classes.Joint(name=joint_cur.name+"_End")
                    end_extra_link = True
                    joint_stack.append(joint_dummy)
                # joint_dummy = motion_classes.Joint(name="END")
                # joint_stack.append(joint_dummy)
                cnt += 2
            elif word == "{":
                total_depth += 1
                cnt += 1
            elif word == "}":
                joint_stack.pop()
                total_depth -= 1
                cnt += 1
                if total_depth == 0:
                    for i in range(len(joint_list)):
                        motion.skel.add_joint(
                            joint_list[i], parent_joint_list[i],
                        )
                    break
            elif word == "hierarchy":
                cnt += 1
            else:
                raise Exception(f"Unknown Token {word} at token {cnt}")

    if load_motion:
        assert motion.skel is not None
        assert np.allclose(motion.skel.v_up, v_up_skel)
        assert np.allclose(motion.skel.v_face, v_face_skel)
        assert np.allclose(motion.skel.v_up_env, v_up_env)
        while cnt < len(words):
            word = words[cnt].lower()
            if word == "motion":
                num_frames = int(words[cnt + 2])
                dt = float(words[cnt + 5])
                motion.set_fps(round(1 / dt))
                cnt += 6
                t = 0.0
                range_num_dofs = range(motion.skel.num_dofs)
                positions = np.zeros(
                    (num_frames, motion.skel.num_joints(), 3, 3)
                )
                rotations = np.zeros((num_frames, motion.skel.num_joints(), 3))
                T = np.zeros((num_frames, motion.skel.num_joints(), 4, 4))
                T[...] = constants.eye_T()
                position_channels = {
                    "xposition": 0,
                    "yposition": 1,
                    "zposition": 2,
                }
                rotation_channels = {
                    "xrotation": 0,
                    "yrotation": 1,
                    "zrotation": 2,
                }
                for frame_idx in range(num_frames):
                    # if frame_idx == 1:
                    #     break
                    raw_values = [
                        float(words[cnt + j]) for j in range_num_dofs
                    ]
                    cnt += motion.skel.num_dofs
                    cnt_channel = 0
                    for joint_idx, joint in enumerate(motion.skel.joints):
                        for channel in joint.info["bvh_channels"]:
                            value = raw_values[cnt_channel]
                            if channel in position_channels:
                                value = scale * value
                                positions[frame_idx][joint_idx][
                                    position_channels[channel]
                                ][position_channels[channel]] = value
                            elif channel in rotation_channels:
                                value = conversions.deg2rad(value)
                                rotations[frame_idx][joint_idx][
                                    rotation_channels[channel]
                                ] = value
                            else:
                                raise Exception("Unknown Channel")
                            cnt_channel += 1

                for joint_idx, joint in enumerate(motion.skel.joints):
                    for channel in joint.info["bvh_channels"]:
                        if channel in position_channels:
                            T[:, joint_idx] = T[:, joint_idx] @ conversions.p2T(
                                positions[
                                    :,
                                    joint_idx,
                                    position_channels[channel],
                                    :,
                                ]
                            )
                        elif channel == "xrotation":
                            T[:, joint_idx] = T[:, joint_idx] @ conversions.R2T(
                                conversions.Ax2R(
                                    rotations[
                                        :,
                                        joint_idx,
                                        rotation_channels[channel],
                                    ]
                                )
                            )
                        elif channel == "yrotation":
                            T[:, joint_idx] = T[:, joint_idx] @ conversions.R2T(
                                conversions.Ay2R(
                                    rotations[
                                        :,
                                        joint_idx,
                                        rotation_channels[channel],
                                    ]
                                )
                            )
                        elif channel == "zrotation":
                            T[:, joint_idx] = T[:, joint_idx] @ conversions.R2T(
                                conversions.Az2R(
                                    rotations[
                                        :,
                                        joint_idx,
                                        rotation_channels[channel],
                                    ]
                                )
                            )

                for i in range(num_frames):
                    motion.add_one_frame(list(T[i]))
                    t += dt
            else:
                cnt += 1
        if load_velocity:
            motion = MotionWithVelocity.from_motion(motion)
        assert motion.num_frames() > 0

    ### ad-hoc remove prefix from 'prefix:joint_name' (e.g. mixamorig:Spine)
    for ji, joint in enumerate(motion.skel.joints): 
        if ':' in joint.name:
            motion.skel.change_joint_name(ji, joint.name.split(':')[-1])

    return motion


def _write_hierarchy(motion, file, joint, scale=1.0, rot_order="XYZ", tab=""):
    def rot_order_to_str(order):
        if order == "xyz" or order == "XYZ":
            return "Xrotation Yrotation Zrotation"
        elif order == "zyx" or order == "ZYX":
            return "Zrotation Yrotation Xrotation"
        elif order == "zxy" or order == "ZXY": # SM)
            return "Zrotation Xrotation Yrotation"
        else:
            raise NotImplementedError

    joint_order = [joint.name]
    is_root_joint = joint.parent_joint is None
    if is_root_joint:
        file.write(tab + "ROOT %s\n" % joint.name)
    else:
        file.write(tab + "JOINT %s\n" % joint.name)
    file.write(tab + "{\n")
    R, p = conversions.T2Rp(joint.xform_from_parent_joint)
    p *= scale
    file.write(tab + "\tOFFSET %f %f %f\n" % (p[0], p[1], p[2]))
    if is_root_joint:
        file.write(
            tab
            + "\tCHANNELS 6 Xposition Yposition Zposition %s\n"
            % rot_order_to_str(rot_order)
        )
    else:
        file.write(tab + "\tCHANNELS 3 %s\n" % rot_order_to_str(rot_order))
    for child_joint in joint.child_joints:
        child_joint_order = _write_hierarchy(
            motion, file, child_joint, scale, rot_order, tab + "\t"
        )
        joint_order.extend(child_joint_order)
    if len(joint.child_joints) == 0:
        file.write(tab + "\tEnd Site\n")
        file.write(tab + "\t{\n")
        file.write(tab + "\t\tOFFSET %f %f %f\n" % (0.0, 0.0, 0.0))
        file.write(tab + "\t}\n")
    file.write(tab + "}\n")
    return joint_order


def save(motion, filename, scale=1.0, rot_order="XYZ", verbose=False, ee_as_joint=True, root_motion_local=False):
    if verbose:
        print(" >  >  Save BVH file: %s" % filename)
    with open(filename, "w") as f:
        """ Write hierarchy """
        if verbose:
            print(" >  >  >  >  Write BVH hierarchy")
        f.write("HIERARCHY\n")
        if ee_as_joint: # SM)
            joint_order = _write_hierarchy_ee_as_joint(
                motion, f, motion.skel.root_joint, scale, rot_order
            )
        else:
            joint_order = _write_hierarchy(
                motion, f, motion.skel.root_joint, scale, rot_order
            )
        """ Write data """
        if verbose:
            print(" >  >  >  >  Write BVH data")
        t_start = 0
        dt = 1.0 / motion.fps
        num_frames = motion.num_frames()
        f.write("MOTION\n")
        f.write("Frames: %d\n" % num_frames)
        f.write("Frame Time: %f\n" % dt)
        t = t_start

        for i in range(num_frames):
            if verbose and i % motion.fps == 0:
                print(
                    "\r >  >  >  >  %d/%d processed (%d FPS)"
                    % (i + 1, num_frames, motion.fps),
                    end=" ",
                )
            pose = motion.get_pose_by_frame(i)

            for joint_name in joint_order:
                joint = motion.skel.get_joint(joint_name)
                if joint == motion.skel.root_joint:
                    R, p = conversions.T2Rp(pose.get_transform(joint, local=root_motion_local))
                    p *= scale
                    R1, R2, R3 = conversions.R2E(R, order=rot_order, degrees=True)
                    f.write(
                        "%f %f %f %f %f %f " % (p[0], p[1], p[2], R1, R2, R3)
                    )
                elif joint.info["dof"] == 0:
                    continue
                else:
                    R, p = conversions.T2Rp(pose.get_transform(joint, local=True))
                    p *= scale
                    R1, R2, R3 = conversions.R2E(R, order=rot_order, degrees=True)
                    f.write("%f %f %f " % (R1, R2, R3))
            f.write("\n")
            t += dt
            if verbose and i == num_frames - 1:
                print(
                    "\r >  >  >  >  %d/%d processed (%d FPS)"
                    % (i + 1, num_frames, motion.fps)
                )
        f.close()


def load_parallel(files, cpus=20, **kwargs):
    return utils.run_parallel(load, files, num_cpus=cpus, **kwargs)


# SM)
def _write_hierarchy_ee_as_joint(motion, file, joint, scale=1.0, rot_order="XYZ", tab=""):
    def rot_order_to_str(order):
        if order == "xyz" or order == "XYZ":
            return "Xrotation Yrotation Zrotation"
        elif order == "zyx" or order == "ZYX":
            return "Zrotation Yrotation Xrotation"
        elif order == "zxy" or order == "ZXY": #SM
            return "Zrotation Xrotation Yrotation"
        else:
            raise NotImplementedError
    
    # print(joint.name, joint.info['dof'])
    if joint.info["dof"] == 0:
        file.write(tab + "End Site\n")
        file.write(tab + "{\n")
        R, p = conversions.T2Rp(joint.xform_from_parent_joint)
        p *= scale
        file.write(tab + "\tOFFSET %f %f %f\n" % (p[0], p[1], p[2]))
        file.write(tab + "}\n")
        return []

    joint_order = [joint.name]
    is_root_joint = joint.parent_joint is None
    if is_root_joint:
        file.write(tab + "ROOT %s\n" % joint.name)
    else:
        file.write(tab + "JOINT %s\n" % joint.name)

    file.write(tab + "{\n")
    R, p = conversions.T2Rp(joint.xform_from_parent_joint)
    p *= scale
    file.write(tab + "\tOFFSET %f %f %f\n" % (p[0], p[1], p[2]))
    if is_root_joint:
        file.write(
            tab
            + "\tCHANNELS 6 Xposition Yposition Zposition %s\n"
            % rot_order_to_str(rot_order)
        )
    else:
        file.write(tab + "\tCHANNELS 3 %s\n" % rot_order_to_str(rot_order))
    for child_joint in joint.child_joints:
        child_joint_order = _write_hierarchy_ee_as_joint(
            motion, file, child_joint, scale, rot_order, tab + "\t"
        )
        joint_order.extend(child_joint_order)
    file.write(tab + "}\n")
    return joint_order

