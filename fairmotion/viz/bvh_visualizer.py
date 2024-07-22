# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import numpy as np
import os, copy
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PIL import Image
from abc import ABC, abstractclassmethod

from fairmotion.viz import camera, gl_render, glut_viewer
from fairmotion.data import bvh, asfamc
from fairmotion.ops import conversions, math, motion as motion_ops
from fairmotion.viz.utils import TimeChecker
from fairmotion.utils import utils
from IPython import embed
import subprocess


class WorldStepper(ABC):
    def __init__(self, fps=30) -> None:
        self.time_checker = TimeChecker()
        self.fps = fps
        self.isPlaying = False
        self.reset()

    def togglePlaying(self):
        self.isPlaying = not self.isPlaying
        if self.isPlaying:
            self.time_checker.begin()

    @abstractclassmethod
    def reset(self, is_playing=False):
        pass

    @abstractclassmethod
    def idle(self):
        pass

    @abstractclassmethod
    def incFrame(self):
        pass

    @abstractclassmethod
    def decFrame(self):
        pass

    @abstractclassmethod
    def get_update_interval(self):
        pass


# Inherited
class MocapWorldStepper(WorldStepper):
    def __init__(self, viewer, fps=30, play_speed=1, max_frame_num=1) -> None:
        super(MocapWorldStepper, self).__init__(fps)
        self.play_speed = play_speed
        self.max_frame_num = max_frame_num
        self.cur_time = 0.0
        self.cur_frame = 0
        self.last_update_time = 0
        self.update_interval = 0
        self.viewer = viewer

    # @ abstractclassmethod
    def idle(self):
        if self.isPlaying:
            time_elapsed = self.time_checker.get_time(restart=False)
            self.set_time(self.cur_time + time_elapsed * self.play_speed)
            self.time_checker.begin()

    def set_time(self, new_time):
        # print(f'{new_time:4f} {new_time- self.cur_time:4f} {self.last_update_time:2f} {self.update_interval:2f}')
        self.cur_time = new_time
        frame = int(new_time * self.fps + 1e-05) % self.max_frame_num

        if frame != self.cur_frame:
            self.update_interval = new_time - self.last_update_time
            self.last_update_time = new_time

            self.viewer.trackCameraUpdate(frame)
            self.cur_frame = frame

            # print(f'{frame}, {self.update_interval:4f}')

    def get_update_interval(self):
        return self.update_interval

    def incFrame(self):
        self.cur_frame = (self.cur_frame + 1) % self.max_frame_num
        self.cur_time = self.cur_frame / self.fps

    def decFrame(self):
        self.cur_frame = (self.cur_frame - 1) % self.max_frame_num
        self.cur_time = self.cur_frame / self.fps

    def reset(self, is_playing=False):
        self.isPlaying = is_playing
        self.cur_frame = 0
        self.cur_time = 0
        self.time_checker.begin()


class MocapViewer(glut_viewer.Viewer):
    """
    MocapViewer is an extension of the glut_viewer.Viewer class that implements
    requisite callback functions -- render_callback, keyboard_callback,
    idle_callback and overlay_callback.

    ```
    python fairmotion/viz/bvh_visualizer.py \
        --bvh-files $BVH_FILE1
    ```

    To visualize more than 1 motion sequence side by side, append more files 
    to the `--bvh-files` argument. Set `--x-offset` to an appropriate float 
    value to add space separation between characters in the row.

    ```
    python fairmotion/viz/bvh_visualizer.py \
        --bvh-files $BVH_FILE1 $BVH_FILE2 $BVH_FILE3 \
        --x-offset 2
    ```

    To visualize asfamc motion sequence:

    ```
    python fairmotion/viz/bvh_visualizer.py \
        --asf-files tests/data/11.asf \
        --amc-files tests/data/11_01.amc
    ```

    """

    def __init__(
        self,
        motions,
        play_speed=1.0,
        joint_scale=1.0,
        link_scale=1.0,
        render_overlay=False,
        hide_origin=False,
        bvh_scale=1,
        trackCamera=False,  # SM)
        **kwargs,
    ):
        # self.motions = motions
        self.play_speed = play_speed
        self.render_overlay = render_overlay
        self.hide_origin = hide_origin
        self.file_idx = 0
        self.joint_scale = joint_scale  # SM) changed name from self.scale
        self.link_scale = link_scale  # SM) changed name from self.thickness

        # SM)
        self.bvh_scale = bvh_scale
        self.worldStepper = MocapWorldStepper(self)
        self.update_motions(motions)

        self.prior_key_callback = lambda x, y: False
        self.extra_key_callback = lambda x: False
        self.extra_render_callback = None
        self.extra_overlay_callback = None

        super().__init__(**kwargs)

        self.set_trackCamera(trackCamera)

    def set_trackCamera(self, trackCamera):
        self.trackCamera = trackCamera
        if self.trackCamera:
            self.track_pos = self.cam_cur.origin

    def trackCameraUpdate(self, frame):
        if (
            self.trackCamera
            and len(self.motions) > 0
            and (self.motions[0].num_frames() > frame)
        ):
            new_track_pos = np.zeros(3)
            for motion in self.motions:
                new_track_pos += (
                    self.bvh_scale * motion.poses[frame].get_root_transform()[:3, 3]
                )
            new_track_pos /= len(self.motions)
            self.track_pos = new_track_pos

            # self.track_pos = self.bvh_scale* self.motions[0].poses[frame].get_root_transform()[:3,3]
            if not np.array_equal(self.track_pos, self.cam_cur.origin):
                track_diff = self.track_pos - self.cam_cur.origin
                track_diff[1] = 0
                self.cam_cur.translate(track_diff)

    def update_motions(
        self, new_motions, grid_size=0, update_vis_num=True, linear=False, reset=True
    ):
        self.motions = new_motions
        if grid_size > 0:
            n_motions = len(self.motions)
            n_grid = int(np.sqrt(n_motions))
            for mi, motion in enumerate(self.motions):
                m_cp = copy.deepcopy(motion)
                if linear:
                    row_i, col_i = mi, 0
                else:
                    row_i, col_i = mi % n_grid, mi // n_grid
                motion_ops.transform(
                    m_cp, conversions.p2T([row_i * grid_size, 0, col_i * grid_size])
                )
                self.motions[mi] = m_cp

        self.worldStepper.max_frame_num = (
            max([m.num_frames() for m in self.motions]) if len(self.motions) > 0 else 1
        )
        self.worldStepper.fps = self.motions[0].fps if len(self.motions) > 0 else 30
        if update_vis_num:
            self.vis_motion_num = len(self.motions)
        if reset:
            self.worldStepper.reset(True)

    def keyboard_callback(self, key, mode=None):
        if self.prior_key_callback(key, mode):
            return True

        if len(self.motions) > 0:
            motion = self.motions[self.file_idx]
        if key == b"s":
            self.worldStepper.reset()
            return True
        elif key == b"e":
            embed()
            return True

        elif key == b"]":
            if len(self.motions) <= 0:
                return False
            next_frame = min(motion.num_frames() - 1, self.worldStepper.cur_frame + 1)
            self.worldStepper.set_time(motion.frame_to_time(next_frame))
            return True

        elif key == b"[":
            if len(self.motions) <= 0:
                return False
            prev_frame = max(0, self.worldStepper.cur_frame - 1)
            self.worldStepper.set_time(motion.frame_to_time(prev_frame))
            return True

        elif key == b"+":
            self.play_speed = min(self.play_speed + 0.2, 5.0)
            self.worldStepper.play_speed = self.play_speed
            return True

        elif key == b"-":
            self.play_speed = max(self.play_speed - 0.2, 0.2)
            self.worldStepper.play_speed = self.play_speed
            return True
        elif key == b"r" or key == b"v":
            if len(self.motions) <= 0:
                return False
            self.worldStepper.reset()

            fps = motion.fps
            save_dir = input("Enter directory/file to store screenshots/video: ")
            save_path = os.path.join(save_dir, "img")

            start_frame = input("Enter start_frame:")
            if start_frame == "":
                start_frame = 0
            else:
                start_frame = int(start_frame)

            end_frame = input("Enter end_frame:")
            if end_frame == "":
                end_frame = motion.num_frames()
            else:
                end_frame = int(end_frame)

            assert start_frame < end_frame

            cnt_screenshot = 0
            dt = 1 / fps
            gif_images = []
            while cnt_screenshot < end_frame - start_frame:
                print(
                    f"Recording progress: {self.worldStepper.cur_frame}/{end_frame} ({int(100*self.worldStepper.cur_frame/end_frame)}%) \r",
                    end="",
                )
                if key == b"r":
                    utils.create_dir_if_absent(save_path)
                    name = "screenshot_%04d" % (cnt_screenshot)
                    self.save_screen(dir=save_path, name=name, render=True)
                else:
                    image = self.get_screen(render=True)
                    gif_images.append(image.convert("P", palette=Image.ADAPTIVE))
                self.worldStepper.set_time(self.worldStepper.cur_time + dt)
                cnt_screenshot += 1

            command_args = [
                "ffmpeg",
                "-framerate",
                str(fps),
                "-f",
                "image2",
                "-i",
                os.path.join(save_path, "screenshot_%04d.png"),
                "-vcodec",
                "libx264",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-crf",
                "25",
                "-pix_fmt",
                "yuv420p",
                os.path.join(save_dir, f"{save_dir}.mp4"),
            ]
            subprocess.run(command_args)
            print(">>> save done")
            self.worldStepper.reset()

            if key == b"v":
                utils.create_dir_if_absent(os.path.dirname(save_path))
                gif_images[0].save(
                    save_path,
                    save_all=True,
                    optimize=False,
                    append_images=gif_images[1:],
                    loop=0,
                )
            return True

        elif key == b" ":
            self.worldStepper.togglePlaying()
            return True
        # mac setting
        elif key == b"o":
            self.cam_cur.zoom(0.95)
            return True
        elif key == b"p":
            self.cam_cur.zoom(1.05)
            return True

        elif self.extra_key_callback(key):
            return True
        else:
            return False

    # SM)
    def _render_pose(self, pose, body_model, color):
        skel = pose.skel

        for j in skel.joints:
            T = pose.get_transform(j, local=False)
            pos = conversions.T2p(T)
            gl_render.render_point(pos, radius=self.joint_scale, color=color)
            if j.parent_joint is not None:
                gl_render.render_link_cylinder(pose, j, self.link_scale, color)
                # gl_render.render_link_cube(pose, j, self.link_scale, color) # buggy # TODO

            # elif self.draw_root_coord:
            #     gl_render.render_transform(T, scale = 0.1/self.bvh_scale, line_width=3, use_arrow=True) # draw local cordinate of joints

    def _get_char_colors(self, motion_i, motion_num, colors):
        # color = colors[i % len(colors)]
        if motion_num < len(colors):
            return colors[motion_i]
        else:
            # n_width = math.floor(len(self.motions) / len(colors))
            # color =
            return colors[motion_i % len(colors)]

    def _render_characters(self, colors):
        for i, motion in enumerate(self.motions):
            motion_i_frame = min(motion.num_frames() - 1, self.worldStepper.cur_frame)
            pose = motion.get_pose_by_frame(motion_i_frame)
            color = self._get_char_colors(i, len(self.motions), colors)

            glEnable(GL_LIGHTING)
            glEnable(GL_DEPTH_TEST)

            glEnable(GL_LIGHTING)
            self._render_pose(pose, "stick_figure2", color)
            if i >= self.vis_motion_num - 1:
                break

    def render_callback(self):
        gl_render.render_ground(
            size=[20, 20],
            color=[0.8, 0.8, 0.8],
            axis=utils.axis_to_str(self.cam_cur.vup),
            origin=not self.hide_origin,
            use_arrow=True,
            fillIn=True,  # SM)
        )
        colors = [
            np.array([148, 50, 211, 255]) / 255.0,  # purple
            np.array([75, 30, 130, 255]) / 255.0,  # deep purple
            np.array([50, 50, 255, 255]) / 255.0,  # blue
            np.array([85, 160, 173, 255]) / 255.0,  # light blue
            np.array([50, 255, 50, 255]) / 255.0,  # green
            np.array([123, 174, 85, 255]) / 255.0,  # light green
            np.array([255, 255, 0, 255]) / 255.0,  # yellow
            np.array([255, 127, 0, 255]) / 255.0,  # orange,
            np.array([255, 0, 0, 255]) / 255.0,  # red,
        ]

        if len(self.motions) > 0:
            glPushMatrix()
            glScaled(self.bvh_scale, self.bvh_scale, self.bvh_scale)
            self._render_characters(colors)
            if self.extra_render_callback:
                self.extra_render_callback()
            glPopMatrix()

        # SM) Draw Fog
        fogColor = [1.0, 1.0, 1.0, 1.0]
        glFogi(GL_FOG_MODE, GL_LINEAR)
        glFogfv(GL_FOG_COLOR, fogColor)
        glFogf(GL_FOG_DENSITY, 0.05)
        glHint(GL_FOG_HINT, GL_DONT_CARE)
        glFogf(GL_FOG_START, 13)
        glFogf(GL_FOG_END, 40.0)
        glEnable(GL_FOG)

    def idle_callback(self):
        self.worldStepper.idle()

    def overlay_callback(self):
        if self.render_overlay:
            w, h = self.window_size
            # SM) log update_time
            update_interval = self.worldStepper.get_update_interval()
            gl_render.render_text(
                f"fps_inv: {update_interval:.4f}s",
                pos=[0.8 * w, 0.1 * h],
                font=GLUT_BITMAP_TIMES_ROMAN_24,
            )

            frame = self.worldStepper.cur_frame
            gl_render.render_text(
                f"Frame: {frame:d}",
                pos=[0.9 * w, 0.1 * h],
                font=GLUT_BITMAP_TIMES_ROMAN_24,
            )

        if self.extra_overlay_callback:
            self.extra_overlay_callback()


def main(args):
    v_up_env = utils.str_to_axis(args.axis_up)
    if args.bvh_files:
        motions = [
            bvh.load(
                file=filename,
                v_up_skel=v_up_env,
                v_face_skel=utils.str_to_axis(args.axis_face),
                v_up_env=v_up_env,
                joint_scale=args.scale,
            )
            for filename in args.bvh_files
        ]
    else:
        motions = [
            asfamc.load(file=f, motion=m)
            for f, m in zip(args.asf_files, args.amc_files)
        ]

    for i in range(len(motions)):
        motion_ops.translate(motions[i], [args.x_offset * i, 0, 0])
    cam = camera.Camera(
        pos=np.array(args.camera_position),
        origin=np.array(args.camera_origin),
        vup=v_up_env,
        fov=45.0,
    )
    viewer = MocapViewer(
        motions=motions,
        play_speed=args.speed,
        joint_scale=args.joint_scale,
        link_scale=args.link_scale,
        render_overlay=args.render_overlay,
        hide_origin=args.hide_origin,
        title="Motion Graph Viewer",
        cam=cam,
        size=(1280, 720),
    )
    viewer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize BVH file with block body")
    parser.add_argument("--bvh-files", type=str, nargs="+", required=False)
    parser.add_argument("--asf-files", type=str, nargs="+", required=False)
    parser.add_argument("--amc-files", type=str, nargs="+", required=False)
    parser.add_argument("--joint_scale", type=float, default=1.0)
    parser.add_argument(
        "--link_scale",
        type=float,
        default=1.0,
        help="Thickness (radius) of character body",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--axis-up", type=str, choices=["x", "y", "z"], default="z")
    parser.add_argument("--axis-face", type=str, choices=["x", "y", "z"], default="y")
    parser.add_argument(
        "--camera-position",
        nargs="+",
        type=float,
        required=False,
        default=(2.0, 2.0, 2.0),
    )
    parser.add_argument(
        "--camera-origin",
        nargs="+",
        type=float,
        required=False,
        default=(0.0, 0.0, 0.0),
    )
    parser.add_argument("--hide-origin", action="store_true")
    parser.add_argument("--render-overlay", action="store_true")
    parser.add_argument(
        "--x-offset",
        type=int,
        default=2,
        help="Translates each character by x-offset*idx to display them "
        "simultaneously side-by-side",
    )
    args = parser.parse_args()
    assert len(args.camera_position) == 3 and len(args.camera_origin) == 3, (
        "Provide x, y and z coordinates for camera position/origin like "
        "--camera-position x y z"
    )
    main(args)
