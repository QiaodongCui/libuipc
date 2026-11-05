"""Blender: scenes, stills and videos of fem_damping_demos.py's results (the runs side by side, shape keys per frame).

  blender -b --factory-startup -noaudio --python fem_damping_demos_render.py -- OUT_DIR

Reads OUT_DIR/show.npz (three plucked arms) and spin.npz (spinning cubes); writes show.png, show.mp4, show.blend,
spin.png, spin.mp4 and spin.blend. Frame 1 is the start; 60 frames a second.
"""
import math
import os
import shutil
import sys

import bpy
import numpy as np
from mathutils import Vector

out_dir = os.path.abspath(sys.argv[sys.argv.index("--") + 1])  # Blender resolves relative render paths elsewhere
COLOR = {0.0: "#2a78d6", 0.002: "#eb6834", 0.005: "#1baf7a", 0.02: "#eda100", 0.05: "#e87ba4"}  # slots 1-5 by beta
INK = "#52514e"


def rgba(hexstr):
    h = hexstr.lstrip("#")
    srgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb] + [1.0]


def material(name, hexstr):
    m = bpy.data.materials.new(name)
    m.diffuse_color = rgba(hexstr)
    return m


def animated_mesh(name, F, frames, location, mat):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in frames[0]], [], [tuple(int(i) for i in f) for f in F])
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = location
    ob.data.materials.append(mat)
    ob.shape_key_add(name="basis")
    for f, X in enumerate(frames):
        kb = ob.shape_key_add(name=f"f{f:04d}")
        kb.data.foreach_set("co", np.asarray(X, np.float32).ravel())
        for fr, val in ((f, 0.0), (f + 1, 1.0), (f + 2, 0.0)):
            kb.value = val
            kb.keyframe_insert("value", frame=fr)
    return ob


def label(text, location, size, rotation=(math.radians(90), 0, 0), align="CENTER"):
    cu = bpy.data.curves.new("label", "FONT")
    cu.body, cu.size, cu.align_x = text, size, align
    ob = bpy.data.objects.new("label " + text, cu)
    ob.location, ob.rotation_euler = location, rotation
    ob.visible_shadow = False
    ob.data.materials.append(material("ink", INK))
    bpy.context.scene.collection.objects.link(ob)
    return ob


def box(name, center, size, mat):
    bpy.ops.mesh.primitive_cube_add(location=center)
    ob = bpy.context.object
    ob.name, ob.scale = name, (size[0] / 2, size[1] / 2, size[2] / 2)
    ob.data.materials.append(mat)
    return ob


def camera(location, target, lens=35.0, ortho=None):
    cd = bpy.data.cameras.new("camera")
    if ortho:
        cd.type, cd.ortho_scale = "ORTHO", ortho
    else:
        cd.lens = lens
    cam = bpy.data.objects.new("camera", cd)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = location
    cam.rotation_euler = (Vector(target) - Vector(location)).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    return cam


def setup(n_frames, res, fps):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "STUDIO"
    sc.display.shading.color_type = "MATERIAL"
    sc.display.shading.show_shadows = False
    sc.display.shading.show_cavity = True
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.world = bpy.data.worlds.new("background")
    sc.world.color = rgba("#fcfcfb")[:3]
    sc.view_settings.view_transform = "Standard"  # the palette's colours as given (AgX greys them)
    sc.frame_start, sc.frame_end = 1, n_frames
    sc.render.fps = fps
    return sc


def clock(sc, location, size, rotation):
    """A 't = 0.00 s' label that follows the frame."""
    ob = label("t = 0.00 s", location, size, rotation)
    fps = sc.render.fps

    def update(scene, *_):
        ob.data.body = f"t = {(scene.frame_current - 1) / fps:.2f} s"

    bpy.app.handlers.frame_change_pre.append(update)
    return ob


def still(sc, frame, path):
    sc.frame_set(frame)
    sc.render.image_settings.media_type = "IMAGE"
    sc.render.image_settings.file_format = "PNG"
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


def video(sc, path):
    """H.264 of the whole frame range, at exactly `path` (Blender appends the frame range to its own name)."""
    tmp = os.path.join(out_dir, "_video")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    s = sc.render.image_settings
    s.media_type = "VIDEO"
    s.file_format = "FFMPEG"
    sc.render.ffmpeg.format, sc.render.ffmpeg.codec = "MPEG4", "H264"
    sc.render.ffmpeg.constant_rate_factor = "HIGH"
    sc.render.filepath = os.path.join(tmp, "video")
    bpy.ops.render.render(animation=True)
    made = [f for f in os.listdir(tmp) if f.endswith(".mp4")]
    assert len(made) == 1, made
    os.replace(os.path.join(tmp, made[0]), path)
    shutil.rmtree(tmp)


def beta_label(b):
    return "no damping" if b == 0 else f"beta {b:g} s"


# ---------------------------------------------------------------- show: three plucked arms, side view
D = np.load(os.path.join(out_dir, "show.npz"))
betas = [float(b) for b in D["betas"]]
fps = int(D["fps"])
frames = {b: D[f"x_{b:g}"] for b in betas}
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = setup(len(frames[betas[0]]), (1920, 640), fps)
spacing = 0.32
wall = material("wall", "#8a8984")
zs = np.concatenate([f[:, :, 2].ravel() for f in frames.values()])
z_lo, z_hi = float(zs.min()), float(zs.max())
for i, b in enumerate(betas):
    x0 = spacing * i
    animated_mesh(f"arm {beta_label(b)}", D["F"], frames[b], (x0, 0, 0), material(beta_label(b), COLOR[b]))
    box("wall", (x0 - 0.0125, 0, (z_lo + z_hi) / 2), (0.015, 0.06, z_hi - z_lo + 0.04), wall)
    label(beta_label(b), (x0 + 0.1, 0.05, z_lo - 0.04), 0.022)
width = spacing * (len(betas) - 1) + 0.25
cx, cz = spacing * (len(betas) - 1) / 2 + 0.09, (z_lo + z_hi) / 2 - 0.015
camera((cx, -2.0, cz), (cx, 0.0, cz), ortho=width * 1.06)
clock(sc, (cx, 0.05, z_hi + 0.03), 0.02, (math.radians(90), 0, 0))
# the still: when the undamped arm is at its highest in the last second
tip = np.abs(D["rest"][:, 0] - D["rest"][:, 0].max()) < 1e-9
z_tip = frames[betas[0]][:, tip, 2].mean(1)
late = np.arange(len(z_tip)) >= len(z_tip) - fps
pick = int(np.argmax(np.where(late, z_tip, -np.inf)))
still(sc, pick + 1, os.path.join(out_dir, "show.png"))
video(sc, os.path.join(out_dir, "show.mp4"))
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "show.blend"))

# ---------------------------------------------------------------- spin: cubes spinning about z, 3/4 view
D = np.load(os.path.join(out_dir, "spin.npz"))
betas = [float(b) for b in D["betas"]]
frames = {b: D[f"x_{b:g}"] for b in betas}
bpy.app.handlers.frame_change_pre.clear()
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = setup(len(frames[betas[0]]), (1600, 720), 60)
sc.display.shading.show_shadows = True
spacing = 0.24
for i, b in enumerate(betas):
    animated_mesh(f"cube {beta_label(b)}", D["F"], frames[b], (spacing * i, 0, 0), material(beta_label(b), COLOR[b]))
    label(beta_label(b), (spacing * i, -0.13, -0.09), 0.022, rotation=(0, 0, 0))
cx = spacing * (len(betas) - 1) / 2
box("floor", (cx, 0, -0.092), (spacing * len(betas) + 0.2, 0.4, 0.004), material("floor", "#e4e3df"))
camera((cx, -1.05, 0.62), (cx, 0.0, -0.03), lens=50)
clock(sc, (cx, 0.1, 0.12), 0.024, (math.radians(60), 0, 0))  # above the cubes, facing the camera
every, dt = int(D["every"]), float(D["dt"])
wob = D[f"wobble_{betas[0]:g}"][::every]
late = np.arange(len(wob)) >= len(wob) - 60
pick = int(np.argmax(np.where(late, wob, -np.inf)))
still(sc, pick + 1, os.path.join(out_dir, "spin.png"))
video(sc, os.path.join(out_dir, "spin.mp4"))
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "spin.blend"))
print("RENDER DONE")
