"""Blender: scenes and stills of fem_fiber_demos.py's results (the variants side by side, animated by shape keys).

  blender -b --factory-startup -noaudio --python fem_fiber_demos_render.py -- OUT_DIR

Reads OUT_DIR/octopus.npz and jelly.npz; writes octopus.png, jelly.png, jelly_top.png (the settled state, last
frame) and octopus.blend, jelly.blend (the whole run, one shape key per frame; frame 1 is the start).
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

out_dir = os.path.abspath(sys.argv[sys.argv.index("--") + 1])  # Blender resolves relative render paths elsewhere
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4
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
    for p in me.polygons:
        p.use_smooth = True
    ob.shape_key_add(name="basis")
    for f, X in enumerate(frames):
        kb = ob.shape_key_add(name=f"f{f:04d}")
        kb.data.foreach_set("co", np.asarray(X, np.float32).ravel())
        for fr, val in ((f, 0.0), (f + 1, 1.0), (f + 2, 0.0)):
            kb.value = val
            kb.keyframe_insert("value", frame=fr)
    return ob


def label(text, location, size, rotation=(math.radians(90), 0, 0)):
    cu = bpy.data.curves.new("label", "FONT")
    cu.body, cu.size, cu.align_x = text, size, "CENTER"
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


def setup(n_frames, res):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "STUDIO"
    sc.display.shading.color_type = "MATERIAL"
    sc.display.shading.show_shadows = True
    sc.display.shading.show_cavity = True
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.world = bpy.data.worlds.new("background")
    sc.world.color = rgba("#fcfcfb")[:3]
    sc.frame_start, sc.frame_end = 1, n_frames
    sc.render.fps = 60
    return sc


def render(sc, path):
    sc.frame_set(sc.frame_end)
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


# ---------------------------------------------------------------- octopus
D = np.load(os.path.join(out_dir, "octopus.npz"))
names = [str(n) for n in D["names"]]
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = setup(len(D[names[0]]), (2000, 640))
spacing = 0.8
floor = material("floor", "#e4e3df")
box("floor", (spacing * (len(names) - 1) / 2, 0.2, -0.002), (spacing * len(names) + 0.4, 1.6, 0.004), floor)
hub_mat = material("hub", "#8a8984")
for i, n in enumerate(names):
    x0 = (spacing * i, 0, 0)
    animated_mesh(f"octopus {n}", D["F"], D[n], x0, material(n, PALETTE[i]))
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, location=(x0[0], 0, 0.4))
    bpy.context.object.data.materials.append(hub_mat)
    bpy.ops.object.shade_smooth()
    label(n, (x0[0], -0.42, 0.001), 0.07, rotation=(0, 0, 0))  # flat on the floor: no shadow on the scene
camera((spacing * 1.5, -4.4, 1.7), (spacing * 1.5, 0.0, 0.18), lens=52)
render(sc, os.path.join(out_dir, "octopus.png"))
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "octopus.blend"))

# ---------------------------------------------------------------- jelly
D = np.load(os.path.join(out_dir, "jelly.npz"))
names = [str(n) for n in D["names"]]
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = setup(len(D[names[0]]), (1600, 640))
spacing = 0.16
box("floor", (spacing * (len(names) - 1) / 2, 0.0, -0.001), (spacing * len(names) + 0.1, 0.3, 0.002),
    material("floor", "#e4e3df"))
for i, n in enumerate(names):
    x0 = spacing * i
    animated_mesh(f"jelly {n}", D["F"], D[n], (x0, 0, 0), material(n, PALETTE[i]))
    label(n, (x0, -0.072, 0.001), 0.016, rotation=(0, 0, 0))
camera((spacing, -0.66, 0.32), (spacing, 0.0, 0.03), lens=45)
render(sc, os.path.join(out_dir, "jelly.png"))

# top view: labels flat below each block, and the 80 mm rest footprint drawn just above the blocks
outline = material("rest outline", "#0b0b0b")
for i, ob in enumerate(sorted((o for o in sc.objects if o.name.startswith("label")), key=lambda o: o.location.x)):
    ob.rotation_euler, ob.location = (0, 0, 0), (spacing * i, -0.072, 0.001)
for i in range(len(names)):
    for cx, cy, sx, sy in ((0, 0.04, 0.0815, 0.0015), (0, -0.04, 0.0815, 0.0015),
                           (0.04, 0, 0.0015, 0.0815), (-0.04, 0, 0.0015, 0.0815)):
        o = box("rest outline", (spacing * i + cx, cy, 0.09), (sx, sy, 0.001), outline)
        o.visible_shadow = False
camera((spacing, 0.0, 1.0), (spacing, 0.0, 0.0), ortho=0.5)
sc.display.shading.show_shadows = False  # footprints only
sc.render.resolution_x, sc.render.resolution_y = 1600, 600
render(sc, os.path.join(out_dir, "jelly_top.png"))
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "jelly.blend"))
print("RENDER DONE")
