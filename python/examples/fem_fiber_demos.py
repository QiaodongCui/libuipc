"""Two small scenes showing what the AnisotropicFiber material does (headless; render with fem_fiber_demos_render.py).

octopus  Six tapered tentacles (25 cm, 2 cm root radius) cantilevered from a hub, StableNeoHookean E 0.1 MPa,
         under gravity. Each tentacle points in its own direction (every 60 deg), so its fibers do too.
           soft          SNH only: the tentacles droop
           fibers along  + fibers along each tentacle, stiffness 30 E_eff: they hold out
           fibers mixed  the same fibers, but vertical on every other tentacle: only the direction differs
           SNH 31E       isotropic at 31 x E (the same stiffness along the tentacle as "fibers along")
jelly    An 8 cm jelly cube (E 2 kPa, nu 0.45) settling on a frictionless ground under its own weight.
           soft          SNH only: it squashes and bulges the same in x and y
           fibers x      + fibers along x, 20 E_eff: it cannot widen in x, so it bulges in y
           SNH 21E       isotropic at 21 x E: it barely squashes

usage: python fem_fiber_demos.py OUT_DIR [--only octopus|jelly]
Writes OUT_DIR/octopus.npz and jelly.npz (surface triangles and positions per frame for each run) and
OUT_DIR/fiber_demos.txt (the numbers).
"""
import argparse
import itertools
import os
import time

import numpy as np
from uipc import Engine, Logger, Scene, World, view
from uipc.constitution import AnisotropicFiber, ElasticModuli, StableNeoHookean
from uipc.geometry import ground, label_surface, label_triangle_orient, tetmesh
from uipc.stats import SimulationStats
from uipc.unit import MPa, kPa

RHO, GRAV = 1000.0, 9.81


# ---------------------------------------------------------------- meshing
def grid_tets(nx, ny, nz):
    """Grid vertices (integer coordinates) and Kuhn tets (6 per cube, conforming) of an nx x ny x nz block of cubes."""
    I, J, K = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1), indexing="ij")
    G = np.stack([I.ravel(), J.ravel(), K.ravel()], axis=1)

    def vid(c):
        return (c[:, 0] * (ny + 1) + c[:, 1]) * (nz + 1) + c[:, 2]

    cells = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), -1).reshape(-1, 3)
    tets = []
    for perm in itertools.permutations(range(3)):
        corner, path = np.zeros(3, int), [np.zeros(3, int)]
        for ax in perm:
            corner = corner.copy()
            corner[ax] = 1
            path.append(corner)
        tets.append(np.stack([vid(cells + c) for c in path], axis=1))
    return G, np.concatenate(tets)


def orient(V, T):
    e = V[T[:, 1:]] - V[T[:, :1]]
    neg = np.linalg.det(e) < 0
    T = T.copy()
    T[neg] = T[neg][:, [0, 1, 3, 2]]
    return T


def boundary_faces(T):
    """Outward-oriented boundary triangles of a positively oriented tet mesh."""
    faces = np.concatenate([T[:, [1, 2, 3]], T[:, [0, 3, 2]], T[:, [0, 1, 3]], T[:, [0, 2, 1]]])
    key = np.sort(faces, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return faces[cnt[inv.ravel()] == 1]


def tentacle(theta, L=0.25, r0=0.02, taper=0.7, hub=0.04, z0=0.4, nx=50, n=4):
    """Tapered round tentacle along (cos theta, sin theta, 0) from radius `hub`, cross-section a square grid mapped
    to a disk. Returns V, T, the fixed root vertices, the axis and the tip vertices."""
    G, T = grid_tets(nx, n, n)
    s = G[:, 0] / nx * L
    u, v = 2 * G[:, 1] / n - 1, 2 * G[:, 2] / n - 1
    du, dv = u * np.sqrt(1 - v**2 / 2), v * np.sqrt(1 - u**2 / 2)  # square -> disk, orientation-preserving
    r = r0 * (1 - taper * s / L)
    axis = np.array([np.cos(theta), np.sin(theta), 0.0])
    side = np.array([-np.sin(theta), np.cos(theta), 0.0])
    V = (hub + s)[:, None] * axis + (r * du)[:, None] * side + (r * dv)[:, None] * np.array([0, 0, 1.0])
    V[:, 2] += z0
    return V, orient(V, T), G[:, 0] == 0, axis, G[:, 0] == nx


def jelly_cube(side=0.08, n=16, gap=0.002):
    G, T = grid_tets(n, n, n)
    V = G * (side / n) - np.array([side / 2, side / 2, 0.0])
    V[:, 2] += gap
    return V, orient(V, T)


# ---------------------------------------------------------------- one run
def run(meshes, E, nu, fibers, out_dir, name, seconds, dt, contact=False, iso_scale=1.0):
    """meshes: list of (V, T, fixed). fibers: list (per mesh) of (multiplier of E_eff, direction) or None.
    Returns positions per frame of each mesh's vertices, Newton iterations per frame, wall time."""
    cfg = Scene.default_config()
    cfg["dt"] = dt
    cfg["gravity"] = [[0.0], [0.0], [-GRAV]]
    cfg["contact"]["enable"] = 1 if contact else 0
    cfg["contact"]["d_hat"] = 0.001  # the default 1 cm is wider than the 5 mm mesh spacing
    cfg["newton"]["velocity_tol"] = 1e-3
    cfg["newton"]["semi_implicit"]["enable"] = 0
    cfg["linear_system"]["tol_rate"] = 1e-6
    scene = Scene(cfg)
    if contact:
        scene.contact_tabular().default_model(0.0, 1.0 * MPa)  # frictionless
        scene.objects().create("ground").geometries().create(ground(0.0, np.array([[0.0], [0.0], [1.0]])))
    obj = scene.objects().create(name)
    slots = []
    for (V, T, fixed), fib in zip(meshes, fibers):
        m = tetmesh(V, T)
        label_surface(m)
        label_triangle_orient(m)
        StableNeoHookean().apply_to(m, ElasticModuli.youngs_poisson(E * iso_scale, nu), mass_density=RHO)
        if fib is not None:
            mu_s = float(view(m.tetrahedra().find("mu"))[0])
            lam_s = float(view(m.tetrahedra().find("lambda"))[0])
            mu, lam = mu_s, lam_s - mu_s  # small-strain Lame parameters of libuipc's SNH (SNK1)
            E_eff = mu * (3 * lam + 2 * mu) / (lam + mu)
            AnisotropicFiber().apply_to(m, fib[0] * E_eff, np.array(fib[1], float))
        if fixed is not None:
            f = view(m.vertices().find("is_fixed"))
            f[:] = fixed.astype(f.dtype).reshape(f.shape)
        slot, _ = obj.geometries().create(m)
        slots.append(slot)
    stats = SimulationStats()
    engine = Engine("cuda", os.path.join(out_dir, "workspace_" + name.replace(" ", "_")))
    world = World(engine)  # World keeps only a weak reference to its engine
    world.init(scene)
    if not world.is_valid():
        raise SystemExit(f"{name}: world.init failed")
    frames, t0 = [[V.copy() for V, _, _ in meshes]], time.time()
    for _ in range(round(seconds / dt)):
        world.advance()
        world.retrieve()
        stats.collect()
        frames.append([s.geometry().positions().view().reshape(-1, 3).copy() for s in slots])
    return frames, stats.get_values("Newton Iteration", "count")[1], time.time() - t0


# ---------------------------------------------------------------- the demos
def octopus(out_dir, log):
    E, nu, k = 0.1 * MPa, 0.4, 30.0
    tents = [tentacle(np.radians(60 * i)) for i in range(6)]
    meshes = [(V, T, fixed) for V, T, fixed, _, _ in tents]
    runs = {
        "soft": ([None] * 6, 1.0),
        "fibers along": ([(k, t[3]) for t in tents], 1.0),
        "fibers mixed": ([(k, t[3]) if i % 2 == 0 else (k, (0, 0, 1)) for i, t in enumerate(tents)], 1.0),
        "SNH 31E": ([None] * 6, 1.0 + k),
    }
    out = {"F": [boundary_faces(T) for _, T, _ in meshes], "rest": [V for V, _, _ in meshes]}
    log(f"octopus: 6 tentacles x {len(meshes[0][1])} tets, E {E / 1e6:g} MPa, nu {nu}, fibers {k:g} E_eff")
    for name, (fibers, scale) in runs.items():
        frames, newton, wall = run(meshes, E, nu, fibers, out_dir, "octopus " + name, 2.5, 1 / 60, iso_scale=scale)
        drops = [1000 * (V[tip, 2].mean() - frames[-1][i][tip, 2].mean()) for i, (V, _, _, _, tip) in enumerate(tents)]
        out[name] = np.array([np.concatenate(f) for f in frames], np.float32)
        log(f"  {name:13s} tip drop per tentacle (mm): " + " ".join(f"{d:6.1f}" for d in drops)
            + f"   Newton/step {newton.mean():.2f}, wall {wall:.1f} s")
    np.savez(os.path.join(out_dir, "octopus.npz"), offsets=np.cumsum([0] + [len(V) for V, _, _ in meshes]),
             F=np.concatenate([F + o for F, o in zip(out["F"], np.cumsum([0] + [len(V) for V, _, _ in meshes]))]),
             names=np.array(list(runs)), **{n: out[n] for n in runs})


def jelly(out_dir, log):
    E, nu, k = 2 * kPa, 0.45, 20.0
    V, T = jelly_cube()
    runs = {"soft": (None, 1.0), "fibers x": ((k, (1, 0, 0)), 1.0), "SNH 21E": (None, 1.0 + k)}
    log(f"jelly: {len(T)} tets, 8 cm cube, E {E / 1e3:g} kPa, nu {nu}, fibers {k:g} E_eff, frictionless ground")
    out = {}
    for name, (fib, scale) in runs.items():
        frames, newton, wall = run([(V, T, None)], E, nu, [fib], out_dir, "jelly " + name, 3.0, 1 / 60,
                                   contact=True, iso_scale=scale)
        P = frames[-1][0]
        mid = np.abs(V[:, 2] - V[:, 2].mean()) < 0.006  # the vertices around mid-height
        wx, wy = np.ptp(P[mid, 0]) * 1000, np.ptp(P[mid, 1]) * 1000
        h = (P[:, 2].max() - P[:, 2].min()) * 1000
        out[name] = np.array([f[0] for f in frames], np.float32)
        log(f"  {name:9s} height {h:5.1f} mm, width at mid-height x {wx:5.1f} mm, y {wy:5.1f} mm (rest 80 x 80 x 80)"
            f"   Newton/step {newton.mean():.2f}, wall {wall:.1f} s")
    np.savez(os.path.join(out_dir, "jelly.npz"), F=boundary_faces(T), names=np.array(list(runs)), **out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--only", choices=["octopus", "jelly"])
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    Logger.set_level(Logger.Level.Warn)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    if a.only in (None, "octopus"):
        octopus(a.out_dir, log)
    if a.only in (None, "jelly"):
        jelly(a.out_dir, log)
    with open(os.path.join(a.out_dir, "fiber_demos.txt"), "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
