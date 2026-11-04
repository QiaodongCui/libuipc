"""Cantilever arms clamped in a fixed plate, sagging under gravity: isotropic vs anisotropic (fiber) FEM.

Two arms with the same 2 x 2 cm cross-section, each one tet mesh together with its plate. Every plate vertex is
fixed, so the arm's root face is clamped.
  slender  20 cm long (L/h = 10): the sag is almost all bending, set by the stiffness along the arm.
  stubby    4 cm long (L/h = 2):  a fifth of the sag is shear.
Runs per arm:
  SNH E            isotropic StableNeoHookean
  SNH 10E          the same at 10 x E: the only way to stiffen an arm without fibers
  fibers along     SNH E + AnisotropicFiber along the arm, fiber stiffness 9 E_eff (matches 10 E in bending)
  fibers vertical  SNH E + AnisotropicFiber along z, 9 E_eff (barely touches the bending)
Gravity, no contact, BDF1 at a large time step, whose numerical damping settles the arm. The settled tip sag is
printed next to Timoshenko beam theory. Fibers along the arm sag about like 10 E on the slender arm but several
times more on the stubby one, because they stiffen the arm along its length and not in shear.

The moduli libuipc 0.0.29 uses differ from the requested E and nu. StableNeoHookean stores mu = 4/3 mu_L and
lambda = lambda_L + 5/6 mu_L, and its energy (SNK1: 0.5 mu (Ic - 3) + 0.5 lambda (J - 1 - mu/lambda)^2) linearizes
to the Lame parameters (mu, lambda - mu). Beam theory uses those, read back from the mesh. The fiber term
mu_f/2 (|F a| - 1)^2 adds mu_f to the small-strain stiffness along a and nothing in shear.

usage: python fem_cantilever_anisotropy.py OUT_DIR [--seconds S] [--h SPACING]
Writes OUT_DIR/cantilever.txt (the table), cantilever.npz (sag over time, centrelines) and cantilever.png.
"""
import argparse
import itertools
import os
import time

import numpy as np
from uipc import Engine, Logger, Scene, World, view
from uipc.constitution import AnisotropicFiber, ElasticModuli, StableNeoHookean
from uipc.geometry import label_surface, label_triangle_orient, tetmesh
from uipc.stats import SimulationStats
from uipc.unit import MPa, kPa

RHO, GRAV, NU = 1000.0, 9.81, 0.3
SIDE, PLATE_SIDE, PLATE_T = 0.02, 0.04, 0.005  # arm cross-section, plate size and thickness (m)
ARMS = {"slender": dict(L=0.20, E=2.5 * MPa), "stubby": dict(L=0.04, E=40 * kPa)}  # E: about 10 % / 6 % sag
# name: (E scale, fiber stiffness in units of E_eff, fiber direction)
VARIANTS = {
    "SNH E": (1.0, 0.0, None),
    "SNH 10E": (10.0, 0.0, None),
    "fibers along": (1.0, 9.0, (1.0, 0.0, 0.0)),
    "fibers vertical": (1.0, 9.0, (0.0, 0.0, 1.0)),
}


def arm_mesh(L, h):
    """Plate (x in [-PLATE_T, 0]) and arm (x in [0, L], centred in y and z) on one grid of spacing h, 6 tets a cube."""
    n = {k: round(v / h) for k, v in dict(p=PLATE_T, a=L, s=PLATE_SIDE, c=SIDE).items()}
    for k, v in dict(p=PLATE_T, a=L, s=PLATE_SIDE, c=SIDE).items():
        assert abs(n[k] * h - v) < 1e-9, f"{v} m is not a multiple of h = {h} m"
    nx, ns = n["p"] + n["a"], n["s"]
    lo, hi = (ns - n["c"]) // 2, (ns + n["c"]) // 2
    cells = [(i, j, k) for i in range(nx) for j in range(ns) for k in range(ns)
             if i < n["p"] or (lo <= j < hi and lo <= k < hi)]
    cells = np.array(cells)

    def vid(ijk):
        return (ijk[:, 0] * (ns + 1) + ijk[:, 1]) * (ns + 1) + ijk[:, 2]

    tets = []
    for perm in itertools.permutations(range(3)):  # Kuhn subdivision: conforming across cubes
        corner, path = np.zeros(3, int), [np.zeros(3, int)]
        for ax in perm:
            corner = corner.copy()
            corner[ax] = 1
            path.append(corner)
        tets.append(np.stack([vid(cells + c) for c in path], axis=1))
    T = np.concatenate(tets)
    used, T = np.unique(T, return_inverse=True)
    T = T.reshape(-1, 4)
    i, r = np.divmod(used, (ns + 1) ** 2)
    j, k = np.divmod(r, ns + 1)
    V = np.stack([-PLATE_T + i * h, -PLATE_SIDE / 2 + j * h, -PLATE_SIDE / 2 + k * h], axis=1)
    e = V[T[:, 1:]] - V[T[:, :1]]
    neg = np.linalg.det(e) < 0
    T[neg] = T[neg][:, [0, 1, 3, 2]]  # positive orientation for every tet
    return V, T


def effective(mu_s, lam_s):
    """Small-strain Lame mu, lambda of libuipc's StableNeoHookean (SNK1) from its stored mu and lambda."""
    return mu_s, lam_s - mu_s


def moduli(mu, lam, mu_f=0.0, a=None):
    """E along x (uniaxial stress), nu_xy, G of the isotropic material plus a fiber of stiffness mu_f along a
    coordinate axis (it adds mu_f to that axis' normal stiffness and nothing in shear)."""
    C = np.full((3, 3), lam) + 2 * mu * np.eye(3)
    if mu_f > 0:
        ax = int(np.argmax(np.abs(a)))
        assert np.isclose(abs(a[ax]), 1.0), "beam theory here needs an axis-aligned fiber"
        C[ax, ax] += mu_f
    S = np.linalg.inv(C)  # strains from normal stresses
    return 1 / S[0, 0], -S[0, 1] / S[0, 0], mu


def timoshenko(x, L, E_axial, G, nu):
    """Deflection (m, positive down) of a clamped square beam under its own weight: bending part, shear part."""
    A, I = SIDE**2, SIDE**4 / 12
    w, kappa = RHO * GRAV * A, 10 * (1 + nu) / (12 + 11 * nu)  # Cowper's coefficient, rectangle
    return w * x**2 * (6 * L**2 - 4 * L * x + x**2) / (24 * E_axial * I), w * (L * x - x**2 / 2) / (kappa * G * A)


def run(name, L, E, fiber_mult, fiber_dir, V, T, out_dir, seconds, dt):
    cfg = Scene.default_config()
    cfg["dt"] = dt
    cfg["gravity"] = [[0.0], [0.0], [-GRAV]]
    cfg["contact"]["enable"] = 0
    cfg["newton"]["velocity_tol"] = 1e-4  # m/s: converged when no vertex moves more than 1e-4 * dt in an iteration
    cfg["newton"]["semi_implicit"]["enable"] = 0
    cfg["linear_system"]["tol_rate"] = 1e-6
    scene = Scene(cfg)
    mesh = tetmesh(V, T)
    label_surface(mesh)
    label_triangle_orient(mesh)
    StableNeoHookean().apply_to(mesh, ElasticModuli.youngs_poisson(E, NU), mass_density=RHO)
    mu_s = float(view(mesh.tetrahedra().find("mu"))[0])
    lam_s = float(view(mesh.tetrahedra().find("lambda"))[0])
    mu_f = 0.0
    if fiber_mult > 0:
        mu, lam = effective(mu_s, lam_s)
        mu_f = fiber_mult * moduli(mu, lam)[0]  # in units of the isotropic E_eff
        AnisotropicFiber().apply_to(mesh, mu_f, np.array(fiber_dir))
        k = view(mesh.tetrahedra().find("fiber_stiffness"))
        plate = (V[T][:, :, 0] < 1e-9).all(axis=1)  # fixed plate tets: no fiber
        k[plate.reshape(k.shape[:1])] = 0.0
    is_fixed = view(mesh.vertices().find("is_fixed"))
    is_fixed[:] = (V[:, 0] < 1e-9).astype(is_fixed.dtype).reshape(is_fixed.shape)  # the plate and the root face
    slot, _ = scene.objects().create(name).geometries().create(mesh)

    stats = SimulationStats()
    engine = Engine("cuda", os.path.join(out_dir, "workspace_" + name.replace(" ", "_")))
    world = World(engine)  # World keeps only a weak reference to its engine
    world.init(scene)
    if not world.is_valid():
        raise SystemExit(f"{name}: world.init failed")
    tip = np.abs(V[:, 0] - L) < 1e-9
    sag, t0 = [], time.time()
    for _ in range(round(seconds / dt)):
        world.advance()
        world.retrieve()
        stats.collect()
        P = slot.geometry().positions().view().reshape(-1, 3)
        sag.append(-(P[tip, 2] - V[tip, 2]).mean())
    newton = stats.get_values("Newton Iteration", "count")[1]
    return dict(sag=np.array(sag), P=P.copy(), mu_s=mu_s, lam_s=lam_s, mu_f=mu_f, wall=time.time() - t0,
                newton=newton)


def centreline(V, P, h):
    """Mean downward displacement of each cross-section of the arm (x >= 0), by grid layer."""
    arm = V[:, 0] > -1e-9
    layer = np.round(V[arm, 0] / h).astype(int)
    dz = -(P[arm, 2] - V[arm, 2])
    xs = np.unique(layer)
    return xs * h, np.array([dz[layer == i].mean() for i in xs])


def plot(results, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    color = dict(zip(VARIANTS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))  # categorical slots 1-4
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=surface, constrained_layout=True)
    for ax, (arm, R) in zip(axes, results.items()):
        ax.set_facecolor(surface)
        ends = []
        for var in VARIANTS:
            x, d = R[var]["line"]
            ax.plot(x * 100, -d * 1000, color=color[var], lw=2, solid_capstyle="round")
            ax.plot(x * 100, -R[var]["theory_line"] * 1000, color=color[var], lw=1.5, ls=(0, (5, 3)))
            ends.append((var, x[-1] * 100, -d[-1] * 1000))
        ends.sort(key=lambda e: e[2])
        span = max(abs(e[2]) for e in ends)
        last = None
        for label, xe, ye in ends:  # direct labels at the tip, nudged apart
            ylab = ye if last is None else max(ye, last + 0.07 * span)
            ax.annotate(label, (xe, ye), xytext=(xe + 0.02 * R["L"] * 100, ylab), color=ink2, fontsize=9, va="center")
            last = ylab
        ax.set_xlim(0, R["L"] * 100 * 1.3)
        ax.set_title(f"{arm}: {R['L'] * 100:.0f} × 2 × 2 cm arm, E_eff = {R['SNH E']['E_eff'] / 1e3:.0f} kPa",
                     color=ink, fontsize=11, loc="left")
        ax.set_xlabel("distance from the plate (cm)", color=ink2)
        ax.set_ylabel("deflection (mm)", color=ink2)
        ax.axhline(0, color=ink2, lw=0.8)
        ax.grid(color=grid, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=9)
    handles = [Line2D([], [], color=color[v], lw=2) for v in VARIANTS]
    handles += [Line2D([], [], color=ink2, lw=2), Line2D([], [], color=ink2, lw=1.5, ls=(0, (5, 3)))]
    labels = [f"{v} (fibers 9 E_eff)" if VARIANTS[v][1] else v for v in VARIANTS] + ["libuipc", "beam theory"]
    fig.legend(handles, labels, loc="outside lower center", ncol=6, frameon=False, labelcolor=ink2, fontsize=9)
    fig.savefig(path, dpi=150, facecolor=surface)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--dt", type=float, default=1 / 60)
    ap.add_argument("--h", type=float, default=0.0025, help="grid spacing (m)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    Logger.set_level(Logger.Level.Warn)

    lines, results, store = [], {}, {}
    hdr = f"{'arm':8s} {'run':16s} {'E_axial':>9s} {'tip sag':>9s} {'beam th.':>9s} {'meas/th':>7s} {'shear':>5s} " \
          f"{'last 0.5 s':>10s} {'Newton/step':>11s} {'wall':>6s}"
    for arm, spec in ARMS.items():
        L, E0 = spec["L"], spec["E"]
        V, T = arm_mesh(L, a.h)
        lines.append(f"{arm}: {len(T)} tets, {len(V)} vertices ({(V[:, 0] < 1e-9).sum()} fixed), h {a.h * 1000:g} mm")
        R = results[arm] = dict(L=L, E=E0)
        for var, (scale, fiber_mult, fiber_dir) in VARIANTS.items():
            r = run(f"{arm} {var}", L, E0 * scale, fiber_mult, fiber_dir, V, T, a.out_dir, a.seconds, a.dt)
            mu, lam = effective(r["mu_s"], r["lam_s"])
            E_eff, nu_eff, G_eff = moduli(mu, lam)
            E_ax, nu_ax, _ = moduli(mu, lam, r["mu_f"], fiber_dir)
            x, d = centreline(V, r["P"], a.h)
            vb, vs = timoshenko(x, L, E_ax, G_eff, nu_eff)
            n_last = max(1, round(0.5 / a.dt))
            R[var] = dict(sag=r["sag"][-1], theory=vb[-1] + vs[-1], shear=vs[-1] / (vb[-1] + vs[-1]),
                          settle=np.ptp(r["sag"][-n_last:]), E_eff=E_eff, E_ax=E_ax, line=(x, d), theory_line=vb + vs,
                          wall=r["wall"], newton=r["newton"].mean())
            store[f"{arm}/{var}/sag"], store[f"{arm}/{var}/x"], store[f"{arm}/{var}/line"] = r["sag"], x, d
            store[f"{arm}/{var}/newton"] = r["newton"]
            q = R[var]
            lines.append(f"{arm:8s} {var:16s} {E_ax / 1e3:7.0f}kPa {q['sag'] * 1000:7.3f}mm {q['theory'] * 1000:7.3f}mm "
                         f"{q['sag'] / q['theory']:7.3f} {q['shear']:5.0%} {q['settle'] * 1e6:8.1f}um "
                         f"{q['newton']:11.2f} {q['wall']:5.1f}s")
        e, e10, fa, fv = (R[v]["sag"] for v in VARIANTS)
        te, te10, tfa, tfv = (R[v]["theory"] for v in VARIANTS)
        lines.append(f"{arm:8s} sag ratios, measured (beam theory): E / 10E {e / e10:.2f} ({te / te10:.2f}), "
                     f"fibers along / 10E {fa / e10:.2f} ({tfa / te10:.2f}), fibers vertical / E {fv / e:.2f} "
                     f"({tfv / te:.2f})")
    table = "\n".join([hdr] + lines)
    print(table)
    with open(os.path.join(a.out_dir, "cantilever.txt"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    np.savez(os.path.join(a.out_dir, "cantilever.npz"), dt=a.dt, h=a.h, **store)
    plot(results, os.path.join(a.out_dir, "cantilever.png"))


if __name__ == "__main__":
    main()
