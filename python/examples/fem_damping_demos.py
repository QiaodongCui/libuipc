"""Scenes for the RayleighDamping extra constitution (stiffness-proportional damping); render with
fem_damping_demos_render.py.

pluck  The slender cantilever arm of fem_cantilever_anisotropy.py (20 x 2 x 2 cm, StableNeoHookean E 2.5 MPa, clamped
       in a fixed plate), no gravity, released from a small displacement along its first two bending modes in one
       plane. The modes come from libuipc's own linearization: the SNK1 stiffness at rest and the lumped vertex
       masses. (The Kuhn tets make the arm a little stiffer along one diagonal of its cross-section, so it bends
       along the diagonals, each pair of modes 1.7 % apart: a plain vertical pluck would beat.)
       Each mode's coordinate in libuipc's motion is a damped oscillation; its per-step decay is fitted (an exact
       2-step recurrence) and compared with the BDF1 prediction for that mode, with beta 0 (BDF1's own numerical
       damping) and beta > 0. Rayleigh beta damping adds the damping ratio beta w / 2 to a mode of angular frequency
       w, so the second mode (about 6 x the frequency) gets about 6 x the damping.
spin   A free 10 cm cube (E 10 kPa, nu 0.4), no gravity, spinning at 1.5 turns/s about z and released stretched
       10 % along x, so it wobbles while it spins. Damping should stop the wobble and keep the spin: its angular
       momentum, against beta 0 (BDF1 loses some by itself).
show   Three plucked arms under gravity at 480 Hz (the flesh's rate), tip lifted 40 mm and released, beta 0, 0.005
       and 0.02 s: the scene for the render.

usage: python fem_damping_demos.py OUT_DIR [--only pluck|spin|show]
Writes OUT_DIR/damping_demos.txt (the numbers, appended), damping_demos.png (pluck and spin plots),
pluck.npz, spin.npz and show.npz.
"""
import argparse
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling examples (python -I leaves it out)
from fem_cantilever_anisotropy import arm_mesh  # noqa: E402
from fem_fiber_demos import boundary_faces, grid_tets, orient  # noqa: E402

from uipc import Engine, Logger, Scene, World, builtin, view  # noqa: E402
from uipc.constitution import ElasticModuli, RayleighDamping, StableNeoHookean  # noqa: E402
from uipc.geometry import label_surface, label_triangle_orient, tetmesh  # noqa: E402
from uipc.stats import SimulationStats  # noqa: E402
from uipc.unit import MPa, kPa  # noqa: E402

RHO, GRAV = 1000.0, 9.81
ARM = dict(L=0.20, E=2.5 * MPa, nu=0.3, h=0.0025)
COLOR = {0.0: "#2a78d6", 0.002: "#eb6834", 0.005: "#1baf7a", 0.02: "#eda100", 0.05: "#e87ba4"}  # slots 1-5


# ---------------------------------------------------------------- scene helpers
def with_material(m, E, nu, beta):
    label_surface(m)
    label_triangle_orient(m)
    StableNeoHookean().apply_to(m, ElasticModuli.youngs_poisson(E, nu), mass_density=RHO)
    if beta > 0:
        RayleighDamping().apply_to(m, beta)
    return m


def body(X0, X_rest, T, E, nu, beta, fixed=None, velocity=None):
    """A tet mesh starting at X0 with rest shape X_rest. The rest geometry carries the same constitutions: a bare
    one crashes libuipc at init."""
    mesh = with_material(tetmesh(X0, T), E, nu, beta)
    rest = with_material(tetmesh(X_rest, T), E, nu, beta)
    if fixed is not None:
        f = view(mesh.vertices().find(builtin.is_fixed))
        f[:] = fixed.astype(f.dtype).reshape(f.shape)
    if velocity is not None:
        v = view(mesh.vertices().find(builtin.velocity))
        v[:] = velocity.reshape(v.shape)
    return mesh, rest


def config(dt, gravity, velocity_tol, tol_rate):
    cfg = Scene.default_config()
    cfg["dt"] = dt
    cfg["gravity"] = [[0.0], [0.0], [-GRAV if gravity else 0.0]]
    cfg["contact"]["enable"] = 0
    cfg["newton"]["velocity_tol"] = velocity_tol
    cfg["newton"]["semi_implicit"]["enable"] = 0
    cfg["linear_system"]["tol_rate"] = tol_rate
    return cfg


def simulate(bodies, cfg, steps, out_dir, name, observe):
    """Run `steps` steps; observe(step, positions of each body) after every step (step 0 = the start)."""
    scene = Scene(cfg)
    obj = scene.objects().create(name)
    slots = [obj.geometries().create(mesh, rest)[0] for mesh, rest in bodies]
    stats = SimulationStats()
    engine = Engine("cuda", os.path.join(out_dir, "workspace_" + name.replace(" ", "_")))
    world = World(engine)  # World keeps only a weak reference to its engine
    world.init(scene)
    if not world.is_valid():
        raise SystemExit(f"{name}: world.init failed")
    observe(0, [view(m.positions()).reshape(-1, 3).copy() for m, _ in bodies])
    t0 = time.time()
    for n in range(1, steps + 1):
        world.advance()
        world.retrieve()
        stats.collect()
        observe(n, [s.geometry().positions().view().reshape(-1, 3) for s in slots])
    return stats.get_values("Newton Iteration", "count")[1], time.time() - t0


# ---------------------------------------------------------------- the arm's modes, from libuipc's linearization
def stiffness_at_rest(V, T, mu, lam):
    """Stiffness (3n x 3n) of libuipc's StableNeoHookean (SNK1) at rest: linear elasticity with the Lame parameters
    (mu, lambda - mu) of its stored mu and lambda."""
    lam_L = lam - mu
    Dm = np.transpose(V[T[:, 1:]] - V[T[:, :1]], (0, 2, 1))  # columns are the edges
    Dm_inv = np.linalg.inv(Dm)
    vol = np.abs(np.linalg.det(Dm)) / 6
    G = np.concatenate([-Dm_inv.sum(1, keepdims=True), Dm_inv], 1)  # (t, 4, 3) shape gradients
    I3 = np.eye(3)
    Kt = mu * np.einsum("tab,ik->taibk", G @ np.transpose(G, (0, 2, 1)), I3)
    Kt += mu * np.einsum("tak,tbi->taibk", G, G) + lam_L * np.einsum("tai,tbk->taibk", G, G)
    Kt *= vol[:, None, None, None, None]
    dof = 3 * T[:, :, None] + np.arange(3)  # (t, 4, 3)
    rows = np.broadcast_to(dof[:, :, :, None, None], Kt.shape).ravel()
    cols = np.broadcast_to(dof[:, None, None, :, :], Kt.shape).ravel()
    return sp.csr_matrix((Kt.ravel(), (rows, cols)), shape=(3 * len(V),) * 2)


def arm_modes(n_modes=12):
    """Bending modes of the arm, M-normalized: (V, T, fixed, mass per vertex, mu, lambda, list of (w, mode (n, 3),
    plane)). The Kuhn tets make the square arm slightly stiffer along one diagonal of its cross-section, so the
    bending modes bend along the diagonals (plane +1: y = z, -1: y = -z), each pair about 1.7 % apart in frequency.
    Modes are signed so the tip moves up."""
    V, T = arm_mesh(ARM["L"], ARM["h"])
    fixed = V[:, 0] < 1e-9
    m = with_material(tetmesh(V, T), ARM["E"], ARM["nu"], 0.0)
    mu = float(view(m.tetrahedra().find("mu"))[0])
    lam = float(view(m.tetrahedra().find("lambda"))[0])
    mass = RHO * view(m.vertices().find(builtin.volume)).reshape(-1)  # libuipc's lumped vertex masses
    free = np.repeat(~fixed, 3)
    K = stiffness_at_rest(V, T, mu, lam)[free][:, free]
    mf = np.repeat(mass, 3)[free]
    w2, Phi = spla.eigsh(K.tocsc(), k=n_modes, M=sp.diags(mf).tocsc(), sigma=0, which="LM")
    order = np.argsort(w2)
    w2, Phi = w2[order], Phi[:, order]
    tip = np.abs(V[:, 0] - ARM["L"]) < 1e-9
    modes = []
    for k in range(n_modes):
        u = np.zeros(3 * len(V))
        u[free] = Phi[:, k] / np.sqrt(np.sum(mf * Phi[:, k] ** 2))
        u = u.reshape(-1, 3)
        yz = mass @ (u[:, 1] * u[:, 2]) / (mass @ (u[:, 1] ** 2 + u[:, 2] ** 2) / 2)  # +-1 bending, 0 torsion
        if abs(yz) > 0.9 and mass @ (u[:, 1:] ** 2).sum(1) > 0.9:  # bending, not torsion or axial
            modes.append((np.sqrt(w2[k]), u * np.sign(u[tip, 2].mean()), int(np.sign(yz))))
    return V, T, fixed, mass, mu, lam, modes


# ---------------------------------------------------------------- damping of a mode from its coordinate
def fit_roots(q, floor=1e-3):
    """Least-squares fit of q[n+1] = c1 q[n] + c2 q[n-1] (exact for a linear mode under BDF1) over the samples until
    the signal stays below `floor` of its peak; the roots of z^2 - c1 z - c2."""
    big = np.nonzero(np.abs(q) > floor * np.abs(q).max())[0]
    q = q[: big[-1] + 1]
    A = np.stack([q[1:-1], q[:-2]], 1)
    c = np.linalg.lstsq(A, q[2:], rcond=None)[0]
    return np.roots([1.0, -c[0], -c[1]])


def bdf1_roots(w, beta, dt):
    """Roots of BDF1 for a mode of angular frequency w with Rayleigh beta damping:
    (q+ - 2q + q-) / dt^2 = -w^2 q+ - beta w^2 (q+ - q) / dt."""
    a, b = (w * dt) ** 2, beta * w * w * dt
    return np.roots([1 + a + b, -(2 + b), 1.0])


def zeta_of(roots, dt):
    """Damping ratio and natural angular frequency of the continuous mode that decays like the roots per step."""
    s = np.log(roots.astype(complex)) / dt
    s = s[np.argmax(s.imag)]
    return -s.real / abs(s), abs(s)


# ---------------------------------------------------------------- pluck: damping per mode
def pluck(out_dir, log, betas=(0.0, 0.002, 0.005), dt=1 / 1920, seconds=0.6, tips=(1e-3, 3e-4)):
    V, T, fixed, mass, mu, lam, modes = arm_modes()
    plane = modes[0][2]
    (w1, u1, _), (w2, u2, _) = [m for m in modes if m[2] == plane][:2]  # the first two bending in one plane
    tip = np.abs(V[:, 0] - ARM["L"]) < 1e-9
    X0 = V + tips[0] / u1[tip, 2].mean() * u1 + tips[1] / u2[tip, 2].mean() * u2
    proj = np.stack([mass[:, None] * u1, mass[:, None] * u2])  # q_k = u_k^T M (x - X)
    log(f"pluck: arm {len(T)} tets, {(~fixed).sum()} free vertices, mu {mu:.4g}, lambda {lam:.4g} (stored), "
        f"dt 1/{round(1 / dt)}, {seconds:g} s")
    log(f"  bending modes of libuipc's linearization (Hz, plane y = {'+' if plane > 0 else '-'}z first): "
        + ", ".join(f"{w / 2 / np.pi:.3f} ({'+' if p > 0 else '-'})" for w, _, p in modes))
    log(f"  plucked in the plane of the first: f1 {w1 / 2 / np.pi:.3f} Hz, f2 {w2 / 2 / np.pi:.3f} Hz, f2/f1 "
        f"{w2 / w1:.3f} (Euler-Bernoulli 6.267, less with shear); initial tip z {tips[0] * 1e3:g} mm + "
        f"{tips[1] * 1e3:g} mm")
    log(f"  {'beta':>6s} {'mode':>4s} {'zeta meas.':>10s} {'BDF1 pred.':>10s} {'err':>8s} {'f meas.':>8s} "
        f"{'added zeta':>10s} {'beta w/2':>8s} {'Newton':>6s} {'wall':>6s}")
    store, zeta0 = {}, {}
    for beta in betas:
        Q = np.zeros((round(seconds / dt) + 1, 2))

        def observe(n, P):
            Q[n] = np.einsum("kvi,vi->k", proj, P[0] - V)

        bodies = [body(X0, V, T, ARM["E"], ARM["nu"], beta, fixed=fixed)]
        newton, wall = simulate(bodies, config(dt, False, 1e-8, 1e-10), len(Q) - 1, out_dir, f"pluck {beta:g}",
                                observe)
        store[f"q_{beta:g}"] = Q
        for k, w in enumerate((w1, w2)):
            zm, wm = zeta_of(fit_roots(Q[:, k]), dt)
            zp, wp = zeta_of(bdf1_roots(w, beta, dt), dt)
            if beta == 0:
                zeta0[k] = zm
            added = zm - zeta0[k]
            log(f"  {beta:6.3f} {k + 1:4d} {zm:10.5f} {zp:10.5f} {(zm - zp) / zp:8.1e} {wm / 2 / np.pi:7.3f}Hz "
                f"{added:10.5f} {beta * w / 2:8.5f} {newton.mean():6.2f} {wall:5.1f}s")
        if beta > 0:
            r = (zeta_of(fit_roots(Q[:, 1]), dt)[0] - zeta0[1]) / (zeta_of(fit_roots(Q[:, 0]), dt)[0] - zeta0[0])
            log(f"  beta {beta:g}: added zeta mode 2 / mode 1 = {r:.3f}, w2 / w1 = {w2 / w1:.3f}")
    np.savez(os.path.join(out_dir, "pluck.npz"), dt=dt, w=np.array([w1, w2]), betas=np.array(betas), **store)


# ---------------------------------------------------------------- spin: wobble damped, spin kept
def kabsch(X, x, m):
    """Mass-weighted best rotation R and centres with x - c ~ R (X - C)."""
    C, c = m @ X / m.sum(), m @ x / m.sum()
    H = (X - C).T @ (m[:, None] * (x - c))
    U, _, Wt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Wt.T @ U.T))])
    return Wt.T @ D @ U.T, C, c


def spin(out_dir, log, betas=(0.0, 0.02, 0.05), dt=1 / 480, seconds=2.0, turns_per_s=1.5, stretch=1.1):
    E, nu, side, n = 10 * kPa, 0.4, 0.1, 8
    G, T = grid_tets(n, n, n)
    X = G * (side / n) - side / 2
    T = orient(X, T)
    X0 = X * np.array([stretch, stretch**-0.5, stretch**-0.5])
    omega = np.array([0.0, 0.0, 2 * np.pi * turns_per_s])
    v0 = np.cross(omega, X0)
    steps, every = round(seconds / dt), round(1 / (60 * dt))
    log(f"spin: cube {side * 100:g} cm, {len(T)} tets, E {E / 1e3:g} kPa, nu {nu}, {turns_per_s:g} turns/s about z, "
        f"released stretched {stretch:g} x along x (volume kept), no gravity, dt 1/{round(1 / dt)}, {seconds:g} s")
    log(f"  {'beta':>5s} {'L/L0 end':>8s} {'vs beta 0':>9s} {'turns':>6s} {'spin end':>8s} "
        f"{'wobble 0.2 s':>12s} {'wobble end':>10s} {'Newton':>6s} {'wall':>6s}")
    store = {"F": boundary_faces(T), "rest": X, "betas": np.array(betas)}
    L_end0 = None
    for beta in betas:
        P = np.zeros((steps + 1, len(X), 3))

        def observe(k, pos):
            P[k] = pos[0]

        mesh, rest = body(X0, X, T, E, nu, beta, velocity=v0)
        mass = RHO * view(rest.vertices().find(builtin.volume)).reshape(-1)
        newton, wall = simulate([(mesh, rest)], config(dt, False, 1e-6, 1e-8), steps, out_dir, f"spin {beta:g}",
                                observe)
        vel = np.concatenate([v0[None], np.diff(P, axis=0) / dt])  # BDF1: v_n = (x_n - x_{n-1}) / dt
        c = np.einsum("v,nvi->ni", mass, P) / mass.sum()
        L = np.einsum("v,nv->n", mass, np.cross(P - c[:, None], vel)[:, :, 2])
        angle, wobble = np.zeros(steps + 1), np.zeros(steps + 1)
        for k in range(steps + 1):
            R, C, ck = kabsch(X, P[k], mass)
            angle[k] = np.arctan2(R[1, 0], R[0, 0])
            r = P[k] - ck - (X - C) @ R.T
            wobble[k] = np.sqrt(mass @ (r**2).sum(1) / mass.sum())
        angle = np.unwrap(angle)
        last = round(1 / turns_per_s / dt)  # the last turn's mean spin rate
        rate = (angle[-1] - angle[-1 - last]) / (last * dt) / (2 * np.pi)
        tail = round(0.25 / dt)
        wob = lambda a, b: np.ptp(wobble[a:b]) * 1e3  # noqa: E731  (mm, peak to peak)
        L_end0 = L[-1] / L[0] if beta == 0 else L_end0
        log(f"  {beta:5.3f} {L[-1] / L[0]:8.4f} {L[-1] / L[0] / L_end0 - 1:+9.2%} {angle[-1] / 2 / np.pi:6.3f} "
            f"{rate:6.3f}/s {wob(0, round(0.2 / dt)):10.2f}mm {wob(-tail, None):8.2f}mm {newton.mean():6.2f} "
            f"{wall:5.1f}s")
        store[f"x_{beta:g}"] = P[::every].astype(np.float32)
        store[f"L_{beta:g}"], store[f"angle_{beta:g}"], store[f"wobble_{beta:g}"] = L, angle, wobble
    np.savez(os.path.join(out_dir, "spin.npz"), dt=dt, every=every, **store)


# ---------------------------------------------------------------- show: three plucked arms for the render
def show(out_dir, log, betas=(0.0, 0.005, 0.02), dt=1 / 480, seconds=2.5, lift=0.04):
    V, T, fixed, mass, mu, lam, modes = arm_modes()
    tip = np.abs(V[:, 0] - ARM["L"]) < 1e-9
    (_, ua, _), (_, ub, _) = modes[:2]  # first bending mode in each diagonal plane: together a vertical pluck
    X0 = V + lift / 2 * (ua / ua[tip, 2].mean() + ub / ub[tip, 2].mean())
    steps, every = round(seconds / dt), round(1 / (60 * dt))
    frames = {b: [] for b in betas}

    def observe(n, P):
        if n % every == 0:
            for b, p in zip(betas, P):
                frames[b].append(p.astype(np.float32))

    bodies = [body(X0, V, T, ARM["E"], ARM["nu"], b, fixed=fixed) for b in betas]  # no contact: they don't touch
    newton, wall = simulate(bodies, config(dt, True, 1e-4, 1e-6), steps, out_dir, "show", observe)
    log(f"show: {len(betas)} arms, gravity, tip lifted {lift * 1e3:g} mm along mode 1 and released, dt 1/{round(1 / dt)}, "
        f"{seconds:g} s; Newton/step {newton.mean():.2f}, wall {wall:.1f} s")
    store = {"F": boundary_faces(T), "rest": V, "betas": np.array(betas), "fps": 60}
    for b in betas:
        x = np.array(frames[b])
        z = x[:, tip, 2].mean(1) - V[tip, 2].mean()
        swing = lambda a, c: np.ptp(z[a:c]) * 1e3  # noqa: E731
        log(f"  beta {b:5.3f}: tip z from rest at the end {z[-1] * 1e3:6.1f} mm; tip swing peak to peak "
            f"0-0.5 s {swing(0, 31):5.1f} mm, 1-1.5 s {swing(60, 91):5.1f} mm, 2-2.5 s {swing(120, None):5.1f} mm")
        store[f"x_{b:g}"] = x
    np.savez(os.path.join(out_dir, "show.npz"), **store)


# ---------------------------------------------------------------- plot
def plot(out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), facecolor=surface, constrained_layout=True)
    style = dict(lw=1.6, solid_capstyle="round")

    def frame(ax, title, xlabel, ylabel):
        ax.set_facecolor(surface)
        ax.set_title(title, color=ink, fontsize=10.5, loc="left")
        ax.set_xlabel(xlabel, color=ink2)
        ax.set_ylabel(ylabel, color=ink2)
        ax.grid(color=grid, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=9)

    p = os.path.join(out_dir, "pluck.npz")
    if os.path.exists(p):
        D = np.load(p)
        dt, w = float(D["dt"]), D["w"]
        for k, (ax, t_end) in enumerate(zip(axes[0], (0.6, 0.15))):
            for b in D["betas"]:
                q = D[f"q_{b:g}"][:, k]
                t = np.arange(len(q)) * dt
                keep = t <= t_end
                ax.plot(t[keep], q[keep] / q[0], color=COLOR[float(b)], label=f"beta {b:g} s", **style)
            frame(ax, f"pluck, mode {k + 1} ({w[k] / 2 / np.pi:.1f} Hz): its coordinate in libuipc's motion",
                  "time (s)", "mode coordinate / start")
            ax.axhline(0, color=ink2, lw=0.8)
        axes[0, 0].legend(frameon=False, labelcolor=ink2, fontsize=9, loc="upper right")
    p = os.path.join(out_dir, "spin.npz")
    if os.path.exists(p):
        D = np.load(p)
        dt = float(D["dt"])
        for b in D["betas"]:
            L, wob = D[f"L_{b:g}"], D[f"wobble_{b:g}"]
            t = np.arange(len(L)) * dt
            axes[1, 0].plot(t, L / L[0], color=COLOR[float(b)], label=f"beta {b:g} s", **style)
            axes[1, 1].plot(t, wob * 1e3, color=COLOR[float(b)], label=f"beta {b:g} s", **style)
        frame(axes[1, 0], "spin: angular momentum; the damping drags on the rotation (gate: within 1 % of beta 0)",
              "time (s)", "L / L at the start")
        frame(axes[1, 1], "spin: wobble, distance from the best rigid fit of the rest cube", "time (s)", "RMS (mm)")
        axes[1, 0].legend(frameon=False, labelcolor=ink2, fontsize=9, loc="lower left")
    fig.savefig(os.path.join(out_dir, "damping_demos.png"), dpi=140, facecolor=surface)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--only", choices=["pluck", "spin", "show"])
    a = ap.parse_args()
    out_dir = os.path.abspath(a.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    Logger.set_level(Logger.Level.Warn)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    for name, fn in (("pluck", pluck), ("spin", spin), ("show", show)):
        if a.only in (None, name):
            fn(out_dir, log)
    with open(os.path.join(out_dir, "damping_demos.txt"), "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    plot(out_dir)


if __name__ == "__main__":
    main()
