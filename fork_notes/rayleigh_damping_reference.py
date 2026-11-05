"""NumPy reference for Rayleigh beta damping on tets (plan: fork_notes/rayleigh_damping.md).

Per tet, in libuipc's BDF1 incremental-potential form (elastic energies there are multiplied by dt^2), two forms
(vec: column-major, as libuipc; F_n the step-start deformation gradient):
  "green" (the kernel since plan section 13): damping of the Green strain rate, exactly rotation-free
    E_d = beta dt V / 2 * dE : C : dE,  dE = (F^T F - F_n^T F_n) / 2
    C the small-strain elasticity of the tet's SNK1 (Lame mu, lambda - mu); gradient beta dt V * F (C : dE);
    Hessian beta dt V * (M + G), M[dF, dG] = sym(F^T dF) : C : sym(F^T dG) (PSD; the kernel uses only M) and
    G = I (x) (C : dE), the small geometric term
  "stiffness" (steps 1-4): stiffness-proportional with the PSD-projected SNK1 Hessian at F_n
    E_d = beta dt V / 2 * dF^T H(F_n) dF,  dF = vec(F - F_n); it damps rigid rotation of stressed tets
Both equal beta K(rest) at rest.

Checks (printed with PASS/FAIL; exit 1 on any failure):
  - SNK1: the exact Hessian against finite differences of libuipc's PK1; a port of libuipc's analytic eigensystem
    (snk1::ddEddF_spd: twist/flip modes + 3x3 scaling block) against the exact Hessian, and its clamp against a
    9x9 eigen-clamp
  - green: gradient and exact Hessian against finite differences, the closed-form M against J^T C J, M PSD, zero
    energy for a finite rotation in the step of a stressed tet, no net torque, zero for a translation, equal to
    the stiffness form at rest
  - stiffness: gradient and Hessian against finite differences, PSD, zero for a rigid translation, O(theta^4) for a
    small rotation of an unstressed tet, and the rotation leak of a stressed one ((W F)^T H (W F) = -P : W^2 F)
  - the one-tet implicit-Euler oracle (used against libuipc in the tests), both forms: in the linear regime its
    decay per step matches the discrete BDF1 recurrence with damping, and damping lowers it as zeta = beta w / 2

usage: python -I rayleigh_damping_reference.py [--samples N] [--seed S]
"""
import argparse
import sys

import numpy as np

SQ2 = np.sqrt(2.0)


# ---------------------------------------------------------------- helpers
def vec(M):
    return M.reshape(-1, order="F")


def unvec(f):
    return f.reshape(3, 3, order="F")


def rv_svd(F):
    """Rotation-variant SVD (libuipc's math::qr_svd convention): U, V rotations, only sigma_2 may be negative."""
    U, s, Vt = np.linalg.svd(F)
    V = Vt.T.copy()
    if np.linalg.det(U) < 0:
        U[:, 2] *= -1
        s[2] *= -1
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1
        s[2] *= -1
    return U, s, V


def cof(F):
    return np.stack([np.cross(F[:, 1], F[:, 2]), np.cross(F[:, 2], F[:, 0]), np.cross(F[:, 0], F[:, 1])], axis=1)


def skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


def rot_about(axis, angle):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    W = skew(a)
    return np.eye(3) + np.sin(angle) * W + (1 - np.cos(angle)) * W @ W


def dFdx(Dm_inv):
    """(9, 12): vec(F) = dFdx @ x, x vertex-major"""
    D = np.zeros((9, 12))
    for c in range(3):
        for r in range(3):
            for j in range(3):
                D[3 * c + r, 3 * (j + 1) + r] = Dm_inv[j, c]
            D[3 * c + r, r] = -Dm_inv[:, c].sum()
    return D


# ---------------------------------------------------------------- SNK1 (libuipc's StableNeoHookean)
def snk1_energy(mu, lam, F):
    return 0.5 * mu * (np.sum(F * F) - 3) + 0.5 * lam * (np.linalg.det(F) - 1 - mu / lam) ** 2


def snk1_pk1(mu, lam, F):
    """PEPF = mu F + (lambda (J - 1) - mu) dJ/dF  (snk1::dEdF)"""
    return mu * F + (lam * (np.linalg.det(F) - 1) - mu) * cof(F)


def snk1_hessian(mu, lam, F):
    """Exact 9x9: mu I + lambda g g^T + (lambda (J - 1) - mu) d2J/df2, g = vec(cof F)"""
    g = vec(cof(F))
    HJ = np.zeros((9, 9))
    f = [F[:, 0], F[:, 1], F[:, 2]]
    for (i, j, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):  # d2J/df_i df_j = -[f_k]x, antisymmetric in (i, j)
        HJ[3 * i:3 * i + 3, 3 * j:3 * j + 3] = -skew(f[k])
        HJ[3 * j:3 * j + 3, 3 * i:3 * i + 3] = skew(f[k])
    J = np.linalg.det(F)
    return mu * np.eye(9) + lam * np.outer(g, g) + (lam * (J - 1) - mu) * HJ


def snk1_eigensystem(mu, lam, F):
    """Port of snk1::ddEddF_spd before the clamp: eigenvalues (9,) and eigenvectors (9, 9) as columns."""
    J = np.linalg.det(F)
    U, S, V = rv_svd(F)
    ev, Q = np.zeros(9), np.zeros((9, 9))
    ev_scale = lam * (J - 1.0) - mu
    ev[0:3] = mu + S * ev_scale
    ev[3:6] = mu - S * ev_scale
    sV = V / SQ2
    u, v = U.T, sV.T  # columns
    A, B, C = np.outer(u[1], v[2]), np.outer(u[2], v[1]), np.outer(u[0], v[2])
    D, E, Fm = np.outer(u[2], v[0]), np.outer(u[0], v[1]), np.outer(u[1], v[0])
    for k, M in enumerate((B - A, D - C, Fm - E, A + B, C + D, E + Fm)):  # twist, then flip
        Q[:, k] = vec(M)
    s0, s1, s2 = S**2
    ev_scale2 = lam * (2.0 * J - 1.0) - mu
    Ab = np.array([[mu + lam * s1 * s2, ev_scale2 * S[2], ev_scale2 * S[1]],
                   [ev_scale2 * S[2], mu + lam * s0 * s2, ev_scale2 * S[0]],
                   [ev_scale2 * S[1], ev_scale2 * S[0], mu + lam * s0 * s1]])
    w, W = np.linalg.eigh(Ab)
    ev[6:9] = w
    for k in range(3):
        Q[:, 6 + k] = vec(U @ np.diag(W[:, k]) @ V.T)
    return ev, Q


def snk1_hessian_spd(mu, lam, F):
    """snk1::ddEddF_spd: the analytic eigensystem with negative eigenvalues set to 0"""
    ev, Q = snk1_eigensystem(mu, lam, F)
    return (Q * np.maximum(ev, 0)) @ Q.T


def eigen_clamp(H):
    w, Q = np.linalg.eigh(0.5 * (H + H.T))
    return (Q * np.maximum(w, 0)) @ Q.T


# ---------------------------------------------------------------- Green strain rate damping
def small_strain_lame(mu, lam):
    """Lame parameters of SNK1 at rest from its stored mu, lambda; lambda kept >= -2/3 mu so C is PSD."""
    return mu, max(lam - mu, -2.0 * mu / 3.0)


def green_stress(dE, mu_L, lam_L):
    """C : dE"""
    return 2.0 * mu_L * dE + lam_L * np.trace(dE) * np.eye(3)


def green_metric(F, mu_L, lam_L):
    """9x9 M[dF, dG] = sym(F^T dF) : C : sym(F^T dG): entry (3b + i, 3d + k) =
    mu_L ((F F^T)_ik delta_bd + F_id F_kb) + lam_L F_ib F_kd  (the kernel's closed form)"""
    FFt = F @ F.T
    M = np.zeros((9, 9))
    for b in range(3):
        for i in range(3):
            for d in range(3):
                for k in range(3):
                    M[3 * b + i, 3 * d + k] = (mu_L * (FFt[i, k] * (b == d) + F[i, d] * F[k, b])
                                               + lam_L * F[i, b] * F[k, d])
    return M


def green_geometric(S):
    """9x9 G = I (x) S: entry (3b + i, 3d + k) = delta_ik S_bd"""
    return np.kron(S, np.eye(3))


# ---------------------------------------------------------------- the damping term on one tet
class Tet:
    def __init__(self, Xbar, mu, lam, beta, form="green"):
        Dm = (Xbar[1:] - Xbar[0]).T
        self.Dm_inv = np.linalg.inv(Dm)
        self.V = np.linalg.det(Dm) / 6
        assert self.V > 0
        self.D = dFdx(self.Dm_inv)
        self.mu, self.lam, self.beta = mu, lam, beta
        assert form in ("green", "stiffness")
        self.form = form
        self.mu_L, self.lam_L = small_strain_lame(mu, lam)

    def F(self, X):
        return (X[1:] - X[0]).T @ self.Dm_inv

    # elastic, libuipc scaling (times dt^2)
    def elastic(self, X, dt):
        return dt * dt * self.V * snk1_energy(self.mu, self.lam, self.F(X))

    def elastic_grad(self, X, dt):
        return dt * dt * self.V * self.D.T @ vec(snk1_pk1(self.mu, self.lam, self.F(X)))

    def elastic_hess(self, X, dt):
        return dt * dt * self.V * self.D.T @ snk1_hessian_spd(self.mu, self.lam, self.F(X)) @ self.D

    # damping, given the step-start positions Xn
    def dE(self, X, Xn):
        F, Fn = self.F(X), self.F(Xn)
        return 0.5 * (F.T @ F - Fn.T @ Fn)

    def damping(self, X, Xn, dt):
        if self.form == "green":
            dE = self.dE(X, Xn)
            return 0.5 * self.beta * dt * self.V * np.sum(dE * green_stress(dE, self.mu_L, self.lam_L))
        Hn = snk1_hessian_spd(self.mu, self.lam, self.F(Xn))
        df = vec(self.F(X) - self.F(Xn))
        return 0.5 * self.beta * dt * self.V * df @ Hn @ df

    def damping_grad(self, X, Xn, dt):
        if self.form == "green":
            S = green_stress(self.dE(X, Xn), self.mu_L, self.lam_L)
            return self.beta * dt * self.V * self.D.T @ vec(self.F(X) @ S)
        Hn = snk1_hessian_spd(self.mu, self.lam, self.F(Xn))
        return self.beta * dt * self.V * self.D.T @ Hn @ vec(self.F(X) - self.F(Xn))

    def damping_hess(self, X, Xn, dt, exact=False):
        """green: M (the kernel's PSD Hessian), plus the geometric term G if exact; stiffness: constant in the step"""
        if self.form == "green":
            H = green_metric(self.F(X), self.mu_L, self.lam_L)
            if exact:
                H = H + green_geometric(green_stress(self.dE(X, Xn), self.mu_L, self.lam_L))
            return self.beta * dt * self.V * self.D.T @ H @ self.D
        Hn = snk1_hessian_spd(self.mu, self.lam, self.F(Xn))
        return self.beta * dt * self.V * self.D.T @ Hn @ self.D


def implicit_euler(tet, Xbar, X0, V0, mass, fixed, dt, steps, gravity=np.zeros(3)):
    """One tet under SNK1 + damping, BDF1 as libuipc: minimize 1/2 |x - x~|_M^2 + E_el + E_d per step (Newton with
    backtracking, fixed vertices held), v = (x - x_n) / dt. Returns positions (steps + 1, 4, 3)."""
    free = np.repeat(~fixed, 3)
    m3 = np.repeat(mass, 3)
    X, Vel = X0.copy(), V0.copy()
    out = [X.copy()]
    for _ in range(steps):
        Xn = X.copy()
        xt = (Xn + dt * Vel + dt * dt * gravity).reshape(-1)
        xt[~free] = Xn.reshape(-1)[~free]

        def energy(x):
            Xx = x.reshape(4, 3)
            return 0.5 * np.sum(m3 * (x - xt) ** 2) + tet.elastic(Xx, dt) + tet.damping(Xx, Xn, dt)

        def grad(x):
            Xx = x.reshape(4, 3)
            g = m3 * (x - xt) + tet.elastic_grad(Xx, dt) + tet.damping_grad(Xx, Xn, dt)
            return np.where(free, g, 0.0)

        x = Xn.reshape(-1).copy()
        for _ in range(100):
            Xx = x.reshape(4, 3)
            g = grad(x)
            H = np.diag(m3) + tet.elastic_hess(Xx, dt) + tet.damping_hess(Xx, Xn, dt, exact=True)
            dx = np.zeros(12)
            dx[free] = np.linalg.solve(H[np.ix_(free, free)], -g[free])
            if np.abs(dx).max() < 1e-14:
                break
            # Armijo on the energy, but the energy (SNK1's rest energy included) has round-off ~1e-16 of its size:
            # a step whose decrease is below that is judged by the gradient instead (else Newton stalls short)
            a, e0, gn = 1.0, energy(x), np.linalg.norm(g)
            while a > 1e-8:
                e1 = energy(x + a * dx)
                if e1 <= e0 + 1e-4 * a * g @ dx:
                    break
                if abs(e1 - e0) <= 1e-13 * abs(e0) and np.linalg.norm(grad(x + a * dx)) < gn:
                    break
                a *= 0.5
            x = x + a * dx
        Vel = (x.reshape(4, 3) - Xn) / dt
        X = x.reshape(4, 3)
        out.append(X.copy())
    return np.array(out)


# ---------------------------------------------------------------- checks
def rel(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)


def random_F(rng, kind):
    I = np.eye(3)
    if kind == "general":
        G = I + 0.4 * rng.normal(size=(3, 3))
        while np.linalg.det(G) <= 0.05:
            G = I + 0.4 * rng.normal(size=(3, 3))
        return G
    if kind == "compressed":
        return np.diag(rng.uniform(0.3, 0.9, 3)) @ (I + 0.1 * rng.normal(size=(3, 3)))
    u = rng.normal(size=3)
    u /= np.linalg.norm(u)
    return (I - 2 * np.outer(u, u)) @ (I + 0.3 * rng.normal(size=(3, 3)))


def random_tet(rng):
    while True:
        X = rng.normal(size=(4, 3)) * 0.1
        vol = np.linalg.det((X[1:] - X[0]).T) / 6
        edges = [np.linalg.norm(X[i] - X[j]) for i in range(4) for j in range(i + 1, 4)]
        if abs(vol) > 0.05 * max(edges) ** 3:
            return X if vol > 0 else X[[0, 1, 3, 2]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--samples", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    worst, ok = {}, True

    def note(k, v):
        worst[k] = max(worst.get(k, 0.0), v)

    clamped = 0
    kinds = ["general", "compressed", "inverted"]
    for i in range(args.samples):
        mu, lam = rng.uniform(0.5, 2.0), rng.uniform(0.5, 20.0)
        F = random_F(rng, kinds[i % 3])
        # SNK1: exact Hessian vs finite differences of the PK1
        h = 1e-6
        f = vec(F)
        fd = np.stack([(vec(snk1_pk1(mu, lam, unvec(f + h * e))) - vec(snk1_pk1(mu, lam, unvec(f - h * e)))) / (2 * h)
                       for e in np.eye(9)], axis=1)
        H = snk1_hessian(mu, lam, F)
        note("SNK1: exact Hessian vs finite differences of the PK1", rel(H, fd))
        # PK1 vs finite differences of the energy (the PK1 is libuipc's snk1::dEdF)
        fdg = np.array([(snk1_energy(mu, lam, unvec(f + h * e)) - snk1_energy(mu, lam, unvec(f - h * e))) / (2 * h)
                        for e in np.eye(9)])
        note("SNK1: PK1 vs finite differences of the energy", rel(vec(snk1_pk1(mu, lam, F)), fdg))
        # libuipc's analytic eigensystem (ported) vs the exact Hessian
        ev, Q = snk1_eigensystem(mu, lam, F)
        note("SNK1 port: Q^T Q = I", np.abs(Q.T @ Q - np.eye(9)).max())
        note("SNK1 port: Q diag(ev) Q^T vs exact Hessian", rel((Q * ev) @ Q.T, H))
        note("SNK1 port: clamp vs 9x9 eigen-clamp", rel(snk1_hessian_spd(mu, lam, F), eigen_clamp(H)))
        clamped += int((ev < 0).any())

    # both damping forms on random tets
    for i in range(args.samples):
        mu, lam, beta, dt = rng.uniform(0.5, 2.0), rng.uniform(0.5, 20.0), rng.uniform(0.001, 0.05), 1 / 480
        Xbar = random_tet(rng)
        Xn = Xbar @ random_F(rng, kinds[i % 3]).T + 0.01 * rng.normal(size=(4, 3))
        X = Xn + 0.003 * rng.normal(size=(4, 3))
        x, h = X.reshape(-1), 1e-7
        for form in ("stiffness", "green"):
            tet = Tet(Xbar, mu, lam, beta, form)
            g = tet.damping_grad(X, Xn, dt)
            fdg = np.array([(tet.damping((x + h * e).reshape(4, 3), Xn, dt) -
                             tet.damping((x - h * e).reshape(4, 3), Xn, dt)) / (2 * h) for e in np.eye(12)])
            note(f"{form}: gradient vs finite differences", rel(g, fdg))
            fdH = np.stack([(tet.damping_grad((x + h * e).reshape(4, 3), Xn, dt) -
                             tet.damping_grad((x - h * e).reshape(4, 3), Xn, dt)) / (2 * h) for e in np.eye(12)], axis=1)
            Hd = tet.damping_hess(X, Xn, dt, exact=True)
            note(f"{form}: {'exact ' if form == 'green' else ''}Hessian vs finite differences", rel(Hd, fdH))
            Hp = tet.damping_hess(X, Xn, dt)
            note(f"{form}: {'M' if form == 'green' else 'Hessian'} PSD (-min eigenvalue / max |H|)",
                 max(0.0, -np.linalg.eigvalsh(Hp).min()) / np.abs(Hp).max())
            note(f"{form}: rigid translation in the step (E_d / |E_d scale|)",
                 tet.damping(Xn + rng.normal(size=3), Xn, dt) / max(tet.damping(X, Xn, dt), 1e-30))
        # green: the kernel's closed-form M against J^T C J, J = d vec(sym(F^T dF)) / d vec(dF)
        F = tet.F(X)
        J = np.stack([vec(0.5 * (F.T @ unvec(e) + unvec(e).T @ F)) for e in np.eye(9)], axis=1)
        C9 = 2 * tet.mu_L * np.eye(9) + tet.lam_L * np.outer(vec(np.eye(3)), vec(np.eye(3)))
        note("green: closed-form M vs J^T C J", rel(green_metric(F, tet.mu_L, tet.lam_L), J.T @ C9 @ J))
        # green: a finite rotation in the step of a stressed tet costs nothing
        Q = rot_about(rng.normal(size=3), rng.uniform(0.2, 1.0))
        c = rng.normal(size=3) * 0.1
        Xr = (Xn - c) @ Q.T + c
        scale = tet.beta * dt * tet.V * tet.mu * np.abs(tet.F(Xn)).max() ** 4
        note("green: finite rotation in the step of a stressed tet (E_d / (beta dt V mu |F|^4))",
             abs(tet.damping(Xr, Xn, dt)) / scale)
        gX = tet.damping_grad(X, Xn, dt).reshape(4, 3)  # rotation invariance in x: no net torque at any x
        note("green: net torque of the damping force (|sum x x f| / (|x| sum |f|))",
             np.linalg.norm(np.cross(X, gX).sum(0)) / (np.abs(X).max() * np.abs(gX).sum()))
        # both forms agree at rest (beta K(rest))
        Hg = Tet(Xbar, mu, lam, beta, "green").damping_hess(Xbar, Xbar, dt)
        Hs = Tet(Xbar, mu, lam, beta, "stiffness").damping_hess(Xbar, Xbar, dt)
        note("green: Hessian at rest vs the stiffness form's (beta K(rest))", rel(Hg, Hs))

    # rotation in the step: unstressed (O(theta^4)) and stressed (O(theta^2), the leak)
    mu, lam, beta, dt = 1.0, 10.0, 0.02, 1 / 480
    Xbar = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]]) * 0.1
    tet = Tet(Xbar, mu, lam, beta, "stiffness")
    axis = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    W = skew(axis)

    def rot(t):
        return np.eye(3) + np.sin(t) * W + (1 - np.cos(t)) * W @ W

    unstressed = [tet.damping(Xbar @ rot(t).T, Xbar, dt) / t**4 for t in (1e-3, 1e-4)]
    note("stiffness: small rotation of an unstressed tet, E_d/theta^4 constant (relative change)",
         abs(unstressed[1] - unstressed[0]) / unstressed[0])
    Fs = np.diag([1.3, 0.95, 0.9])
    Xs = Xbar @ Fs.T
    leak = [tet.damping(Xs @ rot(t).T, Xs, dt) / t**2 for t in (1e-3, 1e-4)]
    WF = W @ Fs
    H = snk1_hessian(mu, lam, Fs)
    note("rotation identity: (WF)^T H (WF) = -P : W^2 F (relative)",
         abs(vec(WF) @ H @ vec(WF) + np.sum(snk1_pk1(mu, lam, Fs) * (W @ W @ Fs))) / abs(vec(WF) @ H @ vec(WF)))
    leak_pred = 0.5 * beta * dt * tet.V * vec(WF) @ snk1_hessian_spd(mu, lam, Fs) @ vec(WF)
    note("stiffness: stressed tet, E_d/theta^2 vs (beta dt V / 2)(WF)^T H_spd (WF) (relative)",
         abs(leak[1] - leak_pred) / leak_pred)
    green_leak = Tet(Xbar, mu, lam, beta, "green").damping(Xs @ rot(1e-3).T, Xs, dt) / 1e-6

    # the oracle: decay per step in the linear regime vs the discrete BDF1 recurrence with damping
    m = np.array([1.0, 0.05, 1.0, 1.0])  # vertex 1 free, light enough to ring
    fixed = np.array([True, False, True, True])
    mu, lam, dt = 1.0e4, 2.0e4, 1 / 480
    K = Tet(Xbar, mu, lam, 0.0)
    K11 = K.elastic_hess(Xbar, 1.0)[3:6, 3:6]  # stiffness of vertex 1 at rest
    w2, modes = np.linalg.eigh(K11 / m[1])
    w, d = np.sqrt(w2[0]), modes[:, 0]
    for form in ("stiffness", "green"):
        results = []
        for beta in (0.0, 0.004):
            tet = Tet(Xbar, mu, lam, beta, form)
            X0 = Xbar.copy()
            X0[1] += 1e-6 * d  # tiny amplitude: linear regime
            traj = implicit_euler(tet, Xbar, X0, np.zeros((4, 3)), m, fixed, dt, 400)
            a = (traj[:, 1] - Xbar[1]) @ d
            # discrete BDF1 with damping for this mode: (1 + c dt + w^2 dt^2) z^2 - (2 + c dt) z + 1 = 0, c = beta w^2
            c = beta * w * w
            z = np.roots([1 + c * dt + w * w * dt * dt, -(2 + c * dt), 1.0])
            rho = np.abs(z).max()
            peaks = [k for k in range(1, len(a) - 1) if a[k] > a[k - 1] and a[k] >= a[k + 1] and a[k] > 1e-3 * a[0]]
            measured = (a[peaks[-1]] / a[peaks[0]]) ** (1 / (peaks[-1] - peaks[0]))
            results.append((beta, rho, measured))
            note(f"oracle ({form}): decay per step vs discrete BDF1, beta {beta}", abs(measured - rho) / (1 - rho))
        zeta_num = -np.log(results[0][1]) / (w * dt)
        zeta_tot = -np.log(results[1][1]) / (w * dt)
        print(f"oracle ({form}) mode: w = {w:.1f} rad/s ({w / 2 / np.pi:.1f} Hz), BDF1 numerical zeta "
              f"{zeta_num:.4f}, with beta 0.004: {zeta_tot:.4f} (beta w / 2 = {0.004 * w / 2:.4f})")
        note(f"oracle ({form}): added zeta vs beta w / 2 (relative)",
             abs((zeta_tot - zeta_num) - 0.004 * w / 2) / (0.004 * w / 2))

    tol = {"default": 1e-6, "finite": 1e-5, "oracle (stiffness): decay": 2e-2, "oracle (green): decay": 2e-2,
           "oracle (stiffness): added": 0.1, "oracle (green): added": 0.1, "stiffness: small rotation": 1e-4,
           "stiffness: stressed": 1e-3, "green: finite rotation": 1e-12, "green: net torque": 1e-12}
    print(f"{args.samples} random F and {args.samples} random tets (seed {args.seed}); "
          f"{clamped} F had a negative SNK1 eigenvalue (clamped)")
    for k, v in worst.items():
        t = next((tol[p] for p in tol if k.startswith(p)), tol["default"])
        if "finite differences" in k:
            t = tol["finite"]
        good = v <= t
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {k:86s} worst {v:.2e} (tol {t:g})")
    print(f"stiffness form, rotation of an unstressed tet: E_d / theta^4 = {unstressed[0]:.3e}; stressed "
          f"(F = diag(1.3, 0.95, 0.9)): E_d / theta^2 = {leak[1]:.3e}; green form, stressed: {green_leak:.3e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
