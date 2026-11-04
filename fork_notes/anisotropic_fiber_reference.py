"""NumPy reference for the anisotropic fibre energy (Kim et al. 2019, "Anisotropic ARAP", eq. 31) on one tet.

Plan: fork_notes/anisotropic_fiber.md, step 1. The kernel form ("spring form") works on the fibre vector
d = F a = sum_k w_k x_k:
    Psi = V mu/2 (|d| - s)^2,  s = sign(I4)
    g_k = V w_k mu (1 - s/|d|) d
    H_kl = V w_k w_l M,  M = mu n n^T + max(0, mu (1 - s/|d|)) (I - n n^T),  n = d/|d|
It is checked against the paper's own deformation-gradient form (PK1 eq. 35, Hessian eq. 36, eigenvalues
eqs. 37-38) mapped to vertices with dF/dx, against finite differences, and for positive semi-definiteness.
The sign shortcut (det F > 0 => I4 > 0) is checked against the polar decomposition from a rotation-variant
SVD with libuipc's math::qr_svd convention (|sigma_0| >= |sigma_1| >= |sigma_2|, only sigma_2 may be negative,
U and V proper rotations).

usage: python -I anisotropic_fiber_reference.py [--samples N] [--seed S]
Prints each check's worst error and PASS/FAIL; exits 1 if any check fails.
"""
import argparse
import sys

import numpy as np

EPS_D = 1e-12  # |d| below this: no force, no stiffness (the paper zeroes the Hessian at I5 = 0)


# ---------------------------------------------------------------- the kernel ("spring") form
def rv_svd(F):
    """Rotation-variant SVD, libuipc's qr_svd convention: F = U diag(sig) V^T, det U = det V = 1."""
    U, sig, Vt = np.linalg.svd(F)  # sig >= 0, descending
    V = Vt.T.copy()
    if np.linalg.det(U) < 0:
        U[:, 2] *= -1
        sig[2] *= -1
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1
        sig[2] *= -1
    return U, sig, V


def fibre_sign(F, a):
    """s = sign(I4), I4 = a^T S a with F = R S. det F > 0 => S is positive definite => s = +1 without an SVD.
    I4 = 0 counts as +1 (the paper's patch)."""
    if np.linalg.det(F) > 0:
        return 1.0
    _, sig, V = rv_svd(F)
    return 1.0 if np.sum(sig * (V.T @ a) ** 2) >= 0 else -1.0


class Tet:
    """One tet: rest positions Xbar (4, 3), rest-space fibre direction a, fibre stiffness mu."""

    def __init__(self, Xbar, a, mu):
        Dm = (Xbar[1:] - Xbar[0]).T
        self.Dm_inv = np.linalg.inv(Dm)
        self.V = np.linalg.det(Dm) / 6
        assert self.V > 0, "rest tet must be positively oriented"
        self.a = a / np.linalg.norm(a)
        b = self.Dm_inv @ self.a
        self.w = np.array([-b.sum(), *b])  # d = F a = sum_k w_k x_k
        self.mu = mu

    def F(self, X):
        return (X[1:] - X[0]).T @ self.Dm_inv

    def _parts(self, X):
        d = X.T @ self.w
        return d, np.linalg.norm(d), fibre_sign(self.F(X), self.a)

    def energy(self, X):
        d, nd, s = self._parts(X)
        return self.V * self.mu / 2 * (nd - s) ** 2

    def gradient(self, X):
        """(4, 3): dPsi/dx_k"""
        d, nd, s = self._parts(X)
        if nd < EPS_D:
            return np.zeros((4, 3))
        return self.V * self.mu * (1 - s / nd) * np.outer(self.w, d)

    def hessian(self, X, project=True):
        """(12, 12), vertex-major (x0, y0, z0, x1, ...)"""
        d, nd, s = self._parts(X)
        if nd < EPS_D:
            return np.zeros((12, 12))
        n = d / nd
        lam0, lam12 = self.mu, self.mu * (1 - s / nd)
        if project:
            lam0, lam12 = max(lam0, 0.0), max(lam12, 0.0)
        M = lam0 * np.outer(n, n) + lam12 * (np.eye(3) - np.outer(n, n))
        return self.V * np.kron(np.outer(self.w, self.w), M)


# ---------------------------------------------------------------- the paper's deformation-gradient form
def vec(F):
    return F.reshape(-1, order="F")  # column-major, as in the paper (Appendix A)


def dFdx(Dm_inv):
    """(9, 12): vec(F) = dFdx @ x, x vertex-major"""
    D = np.zeros((9, 12))
    for c in range(3):
        for r in range(3):
            for j in range(3):
                D[3 * c + r, 3 * (j + 1) + r] = Dm_inv[j, c]
            D[3 * c + r, r] = -Dm_inv[:, c].sum()
    return D


def paper_pk1(F, a, mu):
    s, I5 = fibre_sign(F, a), np.sum((F @ a) ** 2)
    return mu * (1 - s / np.sqrt(I5)) * F @ np.outer(a, a)  # eq. 35


def paper_hessian_F(F, a, mu):
    """(9, 9) unprojected, eq. 36: mu [(1 - s/sqrt(I5)) H5 + s I5^(-3/2) f_a f_a^T], H5 = A (x) I3 (Appendix A)"""
    s, I5 = fibre_sign(F, a), np.sum((F @ a) ** 2)
    A = np.outer(a, a)
    fa = vec(F @ A)
    return mu * ((1 - s / np.sqrt(I5)) * np.kron(A, np.eye(3)) + s * I5**-1.5 * np.outer(fa, fa))


def project_psd(H):
    lam, Q = np.linalg.eigh(H)
    return (Q * np.maximum(lam, 0)) @ Q.T


# ---------------------------------------------------------------- samples
def random_tet(rng):
    while True:
        X = rng.normal(size=(4, 3))
        Dm = (X[1:] - X[0]).T
        vol = np.linalg.det(Dm) / 6
        edges = [np.linalg.norm(X[i] - X[j]) for i in range(4) for j in range(i + 1, 4)]
        if abs(vol) > 0.05 * max(edges) ** 3:  # not a sliver
            return X if vol > 0 else X[[0, 1, 3, 2]]


def deform(Xbar, a, kind, rng):
    """deformed positions: 'general' (det F > 0), 'compressed' along the fibre, or 'inverted' (det F < 0)"""
    I = np.eye(3)
    if kind == "general":
        G = I + 0.4 * rng.normal(size=(3, 3))
        while np.linalg.det(G) <= 0.05:
            G = I + 0.4 * rng.normal(size=(3, 3))
    elif kind == "compressed":
        G = (I - rng.uniform(0.3, 0.9) * np.outer(a, a)) @ (I + 0.1 * rng.normal(size=(3, 3)))
    else:
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        G = (I - 2 * np.outer(u, u)) @ (I + 0.3 * rng.normal(size=(3, 3)))
    X = Xbar @ G.T + rng.normal(size=3) + 0.02 * rng.normal(size=(4, 3))
    return X


def rel(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    worst = {}
    counts = {"general": 0, "compressed": 0, "inverted": 0, "inverted, s = -1": 0, "lam12 < 0 (clamped)": 0,
              "fd skipped (near I4 = 0)": 0}

    def note(key, value):
        worst[key] = max(worst.get(key, 0.0), value)

    kinds = ["general", "compressed", "inverted"]
    for i in range(args.samples):
        Xbar = random_tet(rng)
        a = rng.normal(size=3)
        tet = Tet(Xbar, a, mu=rng.uniform(0.5, 5.0))
        kind = kinds[i % 3]
        X = deform(Xbar, tet.a, kind, rng)
        counts[kind] += 1
        F = tet.F(X)
        s = fibre_sign(F, tet.a)
        d = F @ tet.a
        if kind == "inverted" and s < 0:
            counts["inverted, s = -1"] += 1
        if tet.mu * (1 - s / np.linalg.norm(d)) < 0:
            counts["lam12 < 0 (clamped)"] += 1

        # 1. sign shortcut and I4 from the SVD vs the polar decomposition
        U, sig, V = rv_svd(F)
        R, S = U @ V.T, V @ np.diag(sig) @ V.T
        note("polar: |F - R S| / |F|", rel(R @ S, F))
        note("polar: |det R - 1|", abs(np.linalg.det(R) - 1))
        I4 = tet.a @ S @ tet.a
        if np.linalg.det(F) > 0:
            note("sign shortcut: det F > 0 but I4 <= 0 (count)", float(I4 <= 0))
        note("sign: kernel s != sign(a^T S a) (count)", float(s != (1.0 if I4 >= 0 else -1.0)))
        note("d = F a: |sum_k w_k x_k - F a|", rel(X.T @ tet.w, d))

        # 2. spring form vs the paper's F form
        D = dFdx(tet.Dm_inv)
        note("dFdx: |dFdx x - vec F|", rel(D @ X.reshape(-1), vec(F)))
        g_paper = tet.V * D.T @ vec(paper_pk1(F, tet.a, tet.mu))
        note("gradient vs paper (eq. 35)", rel(tet.gradient(X).reshape(-1), g_paper))
        HF = paper_hessian_F(F, tet.a, tet.mu)
        note("Hessian vs paper (eq. 36)", rel(tet.hessian(X, project=False), tet.V * D.T @ HF @ D))
        note("projected Hessian vs paper, 9x9 eigen-clamped",
             rel(tet.hessian(X, project=True), tet.V * D.T @ project_psd(HF) @ D))

        # 3. the paper's eigenvalues (eqs. 37-38): {mu, mu (1 - s/sqrt I5) x 2, 0 x 6}
        lam = np.sort(np.linalg.eigvalsh(HF))
        lam12 = tet.mu * (1 - s / np.linalg.norm(d))
        expect = np.sort([tet.mu, lam12, lam12] + [0.0] * 6)
        note("eigenvalues vs eqs. 37-38", np.abs(lam - expect).max() / tet.mu)

        # 4. positive semi-definite after projection
        Hp = tet.hessian(X, project=True)
        note("projected Hessian: -min eigenvalue / max |H|", max(0.0, -np.linalg.eigvalsh(Hp).min()) / np.abs(Hp).max())

        # 5. finite differences, only where s cannot flip inside the stencil
        if abs(I4) < 1e-3 or abs(sig[2] / sig[1]) > 0.999:
            counts["fd skipped (near I4 = 0)"] += 1
            continue
        h = 1e-6
        x = X.reshape(-1)
        fd_g, fd_H = np.zeros(12), np.zeros((12, 12))
        for k in range(12):
            e = np.zeros(12)
            e[k] = h
            fd_g[k] = (tet.energy((x + e).reshape(4, 3)) - tet.energy((x - e).reshape(4, 3))) / (2 * h)
            fd_H[:, k] = (tet.gradient((x + e).reshape(4, 3)) - tet.gradient((x - e).reshape(4, 3))).reshape(-1) / (2 * h)
        note("finite differences: gradient", rel(tet.gradient(X).reshape(-1), fd_g))
        note("finite differences: Hessian (unprojected)", rel(tet.hessian(X, project=False), fd_H))

    tol = {"count": 0.0, "default": 1e-6, "finite": 1e-5}
    failed = False
    print(f"{args.samples} random tets (seed {args.seed}): " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    for k, v in worst.items():
        t = tol["count"] if "(count)" in k else tol["finite"] if k.startswith("finite") else tol["default"]
        ok = v <= t
        failed |= not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {k:52s} worst {v:.2e}  (tolerance {t:g})")

    # 6. the 1D picture (paper Fig. 2e): F = diag(x, 1, 1), fibre along x
    print("1D sweep, F = diag(x, 1, 1), a = e_x, mu = 1: energy per unit volume")
    a = np.array([1.0, 0.0, 0.0])
    for x in (2.0, 1.0, 0.5, 0.0, -0.5, -0.99, -1.01, -2.0):
        F = np.diag([x, 1.0, 1.0])
        s = fibre_sign(F, a)
        print(f"  x {x:6.2f}  s {s:+.0f}  Psi {0.5 * (abs(x) - s) ** 2:6.3f}   (x - 1)^2 / 2 = {0.5 * (x - 1) ** 2:6.3f}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
