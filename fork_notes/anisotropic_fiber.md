# Plan: an anisotropic fibre material (Kim et al. 2019)

Fork-only note (branch `aniso`); not meant for upstream pull requests. Drafted 2026-11-04.

Source: T. Kim, F. de Goes, H. Iben, "Anisotropic Elasticity for Inversion-Safety and Element Rehabilitation",
ACM TOG 38(4), 2019 (SIGGRAPH). It builds on B. Smith, F. de Goes, T. Kim, "Analytic Eigensystems for Isotropic
Distortion Energies", ACM TOG 2019. Both PDFs: `mmd_tools/projects/libuipc_ref/papers/`.

## 1. Scope

**In:**

- The paper's inversion-safe fibre energy, "Anisotropic ARAP" (eq. 31), added on top of an existing tet material
  (StableNeoHookean, or ARAP unchanged).
- Stiffening only: fibre stiffness >= 0.
- One fibre direction and one stiffness per tet.
- The closed-form Hessian projection (no numerical eigendecomposition).

**Out for now:**

- The inversion test (a mirrored start state): deferred.
- Element rehabilitation (paper §6): not needed for the mmd_tools flesh mesh. Its worst Knupp score is 5e-3; the
  paper rehabilitates below about 1e-5, and none of the 335k tets is under 1e-3.
- Softening (negative fibre stiffness): see the open decisions.
- Muscle contraction (eq. 48) and several fibres per tet.

## 2. The math, as the kernel computes it

Per tet, with `a` the unit fibre direction in the rest shape, `F = Ds Dm^-1`:

- **Fibre vector:** `d = F a = sum_k w_k x_k`, with `b = Dm^-1 a` and `w = (-(b1 + b2 + b3), b1, b2, b3)`. The four
  weights `w` are computed once at init, so `d` is a weighted sum of the tet's four vertex positions.
- **Invariants:** `I5 = |F a|^2 = |d|^2`, so `sqrt(I5) = |d|`. `I4 = a^T S a`, where `F = R S` is the polar
  decomposition; only its sign `s = sign(I4)` is needed.
- **Energy (eq. 31):** `Psi = mu_f / 2 * (|d| - s)^2`, times `V dt^2` (rest volume times dt squared, libuipc's
  scaling of elastic energies, as in `stable_neo_hookean_3d.cu`).
- **The sign, usually without an SVD:** if `det F > 0`, `S` is positive definite, so `I4 > 0` and `s = +1`. Only
  tets with `det F <= 0` need an SVD. The fork's `math::qr_svd` (`src/backends/cuda/algorithm/qr_svd.hpp`) returns
  rotations `U`, `V` with signed singular values, so `I4 = sum_i sigma_i (v_i . a)^2` directly. At `I4 = 0`, take
  `s = +1` (the paper's patch: use the value from `I4 = 2`).
- **Gradient (eq. 35):** for vertex `k`, `g_k = V dt^2 * w_k * mu_f (1 - s / |d|) * d`.
- **Projected Hessian:** with `n = d / |d|`,
  `M = mu_f n n^T + max(0, mu_f (1 - s / |d|)) (I - n n^T)` (3x3), and the block for vertices `k`, `l` is
  `V dt^2 * w_k w_l * M`. This is exactly the paper's eigensystem (eqs. 36-38): eigenvalue `mu` along `F A`,
  `mu (1 - s / sqrt(I5))` on the other two directions of the I5 subspace, zero on the remaining six. No
  eigenvectors and no 9x9 matrix are needed.
- **Guard:** if `|d|` is below about 1e-12, gradient and Hessian are zero (the paper zeroes the Hessian at
  `I5 = 0`; `strain_limiting_baraff_witkin_shell_2d.h` guards its `1/sqrt(I5)` the same way).

Reading: the fibre term is an inversion-aware spring of rest length 1 on the stretched fibre `F a`.

## 3. Design in libuipc

- **An extra FEM energy, not a new primary material.** libuipc already layers energies this way: Kirchhoff rod
  bending and discrete shell bending are `FiniteElementExtraConstitution`s (backend:
  `src/backends/cuda/finite_element/finite_element_extra_constitution.h`). The paper also writes
  `Psi = Psi_iso + Psi_aniso`. The SNH kernel stays untouched (minimal diff), it works on top of ARAP too, and tets
  without fibres are skipped.
  - Cost: `F` is recomputed per fibred tet, each fibred tet adds 10 more 3x3 Hessian blocks, and softening cannot
    be projected together with the isotropic part.
- **Python API** (working name `AnisotropicFiber`):
  ```python
  snh.apply_to(mesh, moduli, mass_density=1000)
  AnisotropicFiber().apply_to(mesh, stiffness=9 * E, direction=[1, 0, 0])
  view(mesh.tetrahedra().find("fiber_direction"))[:] = dirs   # per tet, rest space
  ```
  Per-tet attributes: `fiber_direction` (Vector3, normalized at init) and `fiber_stiffness` (Pa). A zero direction
  or a zero stiffness means no fibre in that tet.
- **UID:** the user-defined range (>= 2^32, e.g. 2^32 + 1) in the fork; an official one (the next free is 33) is
  requested in an upstream pull request, so nothing collides if upstream takes 33 first.

## 4. Files

- Front end: `include/uipc/constitution/anisotropic_fiber.h`, `src/constitution/anisotropic_fiber.cpp` (UID
  registration, `apply_to`). Pattern: `kirchhoff_rod_bending`.
- CUDA: `src/backends/cuda/finite_element/constitutions/anisotropic_fiber.cu`, and
  `anisotropic_fiber_function.h` with the energy, gradient and Hessian as `__host__ __device__` functions. Host
  side as in `kirchhoff_rod_bending.cu`, kernels as in `stable_neo_hookean_3d.cu`.
- Python binding: `src/pybind/pyuipc/constitution/anisotropic_fiber.cpp`, registered in `module.cpp`.
- Build: CMake's `GLOB_RECURSE` (`src/backends/cuda/components.cmake`) picks up the new `.cu`; check that XMake
  does too (`agent_docs/rule.md`, rule 7).
- `python/examples/fem_cantilever_anisotropy.py`: add the fibre runs.
- For an upstream pull request, later: a spec page in `docs/specification/constitutions/`, the UID table regenerated
  (`scripts/gen_uid_doc.py`), a Catch2 unit test (like `apps/tests/backends/cuda/fem_projection.cu`), a Python
  regression test in `python/tests/sim_case/`, and the matching `agent_docs` updates.

## 5. Steps, each with a gate

1. **NumPy reference of the math** (about half a day).
   - Gradient vs finite differences of the energy; unprojected Hessian vs finite differences of the gradient.
   - The projected Hessian is positive semi-definite and has the paper's eigenvalues.
   - The `det F > 0 => s = +1` shortcut agrees with a full polar decomposition on random `F`, inverted ones
     included.
2. **Front end, Python binding, UID.** Gate: Python creates and edits the attributes, and a scene initializes with
   the extra energy listed. Measure the incremental build time here.
3. **CUDA kernels:** collect the fibred tets at init (indices with the vertex offset, weights from the rest
   positions, volume, stiffness), then the extent, energy and gradient/Hessian kernels.
   - Gate: fibre stiffness 0 gives the same result as SNH alone.
   - Gate: a single tet settles where the NumPy reference says.
4. **Cantilever, fibres along the arm at 9 x the effective E.**
   - Gate: slender arm about 1.9 mm, stubby arm about 0.6 mm (beam theory 2.07 / 0.64 mm times the 0.93 that
     libuipc measured against beam theory on 0.0.29).
   - Fibres vertical sag about like plain SNH.
   - Compare Newton iterations and step time with SNH alone.
5. **Regression:** libuipc's headless Python tests (244 on 0.0.29) still pass; the mmd_tools flesh spin without
   fibres is unchanged.
6. **Later, in mmd_tools:** fibre directions for the flesh (e.g. along the nearest bone's axis) and a config
   option.

Steps 1-4: about 3-4 days of work, plus build time.

## 6. Open decisions

- **Softening:** with each energy projected on its own, softening acts only through the gradient, so Newton
  converges more slowly. Projecting it properly needs the isotropic and fibre parts in one kernel, i.e. a new
  primary material. Decide after step 4, if softening is wanted at all.
- **Muscle contraction (eq. 48):** `Psi = mu_f (sqrt(I5) - l s)^2` with a per-tet rest length `l` (default 1) is
  almost free to add now.
- **The name, and whether this goes upstream.**

## 7. Risks

- Stiff fibres slow the linear solves: in the paper, fibres at 100 x the isotropic stiffness made a run 33 % slower,
  from more PCG iterations. libuipc's diagonal and MAS preconditioners may react differently.
- The sign flip at `I4 = 0` is a discontinuity; it only occurs in inverted tets, so it belongs to the deferred
  inversion work. Step 1 found a second one there (section 9): which axis counts as inverted is up to the SVD.
- 0.0.29's StableNeoHookean comes out 1.25 x too stiff at nu 0.3 (it stores `mu = 4/3 mu_L`,
  `lambda = lambda_L + 5/6 mu_L`; fixed upstream after 0.0.29 in `1aa9e6b`). Set fibre stiffness against the
  effective E, as `fem_cantilever_anisotropy.py` does.

## 8. Baseline (v0.0.29, `python/examples/fem_cantilever_anisotropy.py`, 2.5 mm grid)

| Arm | Run | Tip sag | Beam theory | libuipc / theory |
|---|---|---|---|---|
| slender 20 x 2 x 2 cm, E 2.5 MPa | SNH E | 17.85 mm | 19.09 mm | 0.935 |
| | SNH 10 E | 1.80 mm | 1.91 mm | 0.941 |
| stubby 4 x 2 x 2 cm, E 40 kPa | SNH E | 2.18 mm | 2.34 mm | 0.930 |
| | SNH 10 E | 0.218 mm | 0.234 mm | 0.930 |

Targets for fibres along the arm at 9 x E_eff (beam theory): slender 2.07 mm (1.09 x the 10 E sag), stubby 0.64 mm
(2.74 x the 10 E sag).

## 9. Step 1 result (2026-11-04): the math checks out

`fork_notes/anisotropic_fiber_reference.py` (NumPy, about 5 s): 3,000 random tets, a third each stretched in
general, compressed along the fibre and inverted (162 of the inverted ones with `s = -1`; 1,599 samples had the
negative eigenvalue that the projection clamps). Every check passes:

| Check | Worst error |
|---|---|
| `d = sum_k w_k x_k` equals `F a` | 2.9e-15 |
| Gradient, spring form vs the paper's PK1 (eq. 35) through `dF/dx` | 5.5e-13 |
| Hessian, spring form vs eq. 36 through `dF/dx` | 3.9e-15 |
| Projected Hessian vs the paper's 9x9 Hessian eigen-clamped | 7.9e-15 |
| Eigenvalues vs eqs. 37-38 | 6.7e-15 |
| Projected Hessian positive semi-definite (-min eigenvalue / max entry) | 1.7e-15 |
| Gradient vs finite differences of the energy | 1.2e-9 |
| Hessian vs finite differences of the gradient | 8.9e-10 |
| `det F > 0` but `I4 <= 0` (the sign shortcut) | 0 cases |
| Kernel sign vs `a^T S a` from the polar decomposition | 0 mismatches |

So the kernel can use the spring form: 4 weights per tet, a 3x3 matrix `M`, and an SVD only when `det F <= 0`.

**Found: in inverted tets, the sign depends on which axis the SVD calls inverted.** With libuipc's convention
(the smallest singular value carries the sign), a fibre that is reflected and longer than the other two axes reads
as not inverted. For `F = diag(x, 1, 1)`, fibre along x:

| x | 2 | 1 | 0.5 | 0 | -0.5 | -0.99 | -1.01 | -2 |
|---|---|---|---|---|---|---|---|---|
| s | +1 | +1 | +1 | +1 | -1 | -1 | +1 | +1 |
| Psi / mu | 0.5 | 0 | 0.125 | 0.5 | 1.125 | 1.98 | 0 | 0.5 |
| (x - 1)^2 / 2 (the smooth 1D curve, paper Fig. 2e) | 0.5 | 0 | 0.125 | 0.5 | 1.125 | 1.98 | 2.02 | 4.5 |

The energy jumps at `x = -1`, and a pure mirror along the fibre (`x = -1`) is a tie the SVD can resolve either way,
with zero fibre energy on one side. This is a property of the S-based invariant, not of the code (Smith et al.'s
I-invariants share the convention). It matters only for inverted tets: for the deferred inversion test, don't start
from an exact mirror along the fibre (mirror and also shorten the fibre axis, so it is the smallest stretch).

## 10. Step 2 result (2026-11-04): front end, Python binding, UID

- Files: `include/uipc/constitution/anisotropic_fiber.h`, `src/constitution/anisotropic_fiber.cpp`,
  `src/pybind/pyuipc/constitution/anisotropic_fiber.{h,cpp}`, one registration line in `module.cpp`. CMake and
  XMake glob these folders; no build file changed. Formatted with clang-format 18.1.8
  (`uv run --no-project --with clang-format==18.1.8 clang-format -i FILES`; the pinned copy under `output/` is not
  installed here, and `scripts/format_changed.py` only sees committed changes).
- UID `2^32 + 1` (4294967297), registered as `AnisotropicFiber`, type `FiniteElement`, user-defined.
- `apply_to(sc, stiffness, direction)` writes `fiber_direction` (normalized) and `fiber_stiffness` on every tet and
  adds the UID to `extra_constitution_uids`; it rejects a negative stiffness and a non-3D mesh. Per-tet edits through
  `view()` work. In Python, `view()` of the `VectorXu64` meta attribute returns `None`; read it with
  `meta().to_json()`.
- Incremental wheel build: 55 s (free RAM stayed above 10.5 GB).
- **libuipc silently ignores an extra constitution without a backend:** with only the front end, a CUDA world
  initializes and steps with the fibre UID in its constitution table. So step 3 must show that the fibre changes the
  result, not just that the scene runs.
- Tests: `python/tests/test_anisotropic_fiber.py`, 8 tests (UID, attributes, re-apply and zero fibre, per-tet edit,
  rejected inputs, scene table on the `none` and `cuda` backends; the CUDA one needs `-m cuda`, which the repo's
  default `addopts` deselects). The headless suite on this build: 252 passed (the 244 of 0.0.29 plus these 8),
  18.9 s, peak 4.1 GB.
