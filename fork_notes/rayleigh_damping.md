# Plan: stiffness-proportional (Rayleigh beta) damping for FEM tets

Fork-only note (branch `aniso` for now; a `damping` branch from it once code lands). Drafted 2026-11-04.

**Why.** libuipc 0.0.29 has no damping term; the only damping is BDF1's numerical damping, small at the flesh's
480 Hz. In Ronova's spin (`mmd_tools/sim/ronova_flesh_spin.json`) the breasts swing up to 62 mm away from the
skinned body and rebound 7 times, even at 5 kPa (`mmd_tools/TODO.md`, "Damping for the flesh simulation").
Production flesh (Houdini FEM and Vellum, Maya muscle) damps with a viscous term. Of the options weighed (mass
damping relative to the skeleton, stiffness-proportional Rayleigh, nonlinear viscoelastic potentials, ligament
springs), stiffness-proportional damping was chosen first: it resists the rate of deformation, so rigid motion (the
walk, the whole spin) is free by construction and no per-vertex reference motion is needed.

## 1. Scope

**In:** damping force `-beta K v` per tetrahedron, with `K` the PSD-projected StableNeoHookean stiffness at the
start of the step; per-tet `beta` (seconds); BDF1; meshes whose primary material is StableNeoHookean.

**Out for now:** mass-proportional damping (`alpha M`, which would have to be relative to the skeleton: pipeline
side, `TODO.md` step 1); damping the AnisotropicFiber stiffness (open decision); BDF2; ARAP or other primaries.

## 2. The math

**Continuous:** `M a = f_int(x) + f_ext - C v`, with `C = beta K`. For a vibration mode of angular frequency `w`
this gives the damping ratio `zeta(w) = beta w / 2`: fast motion is damped more than slow motion.

**BDF1 incremental potential.** libuipc minimizes per step
`E(x) = 1/2 (x - x~)^T M (x - x~) + dt^2 Psi(x)` (elastic energies are multiplied by `dt^2`, see
`stable_neo_hookean_3d.cu`). Implicit Euler with damping,
`M (x - x~) / dt^2 = f_int(x) - C (x - x_n) / dt`, is the minimum of `E` plus

  `E_d(x) = beta dt / 2 * (x - x_n)^T K (x - x_n)`

where `x_n` are the positions at the start of the step (`x_prevs` in `FiniteElementMethod`, set to `xs` by the BDF1
predictor before Newton starts). Check: `dE_d/dx = beta dt K (x - x_n)`, and dividing the stationarity condition by
`dt^2` gives the damping force `beta K (x - x_n) / dt = beta K v`.

**Per tetrahedron**, with `K_tet = V dFdx^T H(F_n) dFdx` and `H(F_n)` the PSD-projected 9x9 SNH Hessian at the
step-start deformation gradient `F_n` (libuipc's `snk1::ddEddF_spd`, the same projection the elastic term uses):

  - energy `E_d = beta dt V / 2 * dF^T H(F_n) dF`, with `dF = vec(F - F_n)` (`F` is linear in `x`, so
    `vec(F) - vec(F_n) = dFdx (x - x_n)`)
  - gradient `beta dt V * dFdx^T H(F_n) dF`
  - Hessian `beta dt V * dFdx^T H(F_n) dFdx`: constant within the step and positive semi-definite, so no new
    projection is needed. Assembly as in `stable_neo_hookean_3d.cu` (`project_F_gradient`, `project_F_hessian_block`).

**Size against the elastic term:** the damping Hessian is `beta / dt` times the elastic one. At 480 Hz and
`beta = 0.02 s` that is about 10x: for the linear solver it is like stiffening the material isotropically about 10x.

**Why `K` at the start of the step and not at rest:** `K(x_bar)` would damp rigid rotations of the whole body (its
null space holds the rest pose's rigid motions, not the rotated pose's), which matters in a spin. `K(x_n)` follows
the body. It is still not exactly rotation-free when the element is stressed: for a rotation `W` of a stressed
element, `(W F)^T H (W F) = -P : W^2 F`, not 0, so a spinning, stretched body loses a little angular velocity. The
gates below measure it; if it matters, use the co-rotated rest stiffness `R K(x_bar) R^T` instead (Mueller-style).
(Correction after step 4: the co-rotated form is rotation-free only for undeformed elements; a damping of the Green
strain rate is exactly rotation-free. Section 12.)

## 3. Design in libuipc

- **A FEM extra constitution**, like AnisotropicFiber (`fork_notes/anisotropic_fiber.md`): it adds `E_d` on the
  tets of meshes that carry its UID, and leaves SNH untouched.
- **Per-tet attribute** `damping_beta` (Float, s, >= 0; 0 = no damping in that tet). It reads the tets' SNH `mu`
  and `lambda` attributes (so it requires StableNeoHookean as the primary; asserted at init).
- **Step-start positions** from `FiniteElementMethod::x_prevs()` (public accessor), through a pointer taken with
  `require<FiniteElementMethod>()` in `do_build`, so the extra-constitution base class stays unchanged. Per step,
  `F_n` and `H(F_n)` can be recomputed in every Newton iteration (cheap, no state) or cached once per step (an
  optimization, later).
- **Python API** (working name `RayleighDamping`):
  ```python
  StableNeoHookean().apply_to(mesh, moduli, mass_density=1000)
  RayleighDamping().apply_to(mesh, beta=0.02)              # seconds
  view(mesh.tetrahedra().find("damping_beta"))[:] = per_tet
  ```
- **UID:** user-defined range, `2^32 + 2`.

## 4. Files (patterns from the fibre work)

- Front end: `include/uipc/constitution/rayleigh_damping.h`, `src/constitution/rayleigh_damping.cpp`.
- Python binding: `src/pybind/pyuipc/constitution/rayleigh_damping.{h,cpp}`, registered in `module.cpp`.
- CUDA: `src/backends/cuda/finite_element/constitutions/rayleigh_damping.cu`, reusing
  `stable_neo_hookean_3d_function.h` (`snk1::ddEddF_spd`) and `fem_utils.h`.
  After adding the `.cu`: delete `build/cp312/src/backends/cuda/cuda.dir/Release/uipc_backend_cuda.device-link.obj`
  (else the link fails with `__cudaRegisterLinkedBinary_..._rayleigh_damping_cu`).
- Tests: `python/tests/test_rayleigh_damping.py`. Examples: `python/examples/fem_damping_demos.py`.
- NumPy reference: `fork_notes/rayleigh_damping_reference.py`.

## 5. Steps, each with a gate

1. **NumPy reference.** Per tet: `E_d`, gradient, Hessian with the SNK1 projection (port of `ddEddF_spd`, checked
   against a 9x9 eigen-clamp). Gates: gradient and Hessian vs finite differences; Hessian PSD; zero for a rigid
   translation in the step; `O(theta^4)` for a small rigid rotation of an unstressed tet; the size of the
   rotation leak for a stressed one (`-P : W^2 F`). Also a NumPy implicit-Euler integrator of one tet (SNH plus
   damping, Newton per step), the oracle for step 3.
2. **Front end, Python binding, UID**, with front-end tests (as `test_anisotropic_fiber.py`).
3. **CUDA kernels.** Gates: `beta = 0` bitwise equal to no damping; one tet released from a displaced pose follows
   the NumPy implicit-Euler trajectory over 100 steps (relative error about 1e-10).
4. **Test scenes** (`fem_damping_demos.py`), with BDF1's own numerical damping measured by a `beta = 0` run and
   removed:
   - **Plucked cantilever** (the existing arm, tip released): logarithmic decrement of the first two bending modes;
     `zeta` must scale with frequency, `zeta_2 / zeta_1 = w_2 / w_1` (about 6.3 for a clamped beam), and match
     `beta w / 2`.
   - **Spinning free cube** (no gravity, initial angular velocity): angular velocity kept (loss vs `beta = 0`
     within about 1 % over a few turns) while a superposed wobble dies out.
   - A render like the fibre demos: the same arm plucked with and without damping.
5. **Regression:** the headless Python suite; Ronova's spin flesh without damping unchanged (compare against
   more than one baseline run: one point near the right hip differs by up to 0.5 mm between runs of one build).
6. **In mmd_tools:** a `damping` config option in `sim/flesh_walk.py` (`beta`, and per region through
   `flesh_regions.py`, like the material regions). Ronova's spin with damping on the breasts only.
   - **Metric** (to write: the earlier numbers were measured by hand): for the breast-region surface vertices, the
     distance from their plainly skinned positions per frame (the comparison `sim/vis_flesh.py` makes); the worst
     value over the run (62 mm now) and the rebounds (local maxima above 10 mm: 7 now).
   - **Also:** 0 inverted tets, min volume ratio, Newton iterations and linear-solve time (with the GPU idle), and
     the walk (`sim/ronova_flesh_*.json` walk config) to check that the flesh does not look sluggish.

## 6. Choosing beta

Measure the swing frequency `f` from the current spin (the breast distance signal above), aim for `zeta` about
0.2-0.3 at `f`, and set `beta = 2 zeta / (2 pi f)`. For example `f = 3 Hz`, `zeta = 0.25` gives `beta = 0.027 s`,
so `beta / dt = 13` at 480 Hz. Faster ripples are then strongly damped (at 30 Hz, `zeta = 2.5`), which suits flesh.

## 7. Open decisions

- **Damp the fibre stiffness too?** `K` here is SNH only. With AnisotropicFiber on the same tets, true
  stiffness-proportional damping would add the fibre Hessian. Not needed for the breasts.
- **`K(x_n)` or co-rotated rest stiffness:** decided after the spinning-cube gate (section 12): neither; the Green
  strain rate (section 13).
- **Per-tet or per-vertex `beta`:** per tet first (regions are per tet in `flesh_regions.py`).
- **Mass-proportional damping relative to the skeleton** (`TODO.md` step 1): only if the slow swing is still too
  large.

## 8. Risks

- **Linear-solve cost.** `beta / dt` about 10 makes the system behave like a material about 10x stiffer: the paper
  measured +40 % at 10x isotropic stiffening, and this machine showed 2-3x slower solves for badly conditioned
  fibre cases. The MAS preconditioner (`linear_system/fem_preconditioner = "mas"`) may help. Measure with the GPU
  idle.
- **Rotation leak under stress** (section 2): measured by the spinning cube.
- **Overdamping** makes flesh look sluggish: check the walk, not just the spin.
- **dt dependence:** `beta` is physical (seconds), so results should not depend on dt beyond BDF1's own damping;
  check by running the spin at 480 and 960 Hz once.

## 9. Step 1 result (2026-11-04): the math checks out

`fork_notes/rayleigh_damping_reference.py` (NumPy, about 5 s): 600 random deformation gradients (stretched,
compressed, inverted) and 600 random tets. Every check passes:

| Check | Worst error |
|---|---|
| SNK1 exact Hessian vs finite differences of libuipc's PK1 (`snk1::dEdF`) | 2.0e-10 |
| Port of libuipc's analytic eigensystem (`snk1::ddEddF_spd`): `Q^T Q = I` | 2.7e-15 |
| Port: `Q diag(ev) Q^T` vs the exact Hessian | 9.9e-15 |
| Port, clamped, vs a 9x9 eigen-clamp of the exact Hessian | 9.8e-15 |
| Damping gradient / Hessian vs finite differences | 3.7e-10 / 3.6e-10 |
| Damping Hessian PSD | 1.3e-15 |
| Rigid translation in the step | 9e-27 (zero) |
| Unstressed tet rotated by theta in the step: `E_d / theta^4` constant | 1.3e-7 |
| `(W F)^T H (W F) = -P : W^2 F` | 0 |
| Stressed tet rotated: `E_d / theta^2` matches `(beta dt V / 2)(W F)^T H_spd (W F)` | 4.4e-8 |
| One-tet implicit-Euler oracle: decay per step vs the discrete BDF1 recurrence (beta 0 / 0.004) | 2e-4 / 6e-4 |
| Added damping ratio vs `beta w / 2` | 2.8 % |

- libuipc's closed-form SNK1 projection is exact: it equals the 9x9 eigen-clamp of the exact Hessian to round-off.
  579 of the 600 random F had a negative eigenvalue, so the clamp is active almost everywhere off the rest state.
- Rotation in the step: an unstressed tet costs `O(theta^4)` (here `3.5e-8 theta^4`), a stressed one leaks at
  `O(theta^2)` (`9.5e-9 theta^2` for `F = diag(1.3, 0.95, 0.9)`), exactly as section 2 predicts; whether that is
  small enough is the spinning-cube gate.
- **BDF1's own damping is a stiffness-proportional damping with beta = dt**: the oracle's mode (9.2 Hz at
  dt = 1/480) shows `zeta = 0.060 = w dt / 2`. So at 480 Hz the flesh already has the equivalent of
  `beta = 0.002 s`; the target `beta` of about 0.02-0.03 s (section 6) is about 10x that, independent of dt. With
  `beta = 0.004`, the added `zeta` (0.112) is within 3 % of `beta w / 2` (0.116); the rest is BDF1's
  discretization of the damping term, which the discrete recurrence matches to 6e-4.

## 10. Step 2 result (2026-11-04): front end, Python binding, UID

- Files: `include/uipc/constitution/rayleigh_damping.h`, `src/constitution/rayleigh_damping.cpp`,
  `src/pybind/pyuipc/constitution/rayleigh_damping.{h,cpp}`, one registration line in `module.cpp`. Formatted with
  clang-format 18.1.8. Incremental wheel build: 47 s.
- UID `2^32 + 2` (4294967298), registered as `RayleighDamping`, type `FiniteElement`, user-defined.
- `apply_to(sc, beta)` writes `damping_beta` on every tet and adds the UID to `extra_constitution_uids`. It rejects
  a negative beta, a non-3D mesh, and a mesh whose primary material is not StableNeoHookean (UID 10: none, or ARAP).
- Tests: `python/tests/test_rayleigh_damping.py`, 9 front-end tests (the CUDA one needs `-m cuda`). Headless Python
  suite: 266 passed (257 + 9), 20.6 s, peak 4.2 GB. As with the fibre, no backend yet: a CUDA scene with the UID
  runs and ignores it.

## 11. Step 3 result (2026-11-04): CUDA kernels

- File: `src/backends/cuda/finite_element/constitutions/rayleigh_damping.cu`. It reuses `snk1::ddEddF_spd` at the
  step-start `F_n` (from `FiniteElementMethod::x_prevs()`, through `require<FiniteElementMethod>()` in
  `do_build`), and `fem::tetrahedron_shape_gradients` / `project_F_gradient` / `project_F_hessian_block` as the SNH
  kernel does. The stale device-link object was deleted before the build (CLAUDE.md, fork build), so it linked the
  first time (42 s).
- Tests (`python/tests/test_rayleigh_damping.py`, 13 with `-m "basic or cuda"`): one tet, vertices 0, 2, 3 fixed,
  vertex 1 (50 g) released from a displaced pose, no gravity, 480 Hz, 100 steps, against the NumPy implicit-Euler
  oracle (`rayleigh_damping_reference.py`):

  | Case | Max difference / amplitude | Amplitude | Last 20 steps | Steps with the clamp active |
  |---|---|---|---|---|
  | small, beta 0.01 | 7.6e-9 | 0.1 mm | 0.0003 mm | 86 |
  | large, beta 0.01 | 8.5e-10 | 40 mm | 0.17 mm | 95 |
  | large, beta 0 | 2.1e-9 | 40 mm | 11.8 mm | 77 |

  `beta = 0` is bitwise equal to no RayleighDamping at all. The clamp is active even in the small case: at rest the
  SNK1 twist eigenvalues `mu + sigma (lambda (J - 1) - mu)` are 0 (rotations cost nothing), and tiny perturbations
  make them slightly negative.
- Headless Python suite on this build: 270 passed (266 + 4), 30 s, peak 4.2 GB.
- **libuipc quirk found:** a geometry created with an explicit rest geometry (`geometries().create(mesh, rest)`)
  whose rest mesh lacks the material (a bare `tetmesh`) crashes libuipc natively at `world.init`, without an error
  message. Giving the rest mesh the same constitutions fixes it. Candidate for an upstream issue (a clear error).

## 12. Step 4 result (2026-11-05): test scenes; the spin gate fails

Scenes: `python/examples/fem_damping_demos.py` (headless: `pluck`, `spin`, `show`) and
`fem_damping_demos_render.py` (Blender: stills, MP4 videos, .blend files). The outputs stay out of the repository.

**Pluck (passes).** The slender arm of `fem_cantilever_anisotropy.py` (20 x 2 x 2 cm, E 2.5 MPa, 33,792 tets), no
gravity, dt 1/1920. Its modes come from libuipc's own linearization, rebuilt in NumPy: the SNK1 stiffness at rest
(linear elasticity with Lame `(mu, lambda - mu)`; it matches the reference's exact Hessian at rest to 2e-16 per tet)
and the lumped vertex masses (density x the rest mesh's vertex `volume`). The Kuhn tets make the arm 1.7 % stiffer
along one diagonal of its cross-section, so it bends along the diagonals: the gate plucks the first two modes of one
diagonal (4.573 and 27.465 Hz, ratio 6.006; Euler-Bernoulli's 6.267 less shear), tip 1 mm + 0.3 mm. Each mode's
coordinate `u_k^T M (x - X)` is fitted by the exact BDF1 two-step recurrence and compared with the BDF1 prediction
for that mode (`(q+ - 2q + q-)/dt^2 = -w^2 q+ - beta w^2 (q+ - q)/dt`):

| beta (s) | Mode | zeta measured | BDF1 prediction | Error | Added zeta | beta w / 2 |
|---|---|---|---|---|---|---|
| 0 | 1 | 0.00748 | 0.00748 | 1.8e-4 | | |
| 0 | 2 | 0.04490 | 0.04483 | 1.6e-3 | | |
| 0.002 | 1 | 0.03653 | 0.03621 | 0.9 % | 0.0290 | 0.0287 |
| 0.002 | 2 | 0.21654 | 0.21535 | 0.55 % | 0.1716 | 0.1726 |
| 0.005 | 1 | 0.07995 | 0.07927 | 0.86 % | 0.0725 | 0.0718 |
| 0.005 | 2 | 0.47144 | 0.46645 | 1.1 % | 0.4265 | 0.4314 |

- Without damping, libuipc's per-mode decay is BDF1's own to 2e-4 and 1.6e-3. With damping, the added damping
  ratio grows with frequency as designed: mode 2 / mode 1 = 5.91 and 5.89, against `w2 / w1` = 6.01 (BDF1's
  discretization of the damping term predicts 5.94).
- **The ~1 % excess grows with amplitude** (beta 0.005, error against the linear prediction, modes 1 / 2):
  1/4 x the amplitude 0.20 % / 0.17 %, 1 x 0.86 % / 1.1 %, 4 x 3.7 % / 10 % (at 4 x, beta 0 itself is off by
  0.4 % / 2.6 %: geometric nonlinearity). So it is not an implementation error but the damping metric's dependence
  on the state: bending rotates stressed elements, `-P : W^2 F` is positive in tension and negative in compression,
  and the PSD projection keeps only the positive part, so the damping metric exceeds `K(rest)` by O(strain).
- Newton per step 3.2 (beta 0) to 3.8; wall 40 s for beta 0 and 120-136 s with damping (GPU shared with a game:
  only roughly comparable). `newton/velocity_tol` 1e-8 m/s is at round-off here: 292 harmless "Line Search Exits
  with Max Iteration" warnings with `rel_E_increase` about 1e-16.

**Spin (fails).** A free 10 cm cube (3,072 tets, E 10 kPa, nu 0.4), no gravity, 1.5 turns/s about z, released
stretched 1.1 x along x, dt 1/480, 2 s:

| beta (s) | L / L0 after 2 s | vs beta 0 | Turns | Spin at the end | Wobble, first 0.2 s / last 0.25 s (peak to peak) |
|---|---|---|---|---|---|
| 0 | 0.742 | | 2.69 | 1.22 turns/s | 3.2 mm / 0.01 mm |
| 0.02 | 0.238 | -68 % | 1.20 | 0.41 turns/s | 3.4 mm / 0.00 mm |
| 0.05 | 0.115 | -84 % | 0.59 | 0.20 turns/s | 3.4 mm / 0.00 mm |

- BDF1 alone loses angular momentum at `(W dt)^2` per step (for a particle on a circle, `L+ = L + dt m v+ x v`):
  0.69 predicted at a constant spin, 0.74 measured as the spin slows. Ronova's spin is driven by the bones, so this
  is a lag of free flesh, already in today's results.
- **The damping drags on the rotation**: the gate wanted the loss within about 1 % of beta 0; it is 68 % at the flesh's
  beta. This is the rotation leak of section 2, at full size: centrifugal tension makes `-P : W^2 F > 0`, and the
  projection adds the compressed elements' part. The wobble dies within 0.2 s as intended. In the flesh the same leak
  would resist every rigid rotation of stressed flesh (sagging breasts turning with the body, thighs swinging), so
  it would add lag behind the bones instead of only calming the swing. Not usable as is.

**Show.** Three arms under gravity at 480 Hz, tip lifted 40 mm and released (`show.mp4`). Tip swing peak to peak
over 0-0.5 s, 1-1.5 s and 2-2.5 s: beta 0: 106, 39, 15 mm (BDF1's own damping); beta 0.005: 90, 2, 0 mm;
beta 0.02: 64, 0, 0 mm. The damped arms settle at a sag of 17.8 mm.

**Decision needed: an objective damping metric.** The co-rotated rest stiffness (section 2's fallback) is
rotation-free only for undeformed elements: a rotation `W` of `F_n = R S` gives `sym(R^T W R S) = [W', S] / 2`,
which is O(stretch) per unit rotation, so its leak is O(stretch^2) instead of O(stress). Exactly rotation-free at any
deformation is a damping of the Green strain rate (Kelvin-Voigt in the material frame):

  `E_d = beta dt V / 2 * dE : C : dE`, with `dE = E(F) - E(F_n)`, `E(F) = (F^T F - I) / 2`

and `C` the small-strain elasticity tensor of the tet's SNK1 (Lame `mu`, `lambda - mu`). A rotation within the step
leaves `F^T F` and so `dE` unchanged: no damping, at any rotation and any stress. At rest it equals `beta K(rest)`, so
`zeta = beta w / 2` stays. Gradient `beta dt V * F (C : dE)` (as PK1 = F S); Hessian `beta dt V (J^T C J + I (x) S_d)`
with `J = d dE / dF`; the second term is O(dE), small within a step, and is dropped (Gauss-Newton), so the Hessian is
PSD without an eigen-decomposition (cheaper than the current `ddEddF_spd`). Steps to switch: NumPy reference and
oracle, the kernel's energy / gradient / Hessian, the tests, then the pluck (the excess should drop to O(strain^2))
and the spin gates again.

## 13. The switch to the Green strain rate (2026-11-05): every gate passes

**Kernel** (`rayleigh_damping.cu`): `E_d = beta dt V / 2 * dE : C : dE`, `dE = (F^T F - F_n^T F_n) / 2`; gradient
`beta dt V * F (C : dE)`; Hessian `beta dt V * M`, `M(3b + i, 3d + k) = mu_L ((F F^T)_ik delta_bd + F_id F_kb) +
lam_L F_ib F_kd` (closed form, no eigen-decomposition; the geometric term `I (x) (C : dE)` is left out, Gauss-Newton).
Per tet the small-strain Lame parameters of SNK1: `mu_L = mu`, `lam_L = max(lambda - mu, -2/3 mu)` (so C is PSD).
The front end, UID, attribute and Python API are unchanged; only their descriptions changed. Editing the `.cu` hit
the device-link error too: delete the device-link object after any `.cu` change (CLAUDE.md updated).

**NumPy reference** (`rayleigh_damping_reference.py`): both forms, `Tet(..., form="green" | "stiffness")`, the
stiffness form kept as the record of steps 1-4. All checks pass; the new ones:

| Check (green) | Worst |
|---|---|
| Gradient / exact Hessian vs finite differences | 4.6e-10 / 4.3e-10 |
| The kernel's closed-form M vs `J^T C J` | 2.8e-16 |
| M PSD | 1.1e-15 |
| Finite rotation (0.2-1 rad) in the step of a stressed tet: `E_d / (beta dt V mu F^4)` | 7e-30 (zero) |
| Net torque of the damping force | 1.2e-16 |
| Hessian at rest vs the stiffness form's (`beta K(rest)`) | 1.8e-15 |
| Oracle: decay per step vs discrete BDF1 (beta 0 / 0.004), added zeta vs `beta w / 2` | 2e-4 / 6e-4, 2.8 % (as the stiffness form) |

**Oracle fix.** Its Armijo line search compared total energies, about 3e-6 J here (mostly SNK1's rest energy
`mu^2 / (2 lambda)`), with round-off about 3e-22 J. The green energy is quartic in x, so after the first Newton step
about 2e-11 m remain, worth a decrease of about 1e-23 J: below round-off, so the oracle stalled there (its gradient
1.3e-12 against 1e-20 at libuipc's point). Below round-off it now accepts a step that lowers the gradient norm. The
stiffness form is quadratic in x, so its first Newton step was exact and the stall never showed.

**Tests** (`test_rayleigh_damping.py`, 15 with `-m "basic or cuda"`): one tet against the oracle, max error /
amplitude: small 7.6e-10, large 1.5e-12, large without damping 2.3e-12, a free tet stretched 1.1 x and spinning at
2 turns/s 4.9e-12. The spinning tet keeps its angular momentum: beta 0.01 vs none +8.0e-4 at the end (the oracle
says +8.04e-4; the stiffness form lost 25 % there), while its ringing drops from 0.192 to 0.120 mm. Python suite with
the CUDA tests (`-m "not example"`): 271 passed.

**Gates** (the scenes of section 12, same settings):

| Pluck, zeta vs BDF1 prediction | Stiffness form | Green |
|---|---|---|
| beta 0.002, mode 1 / 2 | 0.9 % / 0.55 % | 5.6e-5 / 4.7e-4 |
| beta 0.005, mode 1 / 2 | 0.86 % / 1.1 % | 2.3e-5 / 2.0e-4 |
| beta 0.005 at 4 x the amplitude, mode 1 / 2 (beta 0 alone: 3.8e-3 / 2.6e-2) | 3.7 % / 10 % | 3.9e-4 / 3.2e-3 |

- Added zeta, mode 2 / mode 1: 5.94 (beta 0.002) and 5.87 (0.005), exactly BDF1's prediction (5.94, 5.87); `w2 / w1`
  is 6.01. Added zeta vs `beta w / 2`: mode 1 within 0.1 %, mode 2 2.3 % below (BDF1's discretization of the term).
- **Spin:** L / L0 after 2 s 0.7416 (beta 0), 0.7416 (0.02), 0.7420 (0.05): the damping no longer touches the spin
  (gate: within 1 %; the stiffness form lost 68 % / 84 %). Turns 2.688 / 2.686 / 2.685. The wobble: RMS peak to peak
  over 0.1-0.3 s 1.09 mm without damping, 0.02 mm at beta 0.02; beta 0.05 is overdamped for the cube's ~11 Hz
  modes (zeta about 1.7) and creeps back without ringing.
- **Show** (tip swing peak to peak 0-0.5, 1-1.5, 2-2.5 s): beta 0: 106.5, 38.9, 14.8 mm; beta 0.005: 96.9, 4.6,
  0.2 mm; beta 0.02: 77.7, 0, 0 mm (the stiffness form damped this 40 mm pluck a little more: its projection excess).
  Settled sag 17.8-17.9 mm.
- Wall (GPU shared with a game): pluck 39 s at beta 0, 99-136 s with damping; Newton per step 3.2-3.4.

Next: step 5 (Ronova's spin flesh without damping unchanged, against more than one baseline run), then step 6.
