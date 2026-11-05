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
gates below measure it; if it matters, use the co-rotated rest stiffness `R K(x_bar) R^T` instead (Mueller-style),
which is exactly rotation-free.

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
- **`K(x_n)` or co-rotated rest stiffness:** decide after the spinning-cube gate.
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
