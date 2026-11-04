"""Front-end checks for the AnisotropicFiber extra constitution (fork branch `aniso`, plan step 2).

The physics (the CUDA kernels) is checked separately; these cover the UID, the per-tetrahedron attributes, the
rejected inputs and that a scene lists the constitution.
"""

import os
import sys

import numpy as np
import pytest

from uipc import Engine, Scene, World, builtin, view
from uipc import Exception as UIPCException
from uipc.constitution import AnisotropicFiber, ElasticModuli, StableNeoHookean
from uipc.geometry import label_surface, label_triangle_orient, tetmesh, trimesh

from conftest import skip_cuda_on_macos, skip_cuda_on_macos_reason

UID = 2**32 + 1  # user-defined range while the constitution lives in a fork


def cube():
    """A 0.1 m cube of 5 positively oriented tets with StableNeoHookean."""
    V = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], float) * 0.1
    T = np.array([[0, 1, 3, 5], [0, 3, 2, 6], [0, 5, 4, 6], [3, 5, 6, 7], [0, 3, 5, 6]])
    e = V[T[:, 1:]] - V[T[:, :1]]
    neg = np.linalg.det(e) < 0
    T[neg] = T[neg][:, [0, 1, 3, 2]]
    mesh = tetmesh(V, T)
    label_surface(mesh)
    label_triangle_orient(mesh)
    StableNeoHookean().apply_to(mesh, ElasticModuli.youngs_poisson(1e5, 0.3), mass_density=1000.0)
    return mesh


def extra_uids(mesh):
    # view() of a VectorXu64 meta attribute is not readable from Python; go through the JSON form
    values = mesh.meta().to_json()[builtin.extra_constitution_uids]["attribute"]["values"]
    return np.array(values, dtype=np.uint64).reshape(-1).tolist()


@pytest.mark.basic
def test_uid_registered():
    reg = builtin.ConstitutionUIDCollection.instance()
    assert reg.exists(UID)
    info = reg.find(UID)
    assert info.name == "AnisotropicFiber"
    assert info.type == "FiniteElement"
    assert info.is_user_defined_uid(UID)


@pytest.mark.basic
def test_apply_to_sets_attributes():
    mesh = cube()
    AnisotropicFiber().apply_to(mesh, 2.0e5, np.array([2.0, 0.0, 0.0]))
    d = view(mesh.tetrahedra().find("fiber_direction")).reshape(-1, 3)
    k = view(mesh.tetrahedra().find("fiber_stiffness")).reshape(-1)
    np.testing.assert_allclose(d, np.tile([1.0, 0.0, 0.0], (mesh.tetrahedra().size(), 1)))
    np.testing.assert_allclose(k, np.full(mesh.tetrahedra().size(), 2.0e5))
    assert UID in extra_uids(mesh)


@pytest.mark.basic
def test_reapply_overwrites_and_zero_means_no_fiber():
    mesh = cube()
    fib = AnisotropicFiber()
    fib.apply_to(mesh, 2.0e5, np.array([0.0, 3.0, 4.0]))
    np.testing.assert_allclose(view(mesh.tetrahedra().find("fiber_direction")).reshape(-1, 3)[0], [0.0, 0.6, 0.8])
    fib.apply_to(mesh, 0.0, np.zeros(3))
    np.testing.assert_allclose(view(mesh.tetrahedra().find("fiber_direction")), 0.0)
    np.testing.assert_allclose(view(mesh.tetrahedra().find("fiber_stiffness")), 0.0)
    assert extra_uids(mesh).count(UID) == 1


@pytest.mark.basic
def test_per_tet_edit():
    mesh = cube()
    AnisotropicFiber().apply_to(mesh, 2.0e5, np.array([1.0, 0.0, 0.0]))
    d = view(mesh.tetrahedra().find("fiber_direction"))
    d[2] = np.array([0.0, 0.0, 1.0]).reshape(d[2].shape)
    k = view(mesh.tetrahedra().find("fiber_stiffness"))
    k[3] = 5.0
    d = view(mesh.tetrahedra().find("fiber_direction")).reshape(-1, 3)
    np.testing.assert_allclose(d[2], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(d[1], [1.0, 0.0, 0.0])
    assert view(mesh.tetrahedra().find("fiber_stiffness")).reshape(-1)[3] == 5.0


@pytest.mark.basic
def test_rejects_negative_stiffness():
    with pytest.raises(UIPCException):
        AnisotropicFiber().apply_to(cube(), -1.0, np.array([1.0, 0.0, 0.0]))


@pytest.mark.basic
def test_rejects_non_tet_mesh():
    tri = trimesh(np.eye(3), np.array([[0, 1, 2]]))
    with pytest.raises(UIPCException):
        AnisotropicFiber().apply_to(tri, 1.0, np.array([1.0, 0.0, 0.0]))


def run_one_step(backend, workspace):
    mesh = cube()
    AnisotropicFiber().apply_to(mesh, 2.0e5, np.array([1.0, 0.0, 0.0]))
    scene = Scene(Scene.default_config())
    scene.objects().create("block").geometries().create(mesh)
    os.makedirs(workspace, exist_ok=True)
    engine = Engine(backend, workspace)
    world = World(engine)  # World keeps only a weak reference to its engine
    world.init(scene)
    assert world.is_valid()
    assert UID in [int(u) for u in scene.constitution_tabular().uids()]
    world.advance()
    world.retrieve()


@pytest.mark.basic
def test_scene_lists_uid_none_backend(tmp_path):
    run_one_step("none", str(tmp_path / "none"))


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
def test_scene_lists_uid_cuda_backend(tmp_path):
    run_one_step("cuda", str(tmp_path / "cuda"))


# ---------------------------------------------------------------- physics: one tet against a NumPy static solve
# Rest tet (0, e_x, e_y, e_z) * 0.1 m; vertices 0, 2, 3 fixed; vertex 1 (mass 1 kg) pulled by "gravity". At
# equilibrium the elastic force on vertex 1 balances m g, which NumPy solves from the same formulas: libuipc's
# StableNeoHookean (SNK1: mu/2 (Ic - 3) + lambda/2 (J - 1 - mu/lambda)^2, its stored mu and lambda) plus the fiber
# term mu_f/2 (|F a| - 1)^2 (no inversion here, so sign(I4) = 1).
TET_REST = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) * 0.1
E_TET, NU_TET, MU_FIBER = 1.0e5, 0.3, 5.0e5


def tet_vertex1_force(x1, mu, lam, a, mu_f):
    """Elastic gradient on vertex 1 (N) of the test tet with vertex 1 at x1."""
    X = TET_REST.copy()
    X[1] = x1
    Dm_inv = np.linalg.inv((TET_REST[1:] - TET_REST[0]).T)
    V = np.linalg.det((TET_REST[1:] - TET_REST[0]).T) / 6
    F = (X[1:] - X[0]).T @ Dm_inv
    J = np.linalg.det(F)
    P = mu * F + lam * (J - 1 - mu / lam) * J * np.linalg.inv(F).T
    g = V * P @ Dm_inv[0]  # dF/dx_1 picks row 0 of Dm_inv
    if mu_f > 0:
        d = F @ a
        nd = np.linalg.norm(d)
        g = g + V * (Dm_inv @ a)[0] * mu_f * (1 - 1 / nd) * d
    return g


def tet_equilibrium(gravity, mu, lam, a, mu_f, mass=1.0):
    x = TET_REST[1].copy()
    for _ in range(50):  # Newton with a finite-difference Jacobian
        r = tet_vertex1_force(x, mu, lam, a, mu_f) - mass * gravity
        if np.linalg.norm(r) < 1e-12:
            break
        h = 1e-7
        Jac = np.stack([(tet_vertex1_force(x + h * e, mu, lam, a, mu_f) -
                         tet_vertex1_force(x - h * e, mu, lam, a, mu_f)) / (2 * h) for e in np.eye(3)], axis=1)
        x = x - np.linalg.solve(Jac, r)
    return x


def tet_libuipc(workspace, gravity, a, mu_f, fiber=True):
    """Settled position of vertex 1 in libuipc (BDF1 at a large time step damps it to the static state)."""
    mesh = tetmesh(TET_REST, np.array([[0, 1, 2, 3]]))
    label_surface(mesh)
    label_triangle_orient(mesh)
    StableNeoHookean().apply_to(mesh, ElasticModuli.youngs_poisson(E_TET, NU_TET), mass_density=1000.0)
    if fiber:
        AnisotropicFiber().apply_to(mesh, mu_f, a)
    view(mesh.vertices().find("mass") or mesh.vertices().create("mass", 0.0))[:] = 1.0
    is_fixed = view(mesh.vertices().find(builtin.is_fixed))
    is_fixed[:] = np.array([1, 0, 1, 1]).reshape(is_fixed.shape)
    mu = float(view(mesh.tetrahedra().find("mu"))[0])
    lam = float(view(mesh.tetrahedra().find("lambda"))[0])

    cfg = Scene.default_config()
    cfg["dt"] = 0.05
    cfg["gravity"] = [[gravity[0]], [gravity[1]], [gravity[2]]]
    cfg["contact"]["enable"] = 0
    cfg["newton"]["velocity_tol"] = 1e-9
    cfg["newton"]["semi_implicit"]["enable"] = 0
    cfg["linear_system"]["tol_rate"] = 1e-10
    scene = Scene(cfg)
    slot, _ = scene.objects().create("tet").geometries().create(mesh)
    os.makedirs(workspace, exist_ok=True)
    engine = Engine("cuda", workspace)
    world = World(engine)
    world.init(scene)
    assert world.is_valid()
    for _ in range(200):
        world.advance()
        world.retrieve()
    return slot.geometry().positions().view().reshape(-1, 3)[1].copy(), mu, lam


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
@pytest.mark.parametrize(
    "case, gravity, fiber",
    [
        ("snh only", [10.0, 0.0, 0.0], False),
        ("pull along the fiber", [10.0, 0.0, 0.0], True),
        ("pull across the fiber", [0.0, 10.0, 0.0], True),
        ("pull obliquely", [6.0, 6.0, -3.0], True),
    ],
)
def test_single_tet_matches_numpy(tmp_path, case, gravity, fiber):
    a = np.array([1.0, 0.0, 0.0])
    gravity = np.array(gravity)
    x_uipc, mu, lam = tet_libuipc(str(tmp_path / "ws"), gravity, a, MU_FIBER, fiber)
    x_ref = tet_equilibrium(gravity, mu, lam, a, MU_FIBER if fiber else 0.0)
    u_ref = x_ref - TET_REST[1]
    assert np.linalg.norm(u_ref) > 1e-4  # the load moves the vertex by about a millimetre or more
    assert np.linalg.norm(x_uipc - x_ref) < 1e-6 * np.linalg.norm(u_ref) + 1e-9, (case, x_uipc, x_ref)


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
def test_single_tet_fiber_changes_result_and_zero_stiffness_does_not(tmp_path):
    a, g = np.array([1.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0])
    x_snh, _, _ = tet_libuipc(str(tmp_path / "snh"), g, a, 0.0, fiber=False)
    x_zero, _, _ = tet_libuipc(str(tmp_path / "zero"), g, a, 0.0, fiber=True)
    x_fib, _, _ = tet_libuipc(str(tmp_path / "fib"), g, a, MU_FIBER, fiber=True)
    np.testing.assert_array_equal(x_zero, x_snh)  # no fibred tet: the extra constitution adds nothing
    u_snh, u_fib = x_snh[0] - TET_REST[1, 0], x_fib[0] - TET_REST[1, 0]
    assert u_fib < 0.5 * u_snh  # a fiber 5x the Young's modulus along the pull at least halves the stretch
