"""Checks for the RayleighDamping extra constitution (fork branch `aniso`, plan fork_notes/rayleigh_damping.md).

Front end: the UID, the per-tetrahedron attribute, the rejected inputs and that a scene lists the constitution.
Physics (CUDA, -m cuda): one tet against the NumPy oracle (plucked and spinning), beta 0 a no-op, the damping working,
and a spinning tet keeping its angular momentum (the damping of the Green strain rate is rotation-free).
"""

import os

import numpy as np
import pytest

from uipc import Engine, Scene, World, builtin, view
from uipc import Exception as UIPCException
from uipc.constitution import ARAP, ElasticModuli, RayleighDamping, StableNeoHookean
from uipc.geometry import label_surface, label_triangle_orient, tetmesh, trimesh

from conftest import skip_cuda_on_macos, skip_cuda_on_macos_reason

UID = 2**32 + 2  # user-defined range while the constitution lives in a fork


def cube(material="snh"):
    """A 0.1 m cube of 5 positively oriented tets, with StableNeoHookean, ARAP or no material."""
    V = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], float) * 0.1
    T = np.array([[0, 1, 3, 5], [0, 3, 2, 6], [0, 5, 4, 6], [3, 5, 6, 7], [0, 3, 5, 6]])
    e = V[T[:, 1:]] - V[T[:, :1]]
    neg = np.linalg.det(e) < 0
    T[neg] = T[neg][:, [0, 1, 3, 2]]
    mesh = tetmesh(V, T)
    label_surface(mesh)
    label_triangle_orient(mesh)
    if material == "snh":
        StableNeoHookean().apply_to(mesh, ElasticModuli.youngs_poisson(1e5, 0.3), mass_density=1000.0)
    elif material == "arap":
        ARAP().apply_to(mesh)
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
    assert info.name == "RayleighDamping"
    assert info.type == "FiniteElement"
    assert info.is_user_defined_uid(UID)


@pytest.mark.basic
def test_apply_to_sets_attribute():
    mesh = cube()
    RayleighDamping().apply_to(mesh, 0.02)
    b = view(mesh.tetrahedra().find("damping_beta")).reshape(-1)
    np.testing.assert_allclose(b, np.full(mesh.tetrahedra().size(), 0.02))
    assert UID in extra_uids(mesh)


@pytest.mark.basic
def test_reapply_overwrites_and_per_tet_edit():
    mesh = cube()
    d = RayleighDamping()
    d.apply_to(mesh, 0.02)
    d.apply_to(mesh, 0.0)
    np.testing.assert_allclose(view(mesh.tetrahedra().find("damping_beta")), 0.0)
    assert extra_uids(mesh).count(UID) == 1
    b = view(mesh.tetrahedra().find("damping_beta"))
    b[1] = 0.05
    assert view(mesh.tetrahedra().find("damping_beta")).reshape(-1)[1] == 0.05


@pytest.mark.basic
def test_rejects_negative_beta():
    with pytest.raises(UIPCException):
        RayleighDamping().apply_to(cube(), -0.01)


@pytest.mark.basic
def test_rejects_non_tet_mesh():
    with pytest.raises(UIPCException):
        RayleighDamping().apply_to(trimesh(np.eye(3), np.array([[0, 1, 2]])), 0.01)


@pytest.mark.basic
@pytest.mark.parametrize("material", ["none", "arap"])
def test_rejects_mesh_without_stable_neo_hookean(material):
    with pytest.raises(UIPCException):
        RayleighDamping().apply_to(cube(material), 0.01)


def run_one_step(backend, workspace):
    mesh = cube()
    RayleighDamping().apply_to(mesh, 0.02)
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


# ---------------------------------------------------------------- physics: one tet against the NumPy oracle
# Rest tet (0, e_x, e_y, e_z) * 0.1 m, no gravity, BDF1 at 480 Hz. The oracle (fork_notes/rayleigh_damping_reference.py,
# form "green") integrates the same incremental potential: SNK1 with libuipc's stored mu and lambda plus the damping
# of the Green strain rate. Plucked: vertices 0, 2, 3 fixed, vertex 1 (50 g) released from a displaced pose.
# Spinning: all four vertices free (50 g each), released stretched and spinning about z.
sys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "fork_notes")
TET_REST = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) * 0.1
MASS = np.array([1.0, 0.05, 1.0, 1.0])
FIXED = np.array([True, False, True, True])
DT, STEPS = 1 / 480, 100


def oracle():
    import sys

    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    import rayleigh_damping_reference as ref

    return ref


def spinning_start():
    """Stretched 1.1 x along x (volume kept) about the centroid, spinning at 2 turns/s about z: X0, V0."""
    c = TET_REST.mean(0)
    X0 = c + (TET_REST - c) * np.array([1.1, 1.1**-0.5, 1.1**-0.5])
    return X0, np.cross([0.0, 0.0, 4 * np.pi], X0 - c)


def angular_momentum(traj, V0, mass):
    """L_z about the centre of mass per step, v_n = (x_n - x_{n-1}) / dt as BDF1."""
    v = np.concatenate([V0[None], np.diff(traj, axis=0) / DT])
    c = np.einsum("v,nvi->ni", mass, traj) / mass.sum()
    return np.einsum("v,nv->n", mass, np.cross(traj - c[:, None], v)[:, :, 2])


def tet_trajectory(workspace, X0, beta, V0=None, mass=MASS, fixed=FIXED, velocity_tol=1e-9):
    """libuipc: the 4 vertex positions at each step (STEPS + 1, 4, 3); beta None = no RayleighDamping at all."""
    T = np.array([[0, 1, 2, 3]])
    mesh, rest = tetmesh(X0, T), tetmesh(TET_REST, T)
    for m in (mesh, rest):
        label_surface(m)
        label_triangle_orient(m)
    # the rest geometry needs the same material: a bare one crashes libuipc at init without a message
    for m in (mesh, rest):
        StableNeoHookean().apply_to(m, ElasticModuli.youngs_poisson(3e4, 0.3), mass_density=1000.0)
        if beta is not None:
            RayleighDamping().apply_to(m, beta)
    m_view = view(mesh.vertices().find("mass") or mesh.vertices().create("mass", 0.0))
    m_view[:] = mass.reshape(m_view.shape)
    is_fixed = view(mesh.vertices().find(builtin.is_fixed))
    is_fixed[:] = fixed.astype(int).reshape(is_fixed.shape)
    if V0 is not None:
        vel = view(mesh.vertices().find(builtin.velocity))
        vel[:] = V0.reshape(vel.shape)
    mu = float(view(mesh.tetrahedra().find("mu"))[0])
    lam = float(view(mesh.tetrahedra().find("lambda"))[0])

    cfg = Scene.default_config()
    cfg["dt"] = DT
    cfg["gravity"] = [[0.0], [0.0], [0.0]]
    cfg["contact"]["enable"] = 0
    cfg["newton"]["velocity_tol"] = velocity_tol
    cfg["newton"]["semi_implicit"]["enable"] = 0
    cfg["linear_system"]["tol_rate"] = 1e-10
    scene = Scene(cfg)
    slot, _ = scene.objects().create("tet").geometries().create(mesh, rest)
    os.makedirs(workspace, exist_ok=True)
    engine = Engine("cuda", workspace)
    world = World(engine)
    world.init(scene)
    assert world.is_valid()
    out = [X0.copy()]
    for _ in range(STEPS):
        world.advance()
        world.retrieve()
        out.append(slot.geometry().positions().view().reshape(-1, 3).copy())
    return np.array(out), mu, lam


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
@pytest.mark.parametrize(
    "case, offset, beta",
    [
        ("small", [1e-4, -5e-5, 3e-5], 0.01),
        ("large", [-0.04, 0.03, 0.025], 0.01),
        ("large, no damping", [-0.04, 0.03, 0.025], 0.0),
        ("spinning", None, 0.01),
    ],
)
def test_single_tet_trajectory_matches_oracle(tmp_path, case, offset, beta):
    ref = oracle()
    if case == "spinning":
        (X0, V0), mass, fixed = spinning_start(), np.full(4, 0.05), np.zeros(4, bool)
    else:
        X0, V0, mass, fixed = TET_REST.copy(), np.zeros((4, 3)), MASS, FIXED
        X0[1] += offset
    traj, mu, lam = tet_trajectory(str(tmp_path / "ws"), X0, beta, V0, mass, fixed)
    tet = ref.Tet(TET_REST, mu, lam, beta, "green")
    expect = ref.implicit_euler(tet, TET_REST, X0, V0, mass, fixed, DT, STEPS)
    scale = np.abs(expect - expect[:1]).max()
    assert np.abs(traj - expect).max() < 1e-7 * scale, (case, np.abs(traj - expect).max() / scale)


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
def test_single_tet_damping_works_and_zero_beta_is_a_no_op(tmp_path):
    X0 = TET_REST.copy()
    X0[1] += [-0.02, 0.015, 0.01]
    none, _, _ = tet_trajectory(str(tmp_path / "none"), X0, None)
    zero, _, _ = tet_trajectory(str(tmp_path / "zero"), X0, 0.0)
    damped, _, _ = tet_trajectory(str(tmp_path / "damped"), X0, 0.01)
    np.testing.assert_array_equal(zero, none)  # no damped tet: the extra constitution adds nothing
    amp = lambda t: np.abs(t[-20:, 1] - TET_REST[1]).max()  # noqa: E731
    assert amp(damped) < 0.5 * amp(none)  # damping removes most of the ringing by the last steps


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
def test_spinning_tet_keeps_its_angular_momentum(tmp_path):
    """The objective form never damps rigid rotation: a stretched, spinning free tet keeps its angular momentum as
    without damping (the stiffness-proportional form K(x_n) of steps 1-4 lost 25 % here in the oracle)."""
    (X0, V0), mass, fixed = spinning_start(), np.full(4, 0.05), np.zeros(4, bool)
    free, _, _ = tet_trajectory(str(tmp_path / "free"), X0, None, V0, mass, fixed)
    damped, _, _ = tet_trajectory(str(tmp_path / "damped"), X0, 0.01, V0, mass, fixed)
    L0, L1 = angular_momentum(free, V0, mass), angular_momentum(damped, V0, mass)
    assert abs(L1[-1] / L0[-1] - 1) < 1e-2, L1[-1] / L0[-1]
    ring = lambda t: np.ptp(np.linalg.norm(t[-20:, 1] - t[-20:, 0], axis=1))  # noqa: E731  (edge 0-1 length)
    assert ring(damped) < ring(free)  # while the internal vibration is damped
