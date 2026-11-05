"""Checks for the RayleighDamping extra constitution (fork branch `aniso`, plan fork_notes/rayleigh_damping.md).

Front end (step 2): the UID, the per-tetrahedron attribute, the rejected inputs and that a scene lists the
constitution. The physics (the CUDA kernels) is checked separately.
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
