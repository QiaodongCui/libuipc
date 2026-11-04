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
