#include <pyuipc/constitution/anisotropic_fiber.h>
#include <uipc/constitution/finite_element_extra_constitution.h>
#include <uipc/constitution/anisotropic_fiber.h>
#include <pybind11/eigen.h>

namespace pyuipc::constitution
{
using namespace uipc::constitution;
PyAnisotropicFiber::PyAnisotropicFiber(py::module& m)
{
    auto class_AnisotropicFiber =
        py::class_<AnisotropicFiber, FiniteElementExtraConstitution>(m,
                                                                     "AnisotropicFiber",
                                                                     R"(Inversion-safe fiber energy (Anisotropic ARAP, Kim et al. 2019), layered on a 3D FEM material.

Per tetrahedron: Psi = mu_f / 2 * (sqrt(I5) - sign(I4))^2, with I5 = |F a|^2 and I4 = a^T S a (F = R S),
a the rest-space fiber direction and mu_f the fiber stiffness. Apply it after the primary 3D material.)");

    class_AnisotropicFiber.def(py::init<const Json&>(),
                               py::arg("config") = AnisotropicFiber::default_config(),
                               R"(Create an AnisotropicFiber constitution.
Args:
    config: Configuration dictionary (optional, uses default if not provided).)");

    class_AnisotropicFiber.def_static("default_config",
                                      &AnisotropicFiber::default_config,
                                      R"(Get the default AnisotropicFiber configuration.
Returns:
    dict: Default configuration dictionary.)");

    class_AnisotropicFiber.def("apply_to",
                               &AnisotropicFiber::apply_to,
                               py::arg("sc"),
                               py::arg("stiffness"),
                               py::arg("direction"),
                               R"(Set the same fiber on every tetrahedron of a 3D simplicial complex.

Creates (or overwrites) the per-tetrahedron attributes fiber_direction (rest space, normalized) and
fiber_stiffness; edit them afterwards for per-tetrahedron fibers. Zero direction or zero stiffness
means no fiber in that tetrahedron.
Args:
    sc: SimplicialComplex (tetrahedra) that already has its primary 3D FEM material.
    stiffness: Fiber stiffness mu_f in Pa, >= 0 (softening is not supported).
    direction: Rest-space fiber direction (3-vector).)");
}
}  // namespace pyuipc::constitution
