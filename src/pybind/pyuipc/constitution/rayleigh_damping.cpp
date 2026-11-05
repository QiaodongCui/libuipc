#include <pyuipc/constitution/rayleigh_damping.h>
#include <uipc/constitution/finite_element_extra_constitution.h>
#include <uipc/constitution/rayleigh_damping.h>

namespace pyuipc::constitution
{
using namespace uipc::constitution;
PyRayleighDamping::PyRayleighDamping(py::module& m)
{
    auto class_RayleighDamping =
        py::class_<RayleighDamping, FiniteElementExtraConstitution>(m,
                                                                    "RayleighDamping",
                                                                    R"(Stiffness-proportional (Rayleigh beta) damping for StableNeoHookean tetrahedra.

Adds the damping force -beta K v, K the PSD-projected StableNeoHookean stiffness at the start of the step:
a vibration mode of angular frequency w gets the damping ratio beta w / 2, and rigid motion is not damped
(up to a small leak for stressed elements). Apply it after StableNeoHookean.)");

    class_RayleighDamping.def(py::init<const Json&>(),
                              py::arg("config") = RayleighDamping::default_config(),
                              R"(Create a RayleighDamping constitution.
Args:
    config: Configuration dictionary (optional, uses default if not provided).)");

    class_RayleighDamping.def_static("default_config",
                                     &RayleighDamping::default_config,
                                     R"(Get the default RayleighDamping configuration.
Returns:
    dict: Default configuration dictionary.)");

    class_RayleighDamping.def("apply_to",
                              &RayleighDamping::apply_to,
                              py::arg("sc"),
                              py::arg("beta"),
                              R"(Set the same damping coefficient on every tetrahedron of a 3D simplicial complex.

Creates (or overwrites) the per-tetrahedron attribute damping_beta; edit it afterwards for per-tetrahedron
damping (zero means no damping in that tetrahedron).
Args:
    sc: SimplicialComplex (tetrahedra) with StableNeoHookean already applied.
    beta: Stiffness-proportional damping coefficient in seconds, >= 0.)");
}
}  // namespace pyuipc::constitution
