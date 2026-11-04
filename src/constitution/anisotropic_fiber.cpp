#include <uipc/constitution/anisotropic_fiber.h>
#include <uipc/builtin/constitution_uid_auto_register.h>
#include <uipc/builtin/constitution_type.h>
#include <uipc/common/log.h>

namespace uipc::constitution
{
// A user-defined UID (>= 2^32) while this constitution lives in a fork; an official UID would be
// requested with an upstream pull request.
constexpr U64 AnisotropicFiberUID = builtin::UIDInfo::UserDefinedUIDStart + 1;

REGISTER_CONSTITUTION_UIDS()
{
    list<builtin::UIDInfo> uid_infos;
    builtin::UIDInfo       info;
    info.uid  = AnisotropicFiberUID;
    info.name = "AnisotropicFiber";
    info.type = string{builtin::FiniteElement};
    info.description = "Inversion-safe fiber energy (Anisotropic ARAP, Kim et al. 2019) layered on a 3D FEM material";
    uid_infos.push_back(info);
    return uid_infos;
}

AnisotropicFiber::AnisotropicFiber(const Json& json)
    : m_config{json}
{
}

void AnisotropicFiber::apply_to(geometry::SimplicialComplex& sc,
                                Float                        stiffness,
                                const Vector3&               direction) const
{
    UIPC_ASSERT_THROW(sc.dim() == 3,
                      "AnisotropicFiber only supports a 3D simplicial complex (tetrahedra), yours is {}D",
                      sc.dim());
    UIPC_ASSERT_THROW(stiffness >= 0,
                      "AnisotropicFiber stiffness must be >= 0 (softening is not supported), yours {}",
                      stiffness);

    Base::apply_to(sc);

    const Float   norm = direction.norm();
    const Vector3 a    = norm > 0 ? Vector3{direction / norm} : Vector3::Zero();

    auto dir = sc.tetrahedra().find<Vector3>("fiber_direction");
    if(!dir)
        dir = sc.tetrahedra().create<Vector3>("fiber_direction", Vector3::Zero());
    std::ranges::fill(geometry::view(*dir), a);

    auto k = sc.tetrahedra().find<Float>("fiber_stiffness");
    if(!k)
        k = sc.tetrahedra().create<Float>("fiber_stiffness", 0.0);
    std::ranges::fill(geometry::view(*k), stiffness);
}

U64 AnisotropicFiber::get_uid() const noexcept
{
    return AnisotropicFiberUID;
}

Json AnisotropicFiber::default_config()
{
    return Json::object();
}
}  // namespace uipc::constitution
