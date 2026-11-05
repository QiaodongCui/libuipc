#include <uipc/constitution/rayleigh_damping.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/builtin/constitution_uid_auto_register.h>
#include <uipc/builtin/constitution_type.h>
#include <uipc/common/log.h>

namespace uipc::constitution
{
// A user-defined UID (>= 2^32) while this constitution lives in a fork (AnisotropicFiber is 2^32 + 1).
constexpr U64 RayleighDampingUID = builtin::UIDInfo::UserDefinedUIDStart + 2;
// the primary material it damps: StableNeoHookean (src/constitution/stable_neo_hookean.cpp)
constexpr U64 StableNeoHookeanUID = 10;

REGISTER_CONSTITUTION_UIDS()
{
    list<builtin::UIDInfo> uid_infos;
    builtin::UIDInfo       info;
    info.uid  = RayleighDampingUID;
    info.name = "RayleighDamping";
    info.type = string{builtin::FiniteElement};
    info.description = "Rayleigh beta damping of the Green strain rate for StableNeoHookean tetrahedra";
    uid_infos.push_back(info);
    return uid_infos;
}

RayleighDamping::RayleighDamping(const Json& json)
    : m_config{json}
{
}

void RayleighDamping::apply_to(geometry::SimplicialComplex& sc, Float beta) const
{
    UIPC_ASSERT_THROW(sc.dim() == 3,
                      "RayleighDamping only supports a 3D simplicial complex (tetrahedra), yours is {}D",
                      sc.dim());
    UIPC_ASSERT_THROW(beta >= 0, "RayleighDamping beta must be >= 0, yours {}", beta);
    auto cuid = sc.meta().find<U64>(builtin::constitution_uid);
    UIPC_ASSERT_THROW(cuid && cuid->view().front() == StableNeoHookeanUID,
                      "RayleighDamping damps StableNeoHookean: apply StableNeoHookean first (constitution UID {})",
                      cuid ? cuid->view().front() : 0);

    Base::apply_to(sc);

    auto b = sc.tetrahedra().find<Float>("damping_beta");
    if(!b)
        b = sc.tetrahedra().create<Float>("damping_beta", 0.0);
    std::ranges::fill(geometry::view(*b), beta);
}

U64 RayleighDamping::get_uid() const noexcept
{
    return RayleighDampingUID;
}

Json RayleighDamping::default_config()
{
    return Json::object();
}
}  // namespace uipc::constitution
