#pragma once
#include <uipc/common/type_define.h>
#include <uipc/constitution/finite_element_extra_constitution.h>

namespace uipc::constitution
{
/**
 * @brief Inversion-safe fiber (transversely isotropic) energy, layered on a 3D FEM material.
 *
 * The Anisotropic ARAP energy of Kim et al. 2019, "Anisotropic Elasticity for Inversion-Safety and
 * Element Rehabilitation" (eq. 31), per tetrahedron:
 *   Psi = mu_f / 2 * (sqrt(I5) - sign(I4))^2,  I5 = |F a|^2,  I4 = a^T S a  (F = R S)
 * where a is the rest-space fiber direction and mu_f the fiber stiffness. It adds mu_f to the
 * small-strain stiffness along the fiber and nothing across it.
 *
 * Apply it after the primary 3D material (e.g. StableNeoHookean). Per-tetrahedron attributes:
 *   fiber_direction (Vector3, rest space; normalized by the backend; zero = no fiber)
 *   fiber_stiffness (Float, Pa; >= 0; zero = no fiber)
 */
class UIPC_CONSTITUTION_API AnisotropicFiber : public FiniteElementExtraConstitution
{
    using Base = FiniteElementExtraConstitution;

  public:
    AnisotropicFiber(const Json& json = default_config());

    /**
     * @brief Set the same fiber on every tetrahedron of `sc`.
     * @param stiffness Fiber stiffness mu_f in Pa, >= 0 (softening is not supported).
     * @param direction Rest-space fiber direction; normalized here, zero means no fiber.
     */
    void apply_to(geometry::SimplicialComplex& sc, Float stiffness, const Vector3& direction) const;

    static Json default_config();

  private:
    virtual U64 get_uid() const noexcept final override;
    Json        m_config;
};
}  // namespace uipc::constitution
