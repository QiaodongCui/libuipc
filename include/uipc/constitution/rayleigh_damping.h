#pragma once
#include <uipc/common/type_define.h>
#include <uipc/constitution/finite_element_extra_constitution.h>

namespace uipc::constitution
{
/**
 * @brief Stiffness-proportional (Rayleigh beta) damping for StableNeoHookean tetrahedra, objective form.
 *
 * Damps the rate of the Green strain E = (F^T F - I) / 2 with the tetrahedron's small-strain
 * elasticity C (StableNeoHookean's Lame parameters at rest). In BDF1 incremental-potential form,
 * per tetrahedron, with F_n the deformation gradient at the start of the step:
 *   E_d = beta dt V / 2 * dE : C : dE,  dE = (F^T F - F_n^T F_n) / 2
 * Near rest this is the damping force -beta K v: a vibration mode of angular frequency w gets the
 * damping ratio beta w / 2. Rigid motion is never damped, at any stress.
 *
 * Apply it after StableNeoHookean. Per-tetrahedron attribute:
 *   damping_beta (Float, seconds; >= 0; zero = no damping)
 */
class UIPC_CONSTITUTION_API RayleighDamping : public FiniteElementExtraConstitution
{
    using Base = FiniteElementExtraConstitution;

  public:
    RayleighDamping(const Json& json = default_config());

    /**
     * @brief Set the same damping coefficient on every tetrahedron of `sc`.
     * @param beta Stiffness-proportional damping coefficient in seconds, >= 0.
     */
    void apply_to(geometry::SimplicialComplex& sc, Float beta) const;

    static Json default_config();

  private:
    virtual U64 get_uid() const noexcept final override;
    Json        m_config;
};
}  // namespace uipc::constitution
