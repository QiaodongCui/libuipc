#pragma once
#include <uipc/common/type_define.h>
#include <uipc/constitution/finite_element_extra_constitution.h>

namespace uipc::constitution
{
/**
 * @brief Stiffness-proportional (Rayleigh beta) damping for StableNeoHookean tetrahedra.
 *
 * Adds the damping force -beta K v, K the PSD-projected StableNeoHookean stiffness at the start of
 * the step. In BDF1 incremental-potential form, per tetrahedron:
 *   E_d = beta dt V / 2 * dF^T H(F_n) dF,  dF = vec(F - F_n)
 * A vibration mode of angular frequency w gets the damping ratio beta w / 2; rigid motion is not
 * damped (up to a small leak for stressed elements).
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
