#include <finite_element/finite_element_extra_constitution.h>
#include <finite_element/fem_utils.h>
#include <finite_element/matrix_utils.h>
#include <uipc/builtin/attribute_name.h>
#include <utils/matrix_assembler.h>
#include <cuda_tool/cuda_tool.h>

// Rayleigh beta damping for StableNeoHookean tetrahedra, objective form: a damping of the Green strain rate, as a
// BDF1 incremental-potential term
//   E_d = beta dt V / 2 * dE : C : dE,  dE = (F^T F - F_n^T F_n) / 2
// with F_n the step-start deformation gradient (x_prevs) and C the small-strain elasticity of the tet's SNK1 (Lame
// mu, lambda - mu). At rest it is beta K v (a mode of angular frequency w gets the damping ratio beta w / 2); a
// rotation within the step leaves F^T F unchanged, so rigid motion is never damped, at any stress. Gradient
// beta dt V * F (C : dE); Hessian beta dt V * M with M[dF, dG] = sym(F^T dF) : C : sym(F^T dG), PSD (the small
// geometric term I (x) (C : dE) is left out). Plan and NumPy reference: fork_notes/rayleigh_damping*.
namespace uipc::backend::cuda
{
namespace
{
    constexpr SizeT StencilSize     = 4;
    constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    // C : dE
    __device__ Matrix3x3 green_stress(const Matrix3x3& dE, Float mu_L, Float lam_L)
    {
        return 2.0 * mu_L * dE + lam_L * dE.trace() * Matrix3x3::Identity();
    }

    __global__ void RayleighDamping_do_compute_energy_kernel(
        cuda_tool::CBufferView<Vector4i>  tets,
        cuda_tool::CBufferView<Matrix3x3> Dm_invs,
        cuda_tool::CBufferView<Float>     volumes,
        cuda_tool::CBufferView<Float>     lame_mus,
        cuda_tool::CBufferView<Float>     lame_lambdas,
        cuda_tool::CBufferView<Float>     betas,
        cuda_tool::CBufferView<Vector3>   xs,
        cuda_tool::CBufferView<Vector3>   x_prevs,
        cuda_tool::BufferView<Float>      energies,
        Float                             dt,
        int                               n)
    {
        int I = blockIdx.x * blockDim.x + threadIdx.x;
        if(I >= n)
            return;
        const Vector4i&  tet    = tets(I);
        const Matrix3x3& Dm_inv = Dm_invs(I);

        const Matrix3x3 F = fem::F(xs(tet(0)), xs(tet(1)), xs(tet(2)), xs(tet(3)), Dm_inv);
        const Matrix3x3 Fn =
            fem::F(x_prevs(tet(0)), x_prevs(tet(1)), x_prevs(tet(2)), x_prevs(tet(3)), Dm_inv);

        const Matrix3x3 dE = 0.5 * (F.transpose() * F - Fn.transpose() * Fn);
        const Matrix3x3 S  = green_stress(dE, lame_mus(I), lame_lambdas(I));

        energies(I) = 0.5 * betas(I) * dt * volumes(I) * dE.cwiseProduct(S).sum();
    }

    __global__ void RayleighDamping_do_compute_gradient_hessian_kernel(
        cuda_tool::CBufferView<Vector4i>       tets,
        cuda_tool::CBufferView<Matrix3x3>      Dm_invs,
        cuda_tool::CBufferView<Float>          volumes,
        cuda_tool::CBufferView<Float>          lame_mus,
        cuda_tool::CBufferView<Float>          lame_lambdas,
        cuda_tool::CBufferView<Float>          betas,
        cuda_tool::CBufferView<Vector3>        xs,
        cuda_tool::CBufferView<Vector3>        x_prevs,
        cuda_tool::DoubletVectorView<Float, 3> G3s,
        cuda_tool::TripletMatrixView<Float, 3> H3x3s,
        Float                                  dt,
        bool                                   gradient_only,
        int                                    n)
    {
        int I = blockIdx.x * blockDim.x + threadIdx.x;
        if(I >= n)
            return;
        const Vector4i&  tet    = tets(I);
        const Matrix3x3& Dm_inv = Dm_invs(I);
        const Float      mu_L   = lame_mus(I);
        const Float      lam_L  = lame_lambdas(I);
        const Float      scale  = betas(I) * dt * volumes(I);

        const Matrix3x3 F = fem::F(xs(tet(0)), xs(tet(1)), xs(tet(2)), xs(tet(3)), Dm_inv);
        const Matrix3x3 Fn =
            fem::F(x_prevs(tet(0)), x_prevs(tet(1)), x_prevs(tet(2)), x_prevs(tet(3)), Dm_inv);

        const Matrix3x3 dE = 0.5 * (F.transpose() * F - Fn.transpose() * Fn);
        const Vector9 dEdF = scale * flatten(F * green_stress(dE, mu_L, lam_L));

        // M in F space (column-major flatten, index 3 col + row):
        //   M(3b + i, 3d + k) = mu_L ((F F^T)_ik delta_bd + F_id F_kb) + lam_L F_ib F_kd
        Matrix9x9 Hd;
        if(!gradient_only)
        {
            const Matrix3x3 FFt = F * F.transpose();
#pragma unroll
            for(int b = 0; b < 3; ++b)
#pragma unroll
                for(int i = 0; i < 3; ++i)
#pragma unroll
                    for(int d = 0; d < 3; ++d)
#pragma unroll
                        for(int k = 0; k < 3; ++k)
                            Hd(3 * b + i, 3 * d + k) =
                                scale
                                * (mu_L * ((b == d ? FFt(i, k) : 0.0) + F(i, d) * F(k, b))
                                   + lam_L * F(i, b) * F(k, d));
        }

        auto shape_gradients = fem::tetrahedron_shape_gradients(Dm_inv);

        DoubletVectorAssembler DVA{G3s};
#pragma unroll
        for(int i = 0; i < StencilSize; ++i)
        {
            Vector3 G = fem::project_F_gradient(shape_gradients.col(i), dEdF);
            DVA(I * StencilSize + i).write(tet(i), G);
        }

        if(gradient_only)
            return;

        IndexT hessian_offset = I * HalfHessianSize;
#pragma unroll
        for(int i = 0; i < StencilSize; ++i)
        {
#pragma unroll
            for(int j = i; j < StencilSize; ++j)
            {
                int left  = i;
                int right = j;
                if(tet(left) > tet(right))
                {
                    left  = j;
                    right = i;
                }

                Matrix3x3 H = fem::project_F_hessian_block(
                    shape_gradients.col(left), Hd, shape_gradients.col(right));
                H3x3s(hessian_offset++).write(tet(left), tet(right), H);
            }
        }
    }
}  // namespace

class RayleighDamping final : public FiniteElementExtraConstitution
{
    // keep in sync with src/constitution/rayleigh_damping.cpp (user-defined range while in the fork)
    static constexpr U64 ConstitutionUID     = (1ull << 32) + 2;
    static constexpr U64 StableNeoHookeanUID = 10;
    using Base                               = FiniteElementExtraConstitution;

  public:
    using Base::Base;
    U64 get_uid() const noexcept override { return ConstitutionUID; }

    FiniteElementMethod* fem = nullptr;  // for the step-start positions (x_prevs)

    vector<Vector4i>  h_tets;
    vector<Matrix3x3> h_Dm_invs;
    vector<Float>     h_volumes;
    vector<Float> h_lame_mus;  // small-strain Lame parameters of the tet's SNK1
    vector<Float> h_lame_lambdas;  // (stored mu, lambda - mu), lambda kept >= -2/3 mu so C is PSD
    vector<Float> h_betas;

    cuda_tool::DeviceBuffer<Vector4i>  tets;
    cuda_tool::DeviceBuffer<Matrix3x3> Dm_invs;
    cuda_tool::DeviceBuffer<Float>     volumes;
    cuda_tool::DeviceBuffer<Float>     lame_mus;
    cuda_tool::DeviceBuffer<Float>     lame_lambdas;
    cuda_tool::DeviceBuffer<Float>     betas;

    virtual void do_build(BuildInfo& info) override
    {
        fem = &require<FiniteElementMethod>();
    }

    virtual void do_init(FilteredInfo& info) override
    {
        using ForEachInfo = FiniteElementMethod::ForEachInfo;
        auto geo_slots    = world().scene().geometries();
        auto x_bars = info.rest_positions();  // all FEM vertices, global index

        h_tets.clear();
        h_Dm_invs.clear();
        h_volumes.clear();
        h_lame_mus.clear();
        h_lame_lambdas.clear();
        h_betas.clear();

        info.for_each(  //
            geo_slots,
            [&](const ForEachInfo& I, geometry::SimplicialComplex& sc)
            {
                auto vertex_offset =
                    sc.meta().find<IndexT>(builtin::backend_fem_vertex_offset);
                UIPC_ASSERT(vertex_offset, "Vertex offset not found, why?");
                auto vo = vertex_offset->view().front();

                auto cuid = sc.meta().find<U64>(builtin::constitution_uid);
                UIPC_ASSERT(cuid && cuid->view().front() == StableNeoHookeanUID,
                            "RayleighDamping needs StableNeoHookean as the primary material");
                auto beta_attr = sc.tetrahedra().find<Float>("damping_beta");
                auto mu_attr   = sc.tetrahedra().find<Float>("mu");
                auto lam_attr  = sc.tetrahedra().find<Float>("lambda");
                UIPC_ASSERT(beta_attr && mu_attr && lam_attr,
                            "RayleighDamping: damping_beta / mu / lambda not found on the tetrahedra");

                auto topo  = sc.tetrahedra().topo().view();
                auto bs    = beta_attr->view();
                auto mus_v = mu_attr->view();
                auto lms_v = lam_attr->view();
                for(SizeT t = 0; t < topo.size(); ++t)
                {
                    const Float beta = bs[t];
                    UIPC_ASSERT(beta >= 0,
                                "RayleighDamping: damping_beta must be >= 0, tetrahedron {} has {}",
                                t,
                                beta);
                    if(beta == 0)  // no damping in this tetrahedron
                        continue;

                    const Vector4i tet = topo[t] + Vector4i::Constant(vo);
                    const Vector3& x0  = x_bars[tet(0)];
                    const Vector3& x1  = x_bars[tet(1)];
                    const Vector3& x2  = x_bars[tet(2)];
                    const Vector3& x3  = x_bars[tet(3)];

                    h_tets.push_back(tet);
                    h_Dm_invs.push_back(fem::Dm_inv(x0, x1, x2, x3));
                    h_volumes.push_back(fem::Ds(x0, x1, x2, x3).determinant() / 6.0);
                    h_lame_mus.push_back(mus_v[t]);
                    h_lame_lambdas.push_back(
                        std::max(lms_v[t] - mus_v[t], -2.0 / 3.0 * mus_v[t]));
                    h_betas.push_back(beta);
                }
            });

        tets.resize(h_tets.size());
        tets.view().copy_from(h_tets.data());
        Dm_invs.resize(h_Dm_invs.size());
        Dm_invs.view().copy_from(h_Dm_invs.data());
        volumes.resize(h_volumes.size());
        volumes.view().copy_from(h_volumes.data());
        lame_mus.resize(h_lame_mus.size());
        lame_mus.view().copy_from(h_lame_mus.data());
        lame_lambdas.resize(h_lame_lambdas.size());
        lame_lambdas.view().copy_from(h_lame_lambdas.data());
        betas.resize(h_betas.size());
        betas.view().copy_from(h_betas.data());
    }

    virtual void do_report_extent(ReportExtentInfo& info) override
    {
        info.energy_count(tets.size());
        info.gradient_count(tets.size() * StencilSize);

        if(info.gradient_only())
            return;

        info.hessian_count(tets.size() * HalfHessianSize);
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        auto k = RayleighDamping_do_compute_energy_kernel;
        int  n = (int)tets.size();
        if(n > 0)
        {
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                tets.cview(),
                Dm_invs.cview(),
                volumes.cview(),
                lame_mus.cview(),
                lame_lambdas.cview(),
                betas.cview(),
                info.xs(),
                fem->x_prevs().cview(),
                info.energies(),
                info.dt(),
                n);
        }
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        auto k = RayleighDamping_do_compute_gradient_hessian_kernel;
        int  n = (int)tets.size();
        if(n > 0)
        {
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                tets.cview(),
                Dm_invs.cview(),
                volumes.cview(),
                lame_mus.cview(),
                lame_lambdas.cview(),
                betas.cview(),
                info.xs(),
                fem->x_prevs().cview(),
                info.gradients(),
                info.hessians(),
                info.dt(),
                info.gradient_only(),
                n);
        }
    }
};

REGISTER_SIM_SYSTEM(RayleighDamping);
}  // namespace uipc::backend::cuda
