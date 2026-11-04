#include <finite_element/finite_element_extra_constitution.h>
#include <finite_element/constitutions/anisotropic_fiber_function.h>
#include <finite_element/fem_utils.h>
#include <uipc/builtin/attribute_name.h>
#include <utils/matrix_assembler.h>
#include <cuda_tool/cuda_tool.h>

namespace uipc::backend::cuda
{
namespace
{
    namespace AF = anisotropic_fiber;

    constexpr SizeT StencilSize     = 4;
    constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    __global__ void AnisotropicFiber_do_compute_energy_kernel(
        cuda_tool::CBufferView<Vector4i>  tets,
        cuda_tool::CBufferView<Matrix3x3> Dm_invs,
        cuda_tool::CBufferView<Vector3>   directions,
        cuda_tool::CBufferView<Float>     stiffnesses,
        cuda_tool::CBufferView<Float>     volumes,
        cuda_tool::CBufferView<Vector3>   xs,
        cuda_tool::BufferView<Float>      energies,
        Float                             dt,
        int                               n)
    {
        int I = blockIdx.x * blockDim.x + threadIdx.x;
        if(I >= n)
            return;
        const Vector4i& tet = tets(I);
        const Vector3&  a   = directions(I);
        const Matrix3x3 F =
            fem::F(xs(tet(0)), xs(tet(1)), xs(tet(2)), xs(tet(3)), Dm_invs(I));
        const Vector3 d = F * a;
        const Float   s = AF::fiber_sign(F, a);

        energies(I) = AF::E(stiffnesses(I), d, s) * volumes(I) * dt * dt;
    }

    __global__ void AnisotropicFiber_do_compute_gradient_hessian_kernel(
        cuda_tool::CBufferView<Vector4i>       tets,
        cuda_tool::CBufferView<Matrix3x3>      Dm_invs,
        cuda_tool::CBufferView<Vector3>        directions,
        cuda_tool::CBufferView<Float>          stiffnesses,
        cuda_tool::CBufferView<Float>          volumes,
        cuda_tool::CBufferView<Vector3>        xs,
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
        const Vector3&   a      = directions(I);
        const Float      mu     = stiffnesses(I);
        const Matrix3x3 F = fem::F(xs(tet(0)), xs(tet(1)), xs(tet(2)), xs(tet(3)), Dm_inv);
        const Vector3 d    = F * a;
        const Float   s    = AF::fiber_sign(F, a);
        const Vector4 w    = AF::weights(Dm_inv, a);
        const Float   Vdt2 = volumes(I) * dt * dt;

        const Vector3 g = Vdt2 * AF::dEdd(mu, d, s);
        Vector12      G12;
#pragma unroll
        for(int k = 0; k < StencilSize; ++k)
            G12.segment<3>(3 * k) = w(k) * g;

        DoubletVectorAssembler DVA{G3s};
        DVA.segment<StencilSize>(I * StencilSize).write(tet, G12);

        if(gradient_only)
            return;

        // closed-form SPD projection; the positive Vdt2 and the rank-one weight pattern w w^T keep it SPD
        const Matrix3x3 M = Vdt2 * AF::ddEddd_spd(mu, d, s);
        Matrix12x12     H12x12;
#pragma unroll
        for(int k = 0; k < StencilSize; ++k)
        {
#pragma unroll
            for(int l = 0; l < StencilSize; ++l)
                H12x12.block<3, 3>(3 * k, 3 * l) = (w(k) * w(l)) * M;
        }

        TripletMatrixAssembler TMA{H3x3s};
        TMA.half_block<StencilSize>(I * HalfHessianSize).write(tet, H12x12);
    }
}  // namespace

class AnisotropicFiber final : public FiniteElementExtraConstitution
{
    // keep in sync with src/constitution/anisotropic_fiber.cpp (user-defined range while in the fork)
    static constexpr U64 ConstitutionUID = (1ull << 32) + 1;
    using Base                           = FiniteElementExtraConstitution;

  public:
    using Base::Base;
    U64 get_uid() const noexcept override { return ConstitutionUID; }

    vector<Vector4i>  h_tets;
    vector<Matrix3x3> h_Dm_invs;
    vector<Vector3>   h_directions;
    vector<Float>     h_stiffnesses;
    vector<Float>     h_volumes;

    cuda_tool::DeviceBuffer<Vector4i>  tets;
    cuda_tool::DeviceBuffer<Matrix3x3> Dm_invs;
    cuda_tool::DeviceBuffer<Vector3>   directions;
    cuda_tool::DeviceBuffer<Float>     stiffnesses;
    cuda_tool::DeviceBuffer<Float>     volumes;

    virtual void do_build(BuildInfo& info) override {}

    virtual void do_init(FilteredInfo& info) override
    {
        using ForEachInfo = FiniteElementMethod::ForEachInfo;
        auto geo_slots    = world().scene().geometries();
        auto x_bars = info.rest_positions();  // all FEM vertices, global index

        h_tets.clear();
        h_Dm_invs.clear();
        h_directions.clear();
        h_stiffnesses.clear();
        h_volumes.clear();

        info.for_each(  //
            geo_slots,
            [&](const ForEachInfo& I, geometry::SimplicialComplex& sc)
            {
                auto vertex_offset =
                    sc.meta().find<IndexT>(builtin::backend_fem_vertex_offset);
                UIPC_ASSERT(vertex_offset, "Vertex offset not found, why?");
                auto vo = vertex_offset->view().front();

                auto dir_attr = sc.tetrahedra().find<Vector3>("fiber_direction");
                auto mu_attr = sc.tetrahedra().find<Float>("fiber_stiffness");
                UIPC_ASSERT(dir_attr && mu_attr,
                            "AnisotropicFiber: fiber_direction / fiber_stiffness not found on the "
                            "tetrahedra; set them with AnisotropicFiber::apply_to");

                auto topo = sc.tetrahedra().topo().view();
                auto dirs = dir_attr->view();
                auto mus  = mu_attr->view();
                for(SizeT t = 0; t < topo.size(); ++t)
                {
                    const Float   mu = mus[t];
                    const Vector3 a  = dirs[t];
                    UIPC_ASSERT(mu >= 0,
                                "AnisotropicFiber: fiber_stiffness must be >= 0 (softening is not "
                                "supported), tetrahedron {} has {}",
                                t,
                                mu);
                    const Float na = a.norm();
                    if(mu == 0 || !(na > 0))  // no fiber in this tetrahedron
                        continue;

                    const Vector4i tet = topo[t] + Vector4i::Constant(vo);
                    const Vector3& x0  = x_bars[tet(0)];
                    const Vector3& x1  = x_bars[tet(1)];
                    const Vector3& x2  = x_bars[tet(2)];
                    const Vector3& x3  = x_bars[tet(3)];

                    h_tets.push_back(tet);
                    h_Dm_invs.push_back(fem::Dm_inv(x0, x1, x2, x3));
                    h_volumes.push_back(fem::Ds(x0, x1, x2, x3).determinant() / 6.0);
                    h_directions.push_back(a / na);
                    h_stiffnesses.push_back(mu);
                }
            });

        tets.resize(h_tets.size());
        tets.view().copy_from(h_tets.data());
        Dm_invs.resize(h_Dm_invs.size());
        Dm_invs.view().copy_from(h_Dm_invs.data());
        directions.resize(h_directions.size());
        directions.view().copy_from(h_directions.data());
        stiffnesses.resize(h_stiffnesses.size());
        stiffnesses.view().copy_from(h_stiffnesses.data());
        volumes.resize(h_volumes.size());
        volumes.view().copy_from(h_volumes.data());
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
        auto k = AnisotropicFiber_do_compute_energy_kernel;
        int  n = (int)tets.size();
        if(n > 0)
        {
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                tets.cview(),
                Dm_invs.cview(),
                directions.cview(),
                stiffnesses.cview(),
                volumes.cview(),
                info.xs(),
                info.energies(),
                info.dt(),
                n);
        }
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        auto k = AnisotropicFiber_do_compute_gradient_hessian_kernel;
        int  n = (int)tets.size();
        if(n > 0)
        {
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                tets.cview(),
                Dm_invs.cview(),
                directions.cview(),
                stiffnesses.cview(),
                volumes.cview(),
                info.xs(),
                info.gradients(),
                info.hessians(),
                info.dt(),
                info.gradient_only(),
                n);
        }
    }
};

REGISTER_SIM_SYSTEM(AnisotropicFiber);
}  // namespace uipc::backend::cuda
