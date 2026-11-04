#pragma once
#include <type_define.h>
#include <algorithm/qr_svd.hpp>

// Anisotropic ARAP fiber energy of Kim et al. 2019 (eq. 31), written on the fiber vector d = F a:
//   E = mu / 2 (|d| - s)^2,  s = sign(I4)
//   dE/dd = mu (1 - s / |d|) d
//   SPD-projected d2E/dd2 = mu n n^T + max(0, mu (1 - s / |d|)) (I - n n^T),  n = d / |d|
// d = sum_k w_k x_k with w = (-(b0 + b1 + b2), b0, b1, b2), b = Dm_inv a, so the vertex gradient is
// w_k dE/dd and the vertex Hessian block (k, l) is w_k w_l d2E/dd2. This equals the paper's PK1 (eq. 35),
// Hessian (eq. 36) and eigenvalues (eqs. 37-38) mapped through dF/dx; the NumPy reference that checks it is
// fork_notes/anisotropic_fiber_reference.py.
namespace uipc::backend::cuda::anisotropic_fiber
{
// |d| below this: no force and no stiffness (the paper zeroes the Hessian at I5 = 0)
constexpr Float MinFiberLength = 1e-12;

// s = sign(I4), I4 = a^T S a with F = R S, R a rotation. det F > 0 => S is positive definite => s = +1
// without an SVD. Otherwise the rotation-variant SVD (math::qr_svd: U, V rotations, only the smallest singular
// value may be negative) gives I4 = sum_i sigma_i (v_i . a)^2. I4 = 0 counts as +1 (the paper's patch).
UIPC_GENERIC inline Float fiber_sign(const Matrix3x3& F, const Vector3& a)
{
    if(F.determinant() > 0)
        return 1.0;
    Vector3   sigma;
    Matrix3x3 U, V;
    math::qr_svd(F, sigma, U, V);
    const Vector3 va = V.transpose() * a;
    const Float   I4 = sigma.dot(va.cwiseProduct(va));
    return I4 >= 0 ? 1.0 : -1.0;
}

// d = F a = sum_k w_k x_k
UIPC_GENERIC inline Vector4 weights(const Matrix3x3& Dm_inv, const Vector3& a)
{
    const Vector3 b = Dm_inv * a;
    return Vector4{-(b(0) + b(1) + b(2)), b(0), b(1), b(2)};
}

UIPC_GENERIC inline Float E(Float mu, const Vector3& d, Float s)
{
    const Float l = d.norm() - s;
    return 0.5 * mu * l * l;
}

UIPC_GENERIC inline Vector3 dEdd(Float mu, const Vector3& d, Float s)
{
    const Float nd = d.norm();
    if(nd < MinFiberLength)
        return Vector3::Zero();
    return mu * (1.0 - s / nd) * d;
}

UIPC_GENERIC inline Matrix3x3 ddEddd_spd(Float mu, const Vector3& d, Float s)
{
    const Float nd = d.norm();
    if(nd < MinFiberLength)
        return Matrix3x3::Zero();
    const Vector3   n     = d / nd;
    const Matrix3x3 nnT   = n * n.transpose();
    const Float     lam0  = mu > 0 ? mu : 0.0;
    const Float     l12   = mu * (1.0 - s / nd);
    const Float     lam12 = l12 > 0 ? l12 : 0.0;
    return lam0 * nnT + lam12 * (Matrix3x3::Identity() - nnT);
}
}  // namespace uipc::backend::cuda::anisotropic_fiber
