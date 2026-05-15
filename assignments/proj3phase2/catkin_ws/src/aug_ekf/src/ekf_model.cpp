#include <ekf_model.h>

namespace ekf_imu_vision {

  // 从欧拉角 (phi, theta, psi) 得到旋转矩阵 R (Z-X-Y 顺序)
// 右雅可比 Jr(q)，满足 ω_body = Jr(q) * dq/dt
// 适用于 Z-X-Y 欧拉角 (R = Rz*Rx*Ry)
inline Mat3x3 eulerRightJacobian(const Vec3& q) {
  double phi = q(0), theta = q(1);
  double cp = cos(phi), sp = sin(phi);
  double ct = cos(theta), st = sin(theta);
  
  Mat3x3 Jr;
  Jr << ct,  0,  -st * cp,
         0,  1,   sp,
        st,  0,   ct * cp;
  return Jr;
}

// 右雅可比的逆 Jr^{-1}(q)，满足 dq/dt = Jr^{-1}(q) * ω_body
inline Mat3x3 eulerRightJacobianInv(const Vec3& q) {
  double phi = q(0), theta = q(1);
  double cp = cos(phi), sp = sin(phi);
  double ct = cos(theta), st = sin(theta);
  double inv_cp = 1.0 / cp;  // 注意：φ ≠ ±π/2
  
  Mat3x3 Jr_inv;
  Jr_inv << ct,   0,       st,
            sp * st * inv_cp, 1, -ct * sp * inv_cp,
           -st * inv_cp, 0,  ct * inv_cp;
  return Jr_inv;
}

inline Mat3x3 eulerGinv(const Vec3& q) {
  return eulerRightJacobianInv(q);
}

Mat3x3 eulerToR(const Vec3& q) {
  double phi = q(0), theta = q(1), psi = q(2);
  Mat3x3 R;
  // 按 Z-X-Y 顺序: R = Rz(psi) * Rx(phi) * Ry(theta)
  R(0,0) =  cos(psi)*cos(theta) - sin(phi)*sin(psi)*sin(theta);
  R(0,1) = -cos(phi)*sin(psi);
  R(0,2) =  cos(psi)*sin(theta) + cos(theta)*sin(phi)*sin(psi);
  R(1,0) =  cos(theta)*sin(psi) + cos(psi)*sin(phi)*sin(theta);
  R(1,1) =  cos(phi)*cos(psi);
  R(1,2) =  sin(psi)*sin(theta) - cos(psi)*cos(theta)*sin(phi);
  R(2,0) = -cos(phi)*sin(theta);
  R(2,1) =  sin(phi);
  R(2,2) =  cos(phi)*cos(theta);
  return R;
}

Vec3 rotation2Euler(const Mat3x3& R) {
  double s = R(2,1);
  if (s > 1.0) s = 1.0;
  if (s < -1.0) s = -1.0;
  double phi = asin(s);
  double theta = atan2(-R(2,0), R(2,2));
  double psi   = atan2(-R(0,1), R(1,1));
  return Vec3(phi, theta, psi);
}


Vec15 modelF(const Vec15& x, const Vec6& u, const Vec12& n) {
  Vec3 q = x.segment<3>(3);
  Vec3 v = x.segment<3>(6);
  Vec3 bg = x.segment<3>(9);
  Vec3 ba = x.segment<3>(12);
  
  Vec3 omega_m = u.segment<3>(0);  // 陀螺仪测量
  Vec3 a_m = u.segment<3>(3);      // 加速度计测量
  
  Vec3 ng = n.segment<3>(0);
  Vec3 na = n.segment<3>(3);
  Vec3 nbg = n.segment<3>(6);
  Vec3 nba = n.segment<3>(9);

  Mat3x3 R = eulerToR(q);
  Mat3x3 Jr_inv = eulerRightJacobianInv(q);  // 修正：使用新的 Jr^{-1}

  Vec15 xdot;
  xdot << v,
          Jr_inv * (omega_m - bg - ng),      // 修正
          Vec3(0, 0, -9.81) + R * (a_m - ba - na),
          nbg,
          nba;
  return xdot;
}

Mat3x3 d_Jrinv_omega_dq(const Vec3& q, const Vec3& omega) {
  double phi = q(0), theta = q(1);
  double cp = cos(phi), sp = sin(phi);
  double ct = cos(theta), st = sin(theta);
  double inv_cp = 1.0 / cp;
  double inv_cp2 = inv_cp * inv_cp;
  
  double wx = omega(0), wz = omega(2);
  
  // Jr^{-1}(q) = [ ct,   0,       st;
  //                sp*st/cp,  1,  -ct*sp/cp;
  //               -st/cp,   0,   ct/cp ]
  //
  // d(Jr^{-1} * ω)/dφ = d(Jr^{-1})/dφ * ω
  // d(Jr^{-1})/dφ = [ 0,  0,  0;
  //                    (cp*cp + sp*sp)/cp^2 * st,    0,  -ct*(cp*cp + sp*sp)/cp^2;
  //                    sp*st/cp^2?...,              0,  -sp*ct/cp^2 ]
  
  Mat3x3 J;
  
  // 第0列：对 φ 的导数
  J(0,0) = 0;
  // d(sp*st/cp)/dφ = (cp*st*cp - sp*st*(-sp))/cp^2 = st/cp^2
  J(1,0) = st * inv_cp2 * wx - ct * inv_cp2 * wz;
  J(2,0) = -st * sp * inv_cp2 * wx + ct * sp * inv_cp2 * wz;

  // 第1列：对 θ 的导数
  J(0,1) = -st * wx + ct * wz;
  J(1,1) = sp * ct * inv_cp * wx + sp * st * inv_cp * wz;
  J(2,1) = -ct * inv_cp * wx - st * inv_cp * wz;

  // 第2列：对 ψ 的导数 — 全0
  J(0,2) = 0;
  J(1,2) = 0;
  J(2,2) = 0;

  return J;
}

Mat3x3 d_R_a_dq(const Vec3& q, const Vec3& a) {
  double phi = q(0), theta = q(1), psi = q(2);
  double sp = sin(phi), cp = cos(phi);
  double st = sin(theta), ct = cos(theta);
  double sy = sin(psi), cy = cos(psi);

  double ax = a(0), ay = a(1), az = a(2);

  Mat3x3 J;

  // 第0列：对 phi 的导数
  J(0,0) = -cp * sy * st * ax  +  sp * sy * ay  +  cp * sy * ct * az;
  J(1,0) =  cy * cp * st * ax  -  sp * cy * ay  -  cy * cp * ct * az;
  J(2,0) =  sp * st * ax  +  cp * ay  -  sp * ct * az;

  // 第1列：对 theta 的导数
  J(0,1) = (-cy * st - sp * sy * ct) * ax  +  (cy * ct - sp * sy * st) * az;
  J(1,1) = (-sy * st + cy * sp * ct) * ax  +  (sy * ct + cy * sp * st) * az;
  J(2,1) = -cp * ct * ax  -  cp * st * az;

  // 第2列：对 psi 的导数
  J(0,2) = (-sy * ct - sp * cy * st) * ax  -  cp * cy * ay  +  (-sy * st + sp * cy * ct) * az;
  J(1,2) = ( cy * ct - sp * sy * st) * ax  -  cp * sy * ay  +  ( cy * st + sp * sy * ct) * az;
  J(2,2) = 0.0;

  return J;
}

Mat15x15 jacobiFx(const Vec15& x, const Vec6& u, const Vec12& n) {
  Vec3 q = x.segment<3>(3);
  Vec3 bg = x.segment<3>(9);
  Vec3 ba = x.segment<3>(12);
  Vec3 omega_m = u.segment<3>(0);
  Vec3 a_m     = u.segment<3>(3);

  Mat3x3 Jr_inv = eulerRightJacobianInv(q);
  Mat3x3 R = eulerToR(q);
  
  Vec3 omega_eff = omega_m - bg;
  Vec3 a_eff     = a_m - ba;

  Mat15x15 F = Mat15x15::Zero();

  F.block<3,3>(0, 6) = Mat3x3::Identity();  // ∂pdot/∂v

  // ∂qdot/∂q：Jr^{-1}(q) * ω_eff 对 q 的导数
  // 使用数值差分或解析推导（此处使用之前已有的 d_Ginv_omega_dq，
  // 但需确保它基于新的 Jr_inv）
  F.block<3,3>(3, 3) = d_Jrinv_omega_dq(q, omega_eff);  // 需要重写

  F.block<3,3>(3, 9) = -Jr_inv;  // ∂qdot/∂bg（修正）

  // ∂vdot/∂q：R(q) * a_eff 对 q 的导数
  F.block<3,3>(6, 3) = d_R_a_dq(q, a_eff);  // 保持不变

  F.block<3,3>(6,12) = -R;  // ∂vdot/∂ba

  return F;
}

Mat15x12 jacobiFn(const Vec15& x, const Vec6& u, const Vec12& n) {
  Vec3 q = x.segment<3>(3);
  Mat3x3 Jr_inv = eulerRightJacobianInv(q);  // 修正
  Mat3x3 R = eulerToR(q);

  Mat15x12 Fn = Mat15x12::Zero();
  Fn.block<3,3>(3, 0) = -Jr_inv;             // ∂qdot / ∂n_g（修正）
  Fn.block<3,3>(6, 3) = -R;                  // ∂vdot / ∂n_a
  Fn.block<3,3>(9, 6) = Mat3x3::Identity();   // ∂b_gdot / ∂n_bg
  Fn.block<3,3>(12,9) = Mat3x3::Identity();   // ∂b_adot / ∂n_ba
  return Fn;
}

/* ============================== model of PnP ============================== */

Vec6 modelG1(const Vec15& x, const Vec6& v) {
  Vec6 z;
  z << x.head<3>(), x.segment<3>(3);
  return z + v;
}

Mat6x15 jacobiG1x(const Vec15& x, const Vec6& v) {
  Mat6x15 C = Mat6x15::Zero();
  C.block<3,3>(0,0) = Mat3x3::Identity();
  C.block<3,3>(3,3) = Mat3x3::Identity();
  return C;
}

Mat6x6 jacobiG1v(const Vec15& x, const Vec6& v) {
  return Mat6x6::Identity();
}

/* ============================== model of stereo VO relative pose ============================== */

Vec6 modelG2(const Vec21& x, const Vec6& v) {
  Vec3 p  = x.segment<3>(0);
  Vec3 q  = x.segment<3>(3);
  Vec3 pK = x.segment<3>(15);
  Vec3 qK = x.segment<3>(18);

  Mat3x3 R  = eulerToR(q);
  Mat3x3 RK = eulerToR(qK);
  Mat3x3 R_rel = RK.transpose() * R;
  Vec3 euler_rel = rotation2Euler(R_rel);   // 使用已有的函数
  Vec6 z;
  z << RK.transpose() * (p - pK), euler_rel;
  return z + v;
}
Mat6x21 jacobiG2x(const Vec21& x, const Vec6& v) {
  (void)v;
  Vec3 p  = x.segment<3>(0);
  Vec3 q  = x.segment<3>(3);
  Vec3 pK = x.segment<3>(15);
  Vec3 qK = x.segment<3>(18);

  Mat3x3 R   = eulerToR(q);
  Mat3x3 RK  = eulerToR(qK);
  Mat3x3 RKt = RK.transpose();
  Mat3x3 R_rel = RKt * R;
  Vec3 delta_q = rotation2Euler(R_rel);

  Mat3x3 G_q      = eulerGinv(q).inverse();
  Mat3x3 G_qK     = eulerGinv(qK).inverse();
  Mat3x3 G_dq_inv = eulerGinv(delta_q);

  Mat6x21 C = Mat6x21::Zero();

  C.block<3,3>(0, 0) = RKt;
  C.block<3,3>(0, 15) = -RKt;

  Vec3 d = p - pK;
  Vec3 a = RKt * d;
  C.block<3,3>(0, 18) = -RKt * d_R_a_dq(qK, a);

  C.block<3,3>(3, 3) = G_dq_inv * G_q;
  C.block<3,3>(3, 18) = -G_dq_inv * R_rel.transpose() * G_qK;

  return C;
}

Mat6x6 jacobiG2v(const Vec21& x, const Vec6& v) {
  return Mat6x6::Identity();
}

}  // namespace ekf_imu_vision
