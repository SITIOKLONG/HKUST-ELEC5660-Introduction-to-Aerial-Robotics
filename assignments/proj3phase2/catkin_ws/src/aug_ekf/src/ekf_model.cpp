#include <ekf_model.h>

namespace ekf_imu_vision {

  // 从欧拉角 (phi, theta, psi) 得到旋转矩阵 R (Z-X-Y 顺序)
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
  double phi = asin(R(2,1));
  double theta = atan2(-R(2,0), R(2,2));
  double psi   = atan2(-R(0,1), R(1,1));
  return Vec3(phi, theta, psi);
}

Mat3x3 eulerGinv(const Vec3& q) {
  double phi = q(0), theta = q(1);
  double cp = cos(phi);
  double sp = sin(phi);
  double ct = cos(theta);
  double st = sin(theta);
  double tt = tan(theta);
  Mat3x3 Ginv;
  Ginv << ct,      0,  st,
      sp * tt, 1, -cp * tt,
      -sp/ct,  0,  cp/ct;
  return Ginv;
}


Vec15 modelF(const Vec15& x, const Vec6& u, const Vec12& n) {
  // TODO
  // return the model xdot = f(x,u,n)

  Vec3 p = x.segment<3>(0);
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

  Mat3x3 Ginv = eulerGinv(q);
  Vec15 xdot;
  xdot << v,
          Ginv * (omega_m - bg - ng),
          Vec3(0, 0, -9.81) + R * (a_m - ba - na),
          nbg,
          nba;
  return xdot;
}

Mat3x3 d_Ginv_omega_dq(const Vec3& q, const Vec3& omega) {
  double phi = q(0), theta = q(1);
  double cp = cos(phi), sp = sin(phi);
  double ct = cos(theta), st = sin(theta), tt = tan(theta);

  double wx = omega(0), wy = omega(1), wz = omega(2);

  Mat3x3 J;
  // 列0：对 phi
  J(0,0) = 0;
  J(1,0) = wx * cp * tt + wz * sp * tt;     // wx*cosφ*tanθ + wz*sinφ*tanθ
  J(2,0) = -wx * cp / ct - wz * sp / ct;    // -wx*cosφ/cosθ - wz*sinφ/cosθ

  // 列1：对 theta
  double c2 = ct * ct;   // cos^2 θ
  double factor = (wx * sp - wz * cp) / c2;
  J(0,1) = -wx * st + wz * ct;              // -wx*sinθ + wz*cosθ
  J(1,1) = factor;                          // (wx sinφ - wz cosφ)/cos^2θ
  J(2,1) = factor * st / ct;                // 等价于 factor * tanθ

  // 列2：对 psi —— 全0
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

  Mat3x3 Ginv = eulerGinv(q);

  Mat3x3 R = eulerToR(q);
  Vec3 omega_eff = omega_m - bg;
  Vec3 a_eff     = a_m - ba;

  Mat15x15 F = Mat15x15::Zero();

  F.block<3,3>(0, 6) = Mat3x3::Identity();         // ∂pdot/∂v

  // ∂qdot/∂q ：需要单独计算 d(Ginv * ω_eff) / dq
  F.block<3,3>(3, 3) = d_Ginv_omega_dq(q, omega_eff);  // 占位，需自行推导

  F.block<3,3>(3, 9) = -Ginv;                       // ∂qdot/∂b_g

  // ∂vdot/∂q ：需要计算 d(R * a_eff) / dq
  F.block<3,3>(6, 3) = d_R_a_dq(q, a_eff);          // 占位，需自行推导

  F.block<3,3>(6,12) = -R;                          // ∂vdot/∂b_a

  return F;
}

Mat15x12 jacobiFn(const Vec15& x, const Vec6& u, const Vec12& n) {
  Vec3 q = x.segment<3>(3);
  Mat3x3 Ginv = eulerGinv(q);

  Mat3x3 R = eulerToR(q);

  Mat15x12 Fn = Mat15x12::Zero();
  Fn.block<3,3>(3, 0) = -Ginv;             // ∂qdot / ∂n_g
  Fn.block<3,3>(6, 3) = -R;                // ∂vdot / ∂n_a
  Fn.block<3,3>(9, 6) = Mat3x3::Identity();  // ∂b_gdot / ∂n_bg
  Fn.block<3,3>(12,9) = Mat3x3::Identity();  // ∂b_adot / ∂n_ba
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
    // ------ 提取当前状态 ------
    Vec3 p  = x.segment<3>(0);   // 当前位置
    Vec3 q  = x.segment<3>(3);   // 当前姿态 (phi,theta,psi)
    Vec3 pK = x.segment<3>(15);  // 关键帧位置
    Vec3 qK = x.segment<3>(18);  // 关键帧姿态

    // ------ 旋转矩阵 ------
    Mat3x3 R   = eulerToR(q);
    Mat3x3 RK  = eulerToR(qK);
    Mat3x3 RKt = RK.transpose();          // R_K^T
    Mat3x3 R_rel = RKt * R;               // 相对旋转 R_K^T * R

    // 相对欧拉角（当前均值处）
    Vec3 delta_q = rotation2Euler(R_rel);

    // ------ G 矩阵 ------
    // 利用已有的 eulerGinv 求逆得到 G
    Mat3x3 G_q      = eulerGinv(q).inverse();
    Mat3x3 G_qK     = eulerGinv(qK).inverse();
    Mat3x3 G_dq_inv = eulerGinv(delta_q);   // G^{-1}(Δq)

    // ------ 初始化雅可比 ------
    Mat6x21 C = Mat6x21::Zero();

    // ===== 位置部分（前3行） =====
    // ∂Δp/∂p = R_K^T
    C.block<3,3>(0, 0) = RKt;
    // ∂Δp/∂pK = -R_K^T
    C.block<3,3>(0, 15) = -RKt;

    // ∂Δp/∂qK = ∂(R_K^T * (p - pK)) / ∂qK
    Vec3 d = p - pK;
    Vec3 a = RKt * d;                     // rotated vector for derivative
    C.block<3,3>(0, 18) = -RKt * d_R_a_dq(qK, a);

    // ===== 姿态部分（后3行） =====
    // ∂Δq/∂q = G^{-1}(Δq) * R_rel^T * G(q)???
    C.block<3,3>(3, 3) = G_dq_inv * G_q;

    // ∂Δq/∂qK = -G^{-1}(Δq) * R_rel^T * G(qK)???
    C.block<3,3>(3, 18) = -G_dq_inv * R_rel.transpose() * G_qK;

    return C;
}

Mat6x6 jacobiG2v(const Vec21& x, const Vec6& v) {
  return Mat6x6::Identity();
}

}  // namespace ekf_imu_vision
