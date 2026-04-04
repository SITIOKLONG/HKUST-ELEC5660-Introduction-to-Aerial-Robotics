#include <Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

// slide p35
Eigen::Vector2d computeError(const Eigen::Vector2d &uv_obs,
                             const Eigen::Matrix3d &K,
                             const Eigen::Vector3d &X_w,
                             const Eigen::Matrix3d &R,
                             const Eigen::Vector3d &t) {
  Eigen::Vector3d uv_homo = K * (R * X_w + t);
  Eigen::Vector2d uv_pred(uv_homo.x() / uv_homo.z(), uv_homo.y() / uv_homo.z());
  return uv_obs - uv_pred;
}

Eigen::Matrix<double, 2, 6> computeJacobian(const Eigen::Matrix3d &K,
                                            const Eigen::Matrix3d &R,
                                            const Eigen::Vector3d &t,
                                            const Eigen::Vector3d &X_w) {
  Eigen::Vector3d X_r = R * X_w;      // Point without translation
  Eigen::Vector3d X_c = X_r + t;      // Fully projected point
  double x = X_c.x(), y = X_c.y(), z = X_c.z();
  double inv_z = 1.0 / z;
  double inv_z2 = inv_z * inv_z;

  // ∂γ/∂X_c
  Eigen::Matrix<double, 2, 3> d_gamma_d_Xc;
  d_gamma_d_Xc << -K(0, 0) * inv_z, 0, K(0, 0) * x * inv_z2, 0,
      -K(1, 1) * inv_z, K(1, 1) * y * inv_z2;

  // ∂X_c/∂t
  Eigen::Matrix3d d_Xc_dt = Eigen::Matrix3d::Identity();

  // ∂X_c/∂θ
  // ∂(R·X_w)/∂θ ≈ -[R·X_w]× (反对称矩阵)
  Eigen::Matrix3d d_Xc_dtheta;
  double rx = X_r.x(), ry = X_r.y(), rz = X_r.z();
  d_Xc_dtheta << 0, -rz, ry, rz, 0, -rx, -ry, rx, 0;
  d_Xc_dtheta = -d_Xc_dtheta;

  // chain rule
  // J = ∂γ/∂X_c · [∂X_c/∂θ | ∂X_c/∂t]
  Eigen::Matrix<double, 2, 6> J;
  J.block<2, 3>(0, 0) = d_gamma_d_Xc * d_Xc_dtheta;
  J.block<2, 3>(0, 3) = d_gamma_d_Xc * d_Xc_dt;
  return J;
}

void gaussNewtonPnP(const std::vector<cv::Point3f> &pts_3,
                    const std::vector<cv::Point2f> &pts_2,
                    const Eigen::Matrix3d &K,
                    Eigen::Matrix3d &R, // in/out
                    Eigen::Vector3d &t, int max_iter = 100, double tol = 1e-12) {
  size_t n = pts_3.size();
  double last_error = 1e10;

  for (int iter = 0; iter < max_iter; ++iter) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6, 6); // Slide 36: A
    Eigen::VectorXd b = Eigen::VectorXd::Zero(6);    // Slide 36: b
    double total_error = 0;

    for (size_t i = 0; i < n; ++i) {
      Eigen::Vector3d X_w(pts_3[i].x, pts_3[i].y, pts_3[i].z);
      Eigen::Vector2d uv_obs(pts_2[i].x, pts_2[i].y);

      // Step 1: 计算残差 γ_i(θ₀,t₀)
      Eigen::Vector2d gamma = computeError(uv_obs, K, X_w, R, t);
      total_error += gamma.squaredNorm();

      // Step 2: 计算雅可比 J_i
      Eigen::Matrix<double, 2, 6> J = computeJacobian(K, R, t, X_w);

      // Step 3: 累加法方程 A = ΣJ^TJ, b = -ΣJ^Tγ
      A += J.transpose() * J;
      b += -J.transpose() * gamma;
    }

    // converge
    if (std::abs(last_error - total_error) < tol * last_error) {
    std::cout << "[GN] CONVERGED" << iter << "err: " << total_error <<"\n";
      break;
    }
    last_error = total_error;

    // A^-1 * b
    A.diagonal().array() += 1e-9;
    Eigen::VectorXd delta = A.ldlt().solve(b);

    // update: [θ; t] = [θ₀; t₀] + [δθ; δt]
    Eigen::Vector3d dtheta = delta.head<3>();
    Eigen::Vector3d dt = delta.tail<3>();

    Eigen::Matrix3d dR = Eigen::Matrix3d::Identity();
    if (dtheta.norm() > 1e-8) {
      dR = Eigen::AngleAxisd(dtheta.norm(), dtheta.normalized()).toRotationMatrix();
    }
    
    R = dR * R;
    t = t + dt;
  }
}

void solvePnP(const std::vector<cv::Point3f> &pts_3,
              const std::vector<cv::Point2f> &pts_2, const cv::Mat &K,
              Eigen::Matrix3d &R, Eigen::Vector3d &T) {
  size_t n = pts_3.size();
  if (n < 6 || pts_3.size() != pts_2.size()) {
    std::cerr << "Not enough points or mismatch in number of 2D and 3D points."
              << std::endl;
    return;
  }

  // TODO: Implement the DLT PnP Algorithm

  // 1. Construct the matrix A
  ;

  Eigen::MatrixXd A(2 * n, 9);
  // set the points to matrix A
  for (size_t i = 0; i < n; i++) {
    double X = pts_3[i].x;
    double Y = pts_3[i].y;

    double u = pts_2[i].x;
    double v = pts_2[i].y;

    // the row index for the current point
    int row1 = 2 * i;
    int row2 = 2 * i + 1;

    A(row1, 0) = X;
    A(row1, 1) = Y;
    A(row1, 2) = 1.0;
    A(row1, 3) = 0.0;
    A(row1, 4) = 0.0;
    A(row1, 5) = 0.0;
    A(row1, 6) = -u * X;
    A(row1, 7) = -u * Y;
    A(row1, 8) = -u;

    A(row2, 0) = 0.0;
    A(row2, 1) = 0.0;
    A(row2, 2) = 0.0;
    A(row2, 3) = X;
    A(row2, 4) = Y;
    A(row2, 5) = 1.0;
    A(row2, 6) = -v * X;
    A(row2, 7) = -v * Y;
    A(row2, 8) = -v;
  }

  // 2. build K_eigen
  Eigen::Matrix3d K_eigen;
  K_eigen << K.at<double>(0, 0), K.at<double>(0, 1),
      K.at<double>(0, 2), K.at<double>(1, 0),
      K.at<double>(1, 1), K.at<double>(1, 2),
      K.at<double>(2, 0), K.at<double>(2, 1),
      K.at<double>(2, 2);

  Eigen::Matrix3d K_inv = K_eigen.inverse();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  Eigen::VectorXd x = svd.matrixV().col(8);

  // 3. Extract R and T
  Eigen::Matrix3d H_raw;
  H_raw << x(0), x(1), x(2), x(3), x(4), x(5), x(6), x(7), x(8);
  Eigen::Matrix3d H = K_inv * H_raw;
  Eigen::Vector3d h1 = H.col(0);
  Eigen::Vector3d h2 = H.col(1);
  Eigen::Vector3d h3 = H.col(2);
  Eigen::Vector3d h1_cross_h2 = h1.cross(h2);

  Eigen::Matrix3d H_rot;
  H_rot << h1, h2, h1_cross_h2;

  Eigen::JacobiSVD<Eigen::Matrix3d> svd_R(H_rot, Eigen::ComputeFullU |
                                                     Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd_R.matrixU();
  Eigen::Matrix3d V = svd_R.matrixV();

  R = U * V.transpose();
  T = h3 / h1.norm();

  // 4. Enforce SO(3) constraint on R using SVD
  if (R.determinant() < 0) {
    R.col(2) *= -1;
  }

  // Depth check
  if (T.z() < 0) {
    R.col(0) *= -1;
    R.col(1) *= -1;
    T = -T;
  }

  // Refine with Gauss-Newton (pass Eigen K)
  gaussNewtonPnP(pts_3, pts_2, K_eigen, R, T);
}
