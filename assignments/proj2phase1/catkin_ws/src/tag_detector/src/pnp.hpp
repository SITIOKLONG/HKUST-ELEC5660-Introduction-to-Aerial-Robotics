#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

void solvePnP(
    const std::vector<cv::Point3f> &pts_3,
    const std::vector<cv::Point2f> &pts_2,
    const cv::Mat &K,
    Eigen::Matrix3d &R,
    Eigen::Vector3d &T
) {
    // Minimum number of points required by DLT is 6
    size_t n = pts_3.size();
    if (n < 6 || pts_3.size() != pts_2.size()) {
        std::cerr << "Not enough points or mismatch in number of 2D and 3D points." << std::endl;
        return;
    }

    // TODO: Implement the DLT PnP Algorithm
    
    // 1. Construct the matrix A
;

    Eigen::MatrixXd A(2 * n, 9);
    // set the points to matrix A
    for (size_t i = 0; i < n; i++){
        double X = pts_3[i].x;
        double Y = pts_3[i].y;
        
        double u = pts_2[i].x;
        double v = pts_2[i].y;
        
        // the row index for the current point
        int row1 = 2 * i;
        int row2 = 2 * i + 1;
        
        A(row1, 0) = X;   A(row1, 1) = Y;   A(row1, 2) = 1.0; 
        A(row1, 3) = 0.0; A(row1, 4) = 0.0; A(row1, 5) = 0.0; 
        A(row1, 6) = -u * X; A(row1, 7) = -u * Y; A(row1, 8) = -u;

        A(row2, 0) = 0.0; A(row2, 1) = 0.0; A(row2, 2) = 0.0; 
        A(row2, 3) = X;   A(row2, 4) = Y;   A(row2, 5) = 1.0; 
        A(row2, 6) = -v * X; A(row2, 7) = -v * Y; A(row2, 8) = -v;     
    }
    
    // 2. Solve Ax = 0 using SVD
    Eigen::Matrix3d K_eigen;
    K_eigen << K.at<double>(0,0), K.at<double>(0,1), K.at<double>(0,2),
               K.at<double>(1,0), K.at<double>(1,1), K.at<double>(1,2),
               K.at<double>(2,0), K.at<double>(2,1), K.at<double>(2,2);
    Eigen::Matrix3d K_inv = K_eigen.inverse();
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd x = svd.matrixV().col(8);
    
    // 3. Extract R and T
    Eigen::Matrix3d H_raw;
    H_raw << x(0), x(1), x(2),
             x(3), x(4), x(5),
             x(6), x(7), x(8);
    Eigen::Matrix3d H = K_inv * H_raw;
    Eigen::Vector3d h1 = H.col(0);
    Eigen::Vector3d h2 = H.col(1);
    Eigen::Vector3d h3 = H.col(2);
    Eigen::Vector3d h1_cross_h2 = h1.cross(h2);

    Eigen::Matrix3d H_rot;
    H_rot << h1, h2, h1_cross_h2;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd_R(H_rot, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd_R.matrixU();
    Eigen::Matrix3d V = svd_R.matrixV();
    
    R = U * V.transpose(); 
    T = h3 / h1.norm();   

    // 4. Enforce SO(3) constraint on R using SVD
    if (R.determinant() < 0) {
        Eigen::Matrix3d diag;
        diag.setIdentity();
        diag(2, 2) = -1.0;
        R = U * diag * V.transpose();
    };
    if (T.z() < 0) {
        R = -R;
        T = -T;
    }
}
