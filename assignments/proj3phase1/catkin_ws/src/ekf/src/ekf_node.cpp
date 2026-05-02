#include <iostream>
#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Range.h>
#include <nav_msgs/Odometry.h>
#include <Eigen/Eigen>

using namespace std;
using namespace Eigen;
ros::Publisher odom_pub;
MatrixXd Q = MatrixXd::Identity(12, 12);
MatrixXd Rt = MatrixXd::Identity(6,6);
VectorXd x = VectorXd::Zero(15);
MatrixXd P = MatrixXd::Identity(15, 15);
ros::Time prev_time(0);
const Vector3d g(0, 0, -9.81);


Matrix3d euler_to_rot(const Vector3d& rpy) {
    const double r = rpy(0);
    const double p = rpy(1);
    const double y = rpy(2);

    const double cr = cos(r);
    const double sr = sin(r);
    const double cp = cos(p);
    const double sp = sin(p);
    const double cy = cos(y);
    const double sy = sin(y);

    Matrix3d R;
    R << cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
         sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
         -sp,     cp * sr,              cp * cr;
    return R;
}

Vector3d quat_to_euler(const Quaterniond& q_in) {
    Quaterniond q = q_in.normalized();

    const double qw = q.w();
    const double qx = q.x();
    const double qy = q.y();
    const double qz = q.z();

    const double sinr_cosp = 2.0 * (qw * qx + qy * qz);
    const double cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy);
    const double roll = atan2(sinr_cosp, cosr_cosp);

    double sinp = 2.0 * (qw * qy - qz * qx);
    if (sinp > 1.0) sinp = 1.0;
    if (sinp < -1.0) sinp = -1.0;
    const double pitch = asin(sinp);

    const double siny_cosp = 2.0 * (qw * qz + qx * qy);
    const double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    const double yaw = atan2(siny_cosp, cosy_cosp);

    return Vector3d(roll, pitch, yaw);
}

Quaterniond euler_to_quat(const Vector3d& rpy) {
    const double r = rpy(0);
    const double p = rpy(1);
    const double y = rpy(2);

    const double cr = cos(r * 0.5);
    const double sr = sin(r * 0.5);
    const double cp = cos(p * 0.5);
    const double sp = sin(p * 0.5);
    const double cy = cos(y * 0.5);
    const double sy = sin(y * 0.5);

    Quaterniond q;
    q.w() = cr * cp * cy + sr * sp * sy;
    q.x() = sr * cp * cy - cr * sp * sy;
    q.y() = cr * sp * cy + sr * cp * sy;
    q.z() = cr * cp * sy - sr * sp * cy;
    return q.normalized();
}

Matrix3d dR_a_drpy(const Vector3d& rpy, const Vector3d& a) {
    double r = rpy(0), p = rpy(1), y = rpy(2);
    // 构造 Rz, Ry, Rx 及其导数
    Matrix3d Rx, Ry, Rz;
    Rx << 1, 0, 0,
          0, cos(r), -sin(r),
          0, sin(r),  cos(r);
    Ry << cos(p), 0, sin(p),
          0,      1, 0,
         -sin(p), 0, cos(p);
    Rz << cos(y), -sin(y), 0,
          sin(y),  cos(y), 0,
          0,       0,      1;
    
    Matrix3d dRx, dRy, dRz;
    dRx << 0, 0,       0,
           0, -sin(r), -cos(r),
           0,  cos(r), -sin(r);
    dRy << -sin(p), 0, cos(p),
            0,      0, 0,
           -cos(p), 0, -sin(p);
    dRz << -sin(y), -cos(y), 0,
            cos(y), -sin(y), 0,
            0,       0,      0;
    
    Matrix3d J;
    J.col(0) = Rz * Ry * dRx * a;
    J.col(1) = Rz * dRy * Rx * a;
    J.col(2) = dRz * Ry * Rx * a;
    return J;
}

void imu_callback(const sensor_msgs::Imu::ConstPtr &msg)
{
    // Implement EKF prediction here:
    ros::Time now = msg->header.stamp;

    // 提取当前状态片段
    Vector3d rpy = x.segment<3>(3);
    Vector3d v   = x.segment<3>(6);
    Vector3d bg  = x.segment<3>(9);
    Vector3d ba  = x.segment<3>(12);

    Vector3d omega_m(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
    Vector3d a_m(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
    Vector3d omega = omega_m - bg;
    Vector3d a = a_m - ba;

    Matrix3d R_body2world = euler_to_rot(rpy);  // 需要自己实现
    Matrix3d J_v_rpy = dR_a_drpy(rpy, a);   // 计算 ∂(R*a)/∂rpy  (3x3 矩阵)

    // 构造 F (15x15) 初始化为零，再填非零块
    Matrix<double, 15, 15> F = Matrix<double, 15, 15>::Zero();
    F.block<3,3>(0, 6) = Matrix3d::Identity();   // dp/dv
    F.block<3,3>(3, 9) = -Matrix3d::Identity();  // drpy/dbg  (用简化G=I)
    F.block<3,3>(6, 3) = J_v_rpy;               // dv/drpy
    F.block<3,3>(6, 12) = -R_body2world;               // dv/dba

    Matrix<double, 15, 12> U = Matrix<double, 15, 12>::Zero();
    U.block<3,3>(3, 0) = -Matrix3d::Identity(); // drpy/dn_g
    U.block<3,3>(6, 3) = -R_body2world;         // dv/dn_a
    U.block<3,3>(9, 6) = Matrix3d::Identity(); // dbg/dn_bg
    U.block<3,3>(12,9) = Matrix3d::Identity(); // dba/dn_ba

    double dt = (prev_time.isZero()) ? 0.0 : (now - prev_time).toSec();
    prev_time = now;
    if (dt <= 0.0) return;  // 第一个数据或异常直接跳过

        // 更新位置: p += v*dt
    x.head<3>() += v * dt;

    // 更新姿态: rpy += omega*dt  (注意这里 omega = 陀螺仪测量 - bias)
    x.segment<3>(3) += omega * dt;
    x.segment<3>(3) = x.segment<3>(3).unaryExpr([](double a) {
        return atan2(sin(a), cos(a));
    });

    // 更新速度: v += (R*a + g)*dt
    x.segment<3>(6) += (R_body2world * a + g) * dt;

    // bias 的预测：如果没有其他信息，均值保持不变
    // （随机游走只增加不确定性，不改变均值）

    MatrixXd F_t = MatrixXd::Identity(15,15) + F * dt;
    MatrixXd V_t = U;

    P = F_t * P * F_t.transpose() + V_t * Q * V_t.transpose() * dt; // only 1 *dt
}

// Rotation from the camera frame to the IMU frame.
Eigen::Matrix3d Rcam;
bool initialized = false;
void odom_callback(const nav_msgs::Odometry::ConstPtr &msg)
{
    // Implement EKF correction here:
    // Fixed camera/IMU extrinsic used for the conversion:
    // camera origin expressed in the IMU frame = (0.05, 0.05, 0.0)
    // camera-to-IMU rotation = Quaternion(0, 1, 0, 0) in (w, x, y, z) order
    // equivalent rotation matrix:
    //     [ 1,  0,  0]
    //     [ 0, -1,  0]
    //     [ 0,  0, -1]

    // 位置
    Vector3d p_cam_w(msg->pose.pose.position.x,
                    msg->pose.pose.position.y,
                    msg->pose.pose.position.z);
    // quat
    Quaterniond q_cam_w(msg->pose.pose.orientation.w,
                        msg->pose.pose.orientation.x,
                        msg->pose.pose.orientation.y,
                        msg->pose.pose.orientation.z);

    Vector3d p_cam_in_imu(0.05, 0.05, 0.0);

    Matrix3d R_cam_w = q_cam_w.toRotationMatrix();
    Matrix3d R_imu_w = R_cam_w;     // no * Rcam
    // IMU 在世界系的位置：p_IMU_w = p_cam_w - R_imu_w * p_cam_in_imu
    Vector3d p_IMU_w = p_cam_w - R_imu_w * p_cam_in_imu;
    Vector3d rpy_imu_w = quat_to_euler(Quaterniond(R_imu_w));

    Matrix<double, 6, 1> z;
    z << p_IMU_w, rpy_imu_w;

    if (!initialized) {
        x.head<6>() = z;   // 用视觉测量初始化位置和姿态

        // 位置协方差 = 视觉测量噪声
        P.block<3,3>(0,0) = Rt.block<3,3>(0,0);
        // 姿态协方差 = 视觉测量噪声
        P.block<3,3>(3,3) = Rt.block<3,3>(3,3);
        // 速度：完全未知，设较大值（例如标准差 2 m/s）
        P.block<3,3>(6,6) = Matrix3d::Identity() * 4.0;    // 方差 4.0
        // 陀螺偏置：初始大致知道（逼近0），但仍留一些不确定
        P.block<3,3>(9,9) = Matrix3d::Identity() * 0.01;   // 标准差 0.1 rad/s
        // 加速度偏置：类似，设较大方差
        P.block<3,3>(12,12) = Matrix3d::Identity() * 1.0;  // 标准差 1 m/s²

        initialized = true;
        return;
    }

    Matrix<double, 6, 15> C = Matrix<double, 6, 15>::Zero();
    C.block<3,3>(0,0) = Matrix3d::Identity();   // 观测位置
    C.block<3,3>(3,3) = Matrix3d::Identity();   // 观测姿态

    Matrix<double, 6, 1> z_pred = C * x;  // 预测的观测值
    Matrix<double, 6, 1> y = z - z_pred;  // 新息
    y(5) = atan2(sin(y(5)), cos(y(5)));
    // ROS_INFO_STREAM_THROTTLE(1.0, "innovation y: " << y.transpose());

    Matrix<double, 6, 6> S = C * P * C.transpose() + Rt;  // 新息协方差
    const double maha = y.transpose() * S.inverse() * y;
    if (maha > 16.81) {      // x^2(6) = 16.81
        ROS_WARN_STREAM_THROTTLE(1.0, "Innovation gated (maha=" << maha << ")");
        return;
    }
    Matrix<double, 15, 6> K = P * C.transpose() * S.inverse();

    x = x + K * y;
    P = (Matrix<double, 15, 15>::Identity() - K * C) * P;
    x(3) = atan2(sin(x(3)), cos(x(3)));
    x(4) = atan2(sin(x(4)), cos(x(4)));
    x(5) = atan2(sin(x(5)), cos(x(5)));
    // ROS_INFO_STREAM_THROTTLE(
    //     1.0,
    //     "P diag pos/att: "
    //         << P(0, 0) << ", " << P(1, 1) << ", " << P(2, 2)
    //         << ", " << P(3, 3) << ", " << P(4, 4) << ", " << P(5, 5));
    ROS_INFO_STREAM_THROTTLE(
        1.0,
        "P/Rt diag pos/att: "
            << P(0, 0) / Rt(0, 0) << ", " << P(1, 1) / Rt(1, 1) << ", "
            << P(2, 2) / Rt(2, 2) << ", " << P(3, 3) / Rt(3, 3) << ", "
            << P(4, 4) / Rt(4, 4) << ", " << P(5, 5) / Rt(5, 5));

    nav_msgs::Odometry ekf_odom;
    ekf_odom.header.stamp = msg->header.stamp;
    ekf_odom.header.frame_id = "world";
    ekf_odom.child_frame_id = "imu";

    ekf_odom.pose.pose.position.x = x(0);
    ekf_odom.pose.pose.position.y = x(1);
    ekf_odom.pose.pose.position.z = x(2);

    Quaterniond q_out = euler_to_quat(x.segment<3>(3));
    ekf_odom.pose.pose.orientation.w = q_out.w();
    ekf_odom.pose.pose.orientation.x = q_out.x();
    ekf_odom.pose.pose.orientation.y = q_out.y();
    ekf_odom.pose.pose.orientation.z = q_out.z();

    odom_pub.publish(ekf_odom);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ekf");
    ros::NodeHandle n("~");
    ros::Subscriber s1 = n.subscribe("imu", 1000, imu_callback);
    ros::Subscriber s2 = n.subscribe("tag_odom", 1000, odom_callback);
    odom_pub = n.advertise<nav_msgs::Odometry>("ekf_odom", 100);
    Rcam = Quaterniond(0, 1, 0, 0).toRotationMatrix();
    cout << "R_cam" << endl << Rcam << endl;
    // Q: process noise covariance. Rt: visual measurement noise covariance.
    // You should tune these parameters for a stable filter.

    // 初始协方差 P（假设视觉第一次初始化）
    P.setIdentity();
    P.diagonal() << 0.0001, 0.0001, 0.001,   // 位置 (同 Rt)
                    0.0001,0.0001,0.0001,     // 姿态 (≈ 0.5°)
                    1.0, 1.0, 1.0,            // 速度
                    0.01, 0.01, 0.01,        // gyro bias
                    0.1,  0.1,  0.1;         // acc bias

    // 过程噪声 Q (连续谱密度)
    Q.setIdentity();
    Q.diagonal() << 2.0, 2.0, 2.0,      // gyro noise (rad/s)²/Hz  （原 0.01 → 0.1）
                    5.0, 5.0, 5.0,      // acc noise (m/s²)²/Hz   （原 0.1 → 1.0）
                    5e-3, 5e-3, 5e-3,   // gyro bias random walk （原 1e-5 → 1e-3）
                    5e-2, 5e-2, 5e-2;   // acc bias random walk  （原 1e-4 → 1e-2）

    // 测量噪声 Rt (视觉)
    Rt.setIdentity();
    Rt.diagonal() << 0.002, 0.002, 0.025,   // 位置 1cm,1cm,3cm
                    0.004, 0.004, 0.025;   // 姿态 0.8°,0.8°,1.15°

    ros::spin();
}
