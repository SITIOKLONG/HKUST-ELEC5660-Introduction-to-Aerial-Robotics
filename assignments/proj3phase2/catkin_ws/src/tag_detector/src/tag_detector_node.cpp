#include <iostream>
#include <string>
#include <vector>

#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Odometry.h>
#include <aruco/aruco.h>
#include <aruco/cvdrawingutils.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <Eigen/SVD>
//EIgen SVD libnary, may help you solve SVD
//JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);

#include "pnp.hpp"

using namespace cv;
using namespace aruco;
using namespace Eigen;
using std::string;
using std::vector;

//global varialbles for aruco detector
aruco::CameraParameters CamParam;
MarkerDetector MDetector;
vector<Marker> Markers;
float MarkerSize = 0.20 / 1.5 * 1.524;
float MarkerWithMargin = MarkerSize * 1.2;
MarkerMap TheMarkerMap;
ros::Publisher pub_odom_yourwork;
ros::Publisher pub_odom_ref;
cv::Mat K, D;

double RMSE_sum = 0;
double RMSE_num = 0;


void calculateRMSE(const vector<cv::Point3f> &pts_3, const vector<cv::Point2f> &pts_2, const cv::Mat R, const cv::Mat t)
{
    double rmse = 0;
    vector<cv::Point2f> un_pts_2;
    cv::undistortPoints(pts_2, un_pts_2, K, D);
    for (unsigned int i = 0; i < pts_3.size(); i++)
    {
        cv::Mat p_mat(3, 1, CV_64FC1);
        p_mat.at<double>(0, 0) = pts_3[i].x;
        p_mat.at<double>(1, 0) = pts_3[i].y;
        p_mat.at<double>(2, 0) = pts_3[i].z;
        cv::Mat p = (R * p_mat + t);
        rmse += pow(un_pts_2[i].x - p.at<double>(0) / p.at<double>(2), 2) + pow(un_pts_2[i].y - p.at<double>(1) / p.at<double>(2), 2);
    }
    rmse = sqrt(rmse / pts_3.size());
    RMSE_sum += rmse;
    RMSE_num += 1;
    double avg_rmse = RMSE_sum / RMSE_num;
    ROS_INFO("RMSE: %f, Avg RMSE: %f", rmse, avg_rmse);
}

// test function, can be used to verify your estimation
void calculateReprojectionError(const vector<cv::Point3f> &pts_3, const vector<cv::Point2f> &pts_2, const cv::Mat R, const cv::Mat t)
{
    puts("calculateReprojectionError begins");
    vector<cv::Point2f> un_pts_2;
    cv::undistortPoints(pts_2, un_pts_2, K, D);
    for (unsigned int i = 0; i < pts_3.size(); i++)
    {
        cv::Mat p_mat(3, 1, CV_64FC1);
        p_mat.at<double>(0, 0) = pts_3[i].x;
        p_mat.at<double>(1, 0) = pts_3[i].y;
        p_mat.at<double>(2, 0) = pts_3[i].z;
        cv::Mat p = (R * p_mat + t);
        printf("(%f, %f, %f) -> (%f, %f) and (%f, %f)\n",
               pts_3[i].x, pts_3[i].y, pts_3[i].z,
               un_pts_2[i].x, un_pts_2[i].y,
               p.at<double>(0) / p.at<double>(2), p.at<double>(1) / p.at<double>(2));
    }
    puts("calculateReprojectionError ends");
}

// Convert solvePnP output (tag-to-camera, R_ct/t_ct) to body-in-world.
// The marker map coordinates are in the tag-board frame. Project 2/3 define
// world from tag as R_wt and the downward camera extrinsic as T_bcd.
static inline void cwToWb(const Eigen::Matrix3d& R_cw, const Eigen::Vector3d& t_cw,
                          Eigen::Matrix3d& R_wb, Eigen::Vector3d& t_wb) {
    static const Eigen::Matrix3d R_wt = (Eigen::Matrix3d() << 0, 1, 0,
                                                              1, 0, 0,
                                                              0, 0,-1).finished();
    static const Eigen::Matrix3d R_bc = (Eigen::Matrix3d() << 1, 0, 0,
                                                              0,-1, 0,
                                                              0, 0,-1).finished();

    Eigen::Matrix3d R_tb = R_cw.transpose() * R_bc;
    Eigen::Vector3d t_tc = -R_cw.transpose() * t_cw;

    R_wb = R_wt * R_tb;
    t_wb = R_wt * t_tc;
}

// the main function you need to work with
// pts_id: id of each point
// pts_3: 3D position (x, y, z) in world frame
// pts_2: 2D position (u, v) in image frame
void process(const vector<int> &pts_id, const vector<cv::Point3f> &pts_3, const vector<cv::Point2f> &pts_2, const ros::Time& frame_time)
{
    //version 1, as reference
    cv::Mat r, rvec, t;
    cv::solvePnP(pts_3, pts_2, K, D, rvec, t);
    cv::Rodrigues(rvec, r);
    Matrix3d R_ref;
    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
        {
            R_ref(i,j) = r.at<double>(i, j);
        }
    Vector3d t_ref(t.at<double>(0,0), t.at<double>(1,0), t.at<double>(2,0));
    Matrix3d R_wb_ref;
    Vector3d t_wb_ref;
    cwToWb(R_ref, t_ref, R_wb_ref, t_wb_ref);
    Quaterniond Q_ref(R_wb_ref);
    nav_msgs::Odometry odom_ref;
    odom_ref.header.stamp = frame_time;
    odom_ref.header.frame_id = "world";
    odom_ref.pose.pose.position.x = t_wb_ref(0);
    odom_ref.pose.pose.position.y = t_wb_ref(1);
    odom_ref.pose.pose.position.z = t_wb_ref(2);
    odom_ref.pose.pose.orientation.w = Q_ref.w();
    odom_ref.pose.pose.orientation.x = Q_ref.x();
    odom_ref.pose.pose.orientation.y = Q_ref.y();
    odom_ref.pose.pose.orientation.z = Q_ref.z();
    pub_odom_ref.publish(odom_ref);

    // version 2, your work
    Matrix3d R;
    Vector3d T;
    R.setIdentity();
    T.setZero();
    vector<cv::Point2f> un_pts_2;
    cv::undistortPoints(pts_2, un_pts_2, K, D);


    solvePnP(pts_3, pts_2, K, R, T);
    cv::Mat R_mat(3, 3, CV_64FC1);
    cv::Mat T_mat(3, 1, CV_64FC1);
    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
        {
            R_mat.at<double>(i, j) = R(i, j);
        }
    for(int i=0;i<3;i++)
    {
        T_mat.at<double>(i, 0) = T(i);
    }
    // RMSE is reported in the camera frame, so use the un-inverted (R, T).
    calculateRMSE(pts_3, pts_2, R_mat, T_mat);

    Matrix3d R_wb;
    Vector3d t_wb;
    cwToWb(R, T, R_wb, t_wb);
    Quaterniond Q_yourwork(R_wb);
    nav_msgs::Odometry odom_yourwork;
    odom_yourwork.header.stamp = frame_time;
    odom_yourwork.header.frame_id = "world";
    odom_yourwork.pose.pose.position.x = t_wb(0);
    odom_yourwork.pose.pose.position.y = t_wb(1);
    odom_yourwork.pose.pose.position.z = t_wb(2);
    odom_yourwork.pose.pose.orientation.w = Q_yourwork.w();
    odom_yourwork.pose.pose.orientation.x = Q_yourwork.x();
    odom_yourwork.pose.pose.orientation.y = Q_yourwork.y();
    odom_yourwork.pose.pose.orientation.z = Q_yourwork.z();
    pub_odom_yourwork.publish(odom_yourwork);
}

cv::Point3f getPositionFromIndex(int idx, int nth)
{
    int idx_x = idx % 6, idx_y = idx / 6;
    double p_x = idx_x * MarkerWithMargin - (3 + 2.5 * 0.2) * MarkerSize;
    double p_y = idx_y * MarkerWithMargin - (12 + 11.5 * 0.2) * MarkerSize;
    return cv::Point3f(p_x + (nth == 1 || nth == 2) * MarkerSize,
                       p_y + (nth == 2 || nth == 3) * MarkerSize, 0.0);
}

void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    double t = clock();
    cv_bridge::CvImagePtr bridge_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
    //由于 aarch64 下 libaruco.so 的 calculateExtrinsics 调用 cv::solvePnP 必定引发段错误(ABI冲突)
    //设置 MarkerSize 为 -1 才是设计上安全跳过错误的自带 PnP 函数的做法。
    MDetector.detect(bridge_ptr->image, Markers, CamParam, -1, false);
    ROS_DEBUG("time cost: %f\n", (clock() - t) / CLOCKS_PER_SEC);

    vector<int> pts_id;
    vector<cv::Point3f> pts_3;
    vector<cv::Point2f> pts_2;
    for (unsigned int i = 0; i < Markers.size(); i++)
    {
        int idx = TheMarkerMap.getIndexOfMarkerId(Markers[i].id);

        char str[100];
        sprintf(str, "%d", idx);
        cv::putText(bridge_ptr->image, str, Markers[i].getCenter(), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(-1));
        for (unsigned int j = 0; j < 4; j++)
        {
            sprintf(str, "%d", j);
            cv::putText(bridge_ptr->image, str, Markers[i][j], cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(-1));
        }

        for (unsigned int j = 0; j < 4; j++)
        {
            pts_id.push_back(Markers[i].id * 4 + j);
            pts_3.push_back(getPositionFromIndex(idx, j));
            pts_2.push_back(Markers[i][j]);
        }
    }

    //begin your function
    if (pts_id.size() > 5)
        process(pts_id, pts_3, pts_2, img_msg->header.stamp);

    // Only render the debug window if a display is available — otherwise the
    // OpenCV / Qt calls crash the whole node when running headless (e.g. over
    // SSH without X forwarding).
    if (std::getenv("DISPLAY") != nullptr) {
        cv::imshow("in", bridge_ptr->image);
        cv::waitKey(10);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "tag_detector");
    ros::NodeHandle n("~");

    ros::Subscriber sub_img = n.subscribe("image_raw", 100, img_callback);
    pub_odom_yourwork = n.advertise<nav_msgs::Odometry>("odom_yourwork",10);
    pub_odom_ref = n.advertise<nav_msgs::Odometry>("odom_ref",10);
    //init aruco detector
    string cam_cal, board_config;
    n.getParam("cam_cal_file", cam_cal);
    n.getParam("board_config_file", board_config);
    CamParam.readFromXMLFile(cam_cal);
    TheMarkerMap.readFromFile(board_config);
    if (!TheMarkerMap.getDictionary().empty()) {
        MDetector.setDictionary(TheMarkerMap.getDictionary());
    } else {
        MDetector.setDictionary("ARUCO");
    }

    //init intrinsic parameters
    cv::FileStorage param_reader(cam_cal, cv::FileStorage::READ);
    param_reader["camera_matrix"] >> K;
    param_reader["distortion_coefficients"] >> D;

    //init window for visualization (only if a display is available)
    if (std::getenv("DISPLAY") != nullptr) {
        cv::namedWindow("in", 1);
    }

    ros::spin();
}
