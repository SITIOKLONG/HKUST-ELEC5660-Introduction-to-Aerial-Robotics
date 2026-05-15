#ifndef _EKF_IMU_VISION_H_
#define _EKF_IMU_VISION_H_

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Range.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ekf_model.h>
#include <geometry_msgs/PointStamped.h>
#include <vector>
#include <deque>
#include <set>

#include <stereo_vo/relative_pose.h>

using namespace std;

namespace ekf_imu_vision {

  enum {imu, pnp, vo, keyframe};

struct AugState {
  int type;                       // the type of the state: imu, pnp, vo, keyframe
  ros::Time time_stamp;           // time stamp
  ros::Time key_frame_time_stamp; // active keyframe time of the propagated state
  ros::Time measurement_key_frame_time_stamp; // VO measurement's key_stamp

  Vec(21) mean;                   // estimated mean of the state
                                  //  1: x0:2 ~ x, y, z """
                                  //  2: x3:5 ~ phi theta psi """
                                  //  3: x6:8 ~ vx vy vz """
                                  //  4: x9:11 ~ bgx bgy bgz """
                                  //  5: x12:14 ~  bax bay baz """
                                  //  6: x15:17 ~ keyframe x, y, z """
                                  //  7: x18:20 ~ keyframe phi theta psi """

  Mat(21, 21) covariance;         // covariance of the state
  Vec6 ut;                        // input or measurement of the state
};

class EKFImuVision {
private:
  /* ============================== EKF base ============================== */

  deque<AugState> aug_state_hist_;  // the queue storing the necessary state history in chronological order
  int latest_idx[4];                // the index of the latest state of the 4 type in the queue
  std::set<ros::Time> keyframe_times_;
  Vec6 current_imu_ut;
  
  Vec21 state_;                     // current estimated state
  Mat21x21 cov_;
  bool filter_initialized_;         // whether the filter is initialized
  ros::Time init_time_;             // time when filter was initialized
  ros::Time current_time_;          // current time
  ros::Time current_keyframe_time_; // timestamp of the active VO keyframe (0 if none)
  Vec6 current_keyframe_pose_;      // [pos; euler] of the active keyframe
  Mat6x6 current_keyframe_cov_;     // 6x6 covariance of the active keyframe pose

  // Output smoother: rate-limited copy of the EKF state, used only for publishing.
  bool smoothed_initialized_;
  Vec3 smoothed_pos_;
  Vec3 smoothed_euler_;
  ros::Time last_publish_time_;

  /* ---------- parameter ---------- */
  Mat12x12        Qt_;         // imu
  Mat6x6          Rt1_;        // pnp
  Mat6x6          Rt2_;        // stereo vo

  Mat(21, 15) M_a_;
  Mat(15, 21) M_r_;

  bool initFilter();
  bool initUsingPnP(deque<AugState>::iterator start_it);

  void predictIMU(AugState& cur_state, AugState& prev_state, Vec6 ut);
  void updatePnP(AugState& cur_state, AugState& prev_state);
  void updateVO(AugState& cur_state, AugState& prev_state);
  void changeAugmentedState(AugState& state);

  bool processNewState(AugState& new_state, bool change_keyframe);
  deque<AugState>::iterator insertNewState(AugState& new_state);
  void repropagate(deque<AugState>::iterator& new_input_it,  bool& during_init);
  void removeOldState();

  Vec3 rotation2Euler(const Mat3x3& R);
  Vec3 quaternion2Euler(double w, double x, double y, double z);

  /* ---------- flag ---------- */
  bool            init_;

  /* ============================== ros interface ============================== */
  ros::NodeHandle node_;
  ros::Subscriber imu_sub_, pnp_sub_, stereo_sub_;
  ros::Publisher  fuse_odom_pub_, path_pub_;
  nav_msgs::Path  path_;

  void imuCallback(const sensor_msgs::ImuConstPtr& imu_msg);
  void PnPCallback(const nav_msgs::OdometryConstPtr& msg);
  void stereoVOCallback(const stereo_vo::relative_poseConstPtr& msg);
  void publishFusedOdom();

public:
  EKFImuVision(/* args */);
  ~EKFImuVision();

  void init(ros::NodeHandle& nh);
};

// EKFImuVision::
}  // namespace ekf_imu_vision

#endif
