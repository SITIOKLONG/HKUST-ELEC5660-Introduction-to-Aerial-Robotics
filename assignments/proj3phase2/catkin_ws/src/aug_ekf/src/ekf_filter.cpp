
#include <ekf_filter.h>
namespace ekf_imu_vision {

  static double wrapAngle(double angle) {
    return atan2(sin(angle), cos(angle));
  }

  Mat3x3 euler2Rotation(const Vec3& euler) {
    double phi = euler(0), theta = euler(1), psi = euler(2);
    Eigen::Quaterniond q = Eigen::AngleAxisd(psi, Eigen::Vector3d::UnitZ()) *
                          Eigen::AngleAxisd(phi, Eigen::Vector3d::UnitX()) *
                          Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitY());
    return q.toRotationMatrix();
  }

  Mat3x3 dR_a_drpy(const Vec3& rpy, const Vec3& a) {
    double r = rpy(0), p = rpy(1), y = rpy(2);
    Mat3x3 Rx, Ry, Rz;
    Rx << 1, 0, 0,
      0, cos(r), -sin(r),
      0, sin(r),  cos(r);
    Ry << cos(p), 0, sin(p),
      0,      1, 0,
     -sin(p), 0, cos(p);
    Rz << cos(y), -sin(y), 0,
      sin(y),  cos(y), 0,
      0,       0,      1;

    Mat3x3 dRx, dRy, dRz;
    dRx << 0, 0,       0,
       0, -sin(r), -cos(r),
       0,  cos(r), -sin(r);
    dRy << -sin(p), 0, cos(p),
        0,      0, 0,
       -cos(p), 0, -sin(p);
    dRz << -sin(y), -cos(y), 0,
        cos(y), -sin(y), 0,
        0,       0,      0;

    // For R = Rz(psi) * Rx(phi) * Ry(theta):
    // dR/dphi   = Rz * dRx * Ry
    // dR/dtheta = Rz * Rx * dRy
    // dR/dpsi   = dRz * Rx * Ry
    Mat3x3 J;
    J.col(0) = Rz * dRx * Ry * a;   // d(R*a)/dphi
    J.col(1) = Rz * Rx * dRy * a;   // d(R*a)/dtheta
    J.col(2) = dRz * Rx * Ry * a;   // d(R*a)/dpsi
    return J;
  }

  Mat3x3 eulerRightJacobianInv(const Vec3& q) {
    double phi = q(0), theta = q(1);
    double cp = cos(phi), sp = sin(phi);
    double ct = cos(theta), st = sin(theta);
    double inv_cp = 1.0 / cp; 
    
    Mat3x3 Jr_inv;
    Jr_inv << ct,   0,       st,
              sp * st * inv_cp, 1, -ct * sp * inv_cp,
             -st * inv_cp, 0,  ct * inv_cp;
    return Jr_inv;
  }

EKFImuVision::EKFImuVision(/* args */) {}

EKFImuVision::~EKFImuVision() {}

void EKFImuVision::init(ros::NodeHandle& nh) {
  node_ = nh;

  /* ---------- parameter ---------- */
  Qt_.setZero();
  Rt1_.setZero();
  Rt2_.setZero();

  // addition and removel of augmented state

  // TODO
  // set M_a_ and M_r_
  // M_a_ should add the copied keyframe pose to the 15-state EKF state.
  // M_r_ should remove the copied keyframe pose when a new keyframe is selected.
  M_a_.setZero();
  M_a_.block<15, 15>(0, 0) = Mat15x15::Identity();  // 保留原状态
  M_a_.block<3, 3>(15, 0) = Mat3x3::Identity();  // 复制位置
  M_a_.block<3, 3>(18, 3) = Mat3x3::Identity();  // 复制姿态

  // M_r_: 从21维降维回15维（移除旧关键帧）
  M_r_.setZero();
  M_r_.block<15, 21>(0, 0) << Mat15x15::Identity(), Mat15x6::Zero();

  for (int i = 0; i < 3; i++) {
    /* process noise */
    node_.param("aug_ekf/ng", Qt_(i, i), -1.0);
    node_.param("aug_ekf/na", Qt_(i + 3, i + 3), -1.0);
    node_.param("aug_ekf/nbg", Qt_(i + 6, i + 6), -1.0);
    node_.param("aug_ekf/nba", Qt_(i + 9, i + 9), -1.0);
    node_.param("aug_ekf/pnp_p", Rt1_(i, i), -1.0);
    node_.param("aug_ekf/pnp_q", Rt1_(i + 3, i + 3), -1.0);
    node_.param("aug_ekf/vo_pos", Rt2_(i, i), -1.0);
    node_.param("aug_ekf/vo_rot", Rt2_(i + 3, i + 3), -1.0);
  }

  // Guard against zero/negative measurement noise — the launch scaffold ships
  // with vo_pos=vo_rot=0.000 which causes infinite Kalman gain on VO updates
  // (the state "flies"). Enforce a minimum floor so the filter is always
  // numerically well-conditioned.
  const double MIN_NOISE = 1e-6;
  for (int i = 0; i < 6; ++i) {
    if (Rt1_(i, i) < MIN_NOISE) Rt1_(i, i) = MIN_NOISE;
    if (Rt2_(i, i) < MIN_NOISE) Rt2_(i, i) = MIN_NOISE;
  }
  for (int i = 0; i < 12; ++i) {
    if (Qt_(i, i) < MIN_NOISE) Qt_(i, i) = MIN_NOISE;
  }

  init_ = false;
  filter_initialized_ = false;
  keyframe_times_.clear();
  current_imu_ut.setZero();
  current_keyframe_time_ = ros::Time(0);
  current_keyframe_pose_.setZero();
  current_keyframe_cov_.setIdentity();
  smoothed_initialized_ = false;
  smoothed_pos_.setZero();
  smoothed_euler_.setZero();
  last_publish_time_ = ros::Time(0);
  path_.poses.clear();

  for (int i = 0; i < 4; i++) latest_idx[i] = -1;

  /* ---------- subscribe and publish ---------- */
  imu_sub_ =
      node_.subscribe<sensor_msgs::Imu>("/dji_sdk_1/dji_sdk/imu", 100, &EKFImuVision::imuCallback, this);
  pnp_sub_ = node_.subscribe<nav_msgs::Odometry>("tag_odom", 10, &EKFImuVision::PnPCallback, this);
  // opti_tf_sub_ = node_.subscribe<geometry_msgs::PointStamped>("opti_tf_odom", 10,
  // &EKFImuVision::opticalCallback, this);
  stereo_sub_ = node_.subscribe<stereo_vo::relative_pose>("/vo/Relative_pose", 10,
                                                          &EKFImuVision::stereoVOCallback, this);
  fuse_odom_pub_ = node_.advertise<nav_msgs::Odometry>("ekf_fused_odom", 10);
  path_pub_ = node_.advertise<nav_msgs::Path>("/aug_ekf/Path", 100);

  ros::Duration(0.5).sleep();

  ROS_INFO("Start ekf.");
}

void EKFImuVision::PnPCallback(const nav_msgs::OdometryConstPtr& msg) {
  // Message is body-in-world. In simple mode the bag already provides it that way;
  // in full mode the patched tag_detector converts solvePnP output to body-in-world.
  AugState new_state;
  new_state.time_stamp = msg->header.stamp;
  new_state.key_frame_time_stamp = ros::Time(0);
  new_state.measurement_key_frame_time_stamp = ros::Time(0);
  new_state.type = pnp;

  Eigen::Quaterniond q_wb(msg->pose.pose.orientation.w,
                          msg->pose.pose.orientation.x,
                          msg->pose.pose.orientation.y,
                          msg->pose.pose.orientation.z);
  q_wb.normalize();
  Vec3 p_wb(msg->pose.pose.position.x,
            msg->pose.pose.position.y,
            msg->pose.pose.position.z);
  Vec3 euler = rotation2Euler(q_wb.toRotationMatrix());

  new_state.ut.segment<3>(0) = p_wb;
  new_state.ut.segment<3>(3) = euler;

  new_state.mean.setZero();
  new_state.covariance.setZero();

  if (!processNewState(new_state, false)) {
    return;
  }
}

void EKFImuVision::stereoVOCallback(const stereo_vo::relative_poseConstPtr& msg) {
  // Ignore the bootstrap message with a zero keyframe stamp.
  if (msg->key_stamp.isZero()) return;

  AugState new_state;
  new_state.time_stamp = msg->header.stamp;
  new_state.key_frame_time_stamp = ros::Time(0);
  new_state.measurement_key_frame_time_stamp = msg->key_stamp;
  new_state.type = vo;

  // Relative pose: current body expressed in the keyframe body frame.
  Eigen::Quaterniond q_kb(msg->relative_pose.orientation.w,
                          msg->relative_pose.orientation.x,
                          msg->relative_pose.orientation.y,
                          msg->relative_pose.orientation.z);
  q_kb.normalize();
  Vec3 t_kb(msg->relative_pose.position.x,
            msg->relative_pose.position.y,
            msg->relative_pose.position.z);
  Vec3 euler_rel = rotation2Euler(q_kb.toRotationMatrix());

  new_state.ut.segment<3>(0) = t_kb;
  new_state.ut.segment<3>(3) = euler_rel;
  new_state.mean.setZero();
  new_state.covariance.setZero();

  // Detect a keyframe switch (key_stamp changed since last VO message).
  static ros::Time last_key_stamp(0);
  bool change_keyframe = (last_key_stamp != msg->key_stamp);
  last_key_stamp = msg->key_stamp;

  if (!processNewState(new_state, change_keyframe)) {
    return;
  }
}

void EKFImuVision::imuCallback(const sensor_msgs::ImuConstPtr& imu_msg) {
  // TODO
  // construct a new state using the IMU input and insert the new state into the queue
  AugState new_state;
  new_state.time_stamp = imu_msg->header.stamp;
  new_state.key_frame_time_stamp = ros::Time(0);
  new_state.measurement_key_frame_time_stamp = ros::Time(0);
  new_state.type = imu;

  new_state.ut << imu_msg->angular_velocity.x,
                  imu_msg->angular_velocity.y,
                  imu_msg->angular_velocity.z,
                  imu_msg->linear_acceleration.x,
                  imu_msg->linear_acceleration.y,
                  imu_msg->linear_acceleration.z;

  new_state.mean.setZero();
  new_state.covariance.setIdentity();

  if (!processNewState(new_state, false)) {
    return;
  }
}

void EKFImuVision::predictIMU(AugState& cur_state, AugState& prev_state, Vec6 ut) {
  double dt = (cur_state.time_stamp - prev_state.time_stamp).toSec();
  if (dt <= 0.0) return;

  Vec15 x_prev = prev_state.mean.head<15>();
  Mat15x15 P_prev = prev_state.covariance.topLeftCorner<15, 15>();

  Vec3 rpy = x_prev.segment<3>(3);
  Vec3 v = x_prev.segment<3>(6);
  Vec3 bg = x_prev.segment<3>(9);
  Vec3 ba = x_prev.segment<3>(12);

  Vec3 omega_m = ut.head<3>();
  Vec3 a_m = ut.tail<3>();
  Vec3 omega = omega_m - bg;
  Vec3 a = a_m - ba;

  Mat3x3 Jr_inv = eulerRightJacobianInv(rpy);
  Vec12 n0 = Vec12::Zero();
  Mat15x15 F = jacobiFx(x_prev, ut, n0);
  Mat15x12 U = jacobiFn(x_prev, ut, n0);

  // Use midpoint rotation for better integration accuracy.
  // 1) propagate attitude (forward Euler is OK for small dt at high IMU rate).
  // 2) compute acc_world at midpoint attitude to get a better velocity update.
  // 3) trapezoidal position update captures the 0.5 * a * dt^2 term that
  //    forward Euler (p += v*dt) misses -- saves ~10cm of drift over 7.6s
  //    given typical drone accelerations.
  Vec3 rpy_dot = Jr_inv * omega;
  Vec3 rpy_new = rpy + rpy_dot * dt;
  rpy_new(0) = atan2(sin(rpy_new(0)), cos(rpy_new(0)));
  rpy_new(1) = atan2(sin(rpy_new(1)), cos(rpy_new(1)));
  rpy_new(2) = atan2(sin(rpy_new(2)), cos(rpy_new(2)));

  // Midpoint rotation for the acc rotation:
  Vec3 rpy_mid = rpy + 0.5 * rpy_dot * dt;
  Mat3x3 R_mid = euler2Rotation(rpy_mid);
  Vec3 acc_world_mid = R_mid * a + Vec3(0, 0, -9.81);

  Vec3 v_new = v + acc_world_mid * dt;
  Vec3 p_new = x_prev.head<3>() + v * dt + 0.5 * acc_world_mid * dt * dt;

  Vec15 x_pred;
  x_pred << p_new, rpy_new, v_new, bg, ba;

  Mat15x15 F_t = Mat15x15::Identity() + F * dt;
  Mat15x12 V_t = U * dt;
  Mat15x15 P_pred = F_t * P_prev * F_t.transpose() + V_t * Qt_ * V_t.transpose();

  cur_state.mean.head<15>() = x_pred;
  cur_state.mean.tail<6>() = prev_state.mean.tail<6>();
  cur_state.covariance.topLeftCorner<15,15>() = P_pred;
  // Propagate cross-covariance between current state and keyframe state:
  // P_cur_cross = F_t * P_prev_cross  (since keyframe state is static)
  cur_state.covariance.topRightCorner<15,6>() = F_t * prev_state.covariance.topRightCorner<15,6>();
  cur_state.covariance.bottomLeftCorner<6,15>() = cur_state.covariance.topRightCorner<15,6>().transpose();
  cur_state.covariance.bottomRightCorner<6,6>() = prev_state.covariance.bottomRightCorner<6,6>();
  cur_state.key_frame_time_stamp = prev_state.key_frame_time_stamp;
}

void EKFImuVision::updatePnP(AugState& cur_state, AugState& prev_state) {
  Vec6 z = cur_state.ut;  // [px, py, pz, roll, pitch, yaw]

  Mat6x21 H = Mat6x21::Zero();
  H.block<6,6>(0,0) = Mat6x6::Identity();

  Vec6 z_pred = prev_state.mean.head<6>();

  Vec6 y = z - z_pred;
  for (int i = 3; i < 6; i++) {
    y(i) = wrapAngle(y(i));
  }

  Mat6x6 S = H * prev_state.covariance * H.transpose() + Rt1_;

  // Mahalanobis outlier gate. The provided launch files use optimistic pose
  // noise, so keep the gate loose enough for normal PnP jitter but still reject
  // isolated solvePnP flips.
  // After long PnP outages, covariance is large, S is large, and even sizeable
  // innovations stay below the gate -- so legitimate snap-backs are accepted.
  // But isolated bad solvePnP frames (5m away with small covariance) get d^2 >> 16.81.
  double maha = y.transpose() * S.ldlt().solve(y);
  if (maha > 100.0) {
    ROS_WARN_THROTTLE(1.0, "PnP outlier rejected: maha=%.1f (innov=[%.2f %.2f %.2f])",
                       maha, y(0), y(1), y(2));
    cur_state.mean = prev_state.mean;
    cur_state.covariance = prev_state.covariance;
    cur_state.key_frame_time_stamp = prev_state.key_frame_time_stamp;
    return;
  }

  Mat21x6 K = prev_state.covariance * H.transpose() * S.inverse();
  cur_state.mean = prev_state.mean + K * y;

  Mat21x21 I_KH = Mat21x21::Identity() - K * H;
  cur_state.covariance = I_KH * prev_state.covariance * I_KH.transpose()
                       + K * Rt1_ * K.transpose();

  cur_state.key_frame_time_stamp = prev_state.key_frame_time_stamp;
}

void EKFImuVision::updateVO(AugState& cur_state, AugState& prev_state) {
  ros::Time meas_kf_time = cur_state.measurement_key_frame_time_stamp;
  if (meas_kf_time.isZero() || prev_state.key_frame_time_stamp != meas_kf_time) {
    cur_state.mean = prev_state.mean;
    cur_state.covariance = prev_state.covariance;
    cur_state.key_frame_time_stamp = prev_state.key_frame_time_stamp;
    return;
  }

  const Vec21& x_pred = prev_state.mean;
  const Mat21x21& P_pred = prev_state.covariance;

  Vec6 v6 = Vec6::Zero();
  Vec6 z      = cur_state.ut;
  Vec6 z_pred = modelG2(x_pred, v6);
  Mat6x21 H   = jacobiG2x(x_pred, v6);

  Vec6 y = z - z_pred;
  for (int i = 3; i < 6; i++) {
    y(i) = wrapAngle(y(i));
  }

  Mat6x6  S = H * P_pred * H.transpose() + Rt2_;
  double maha = y.transpose() * S.ldlt().solve(y);
  if (maha > 100.0) {
    ROS_WARN_THROTTLE(1.0, "VO outlier rejected: maha=%.1f (innov=[%.2f %.2f %.2f])",
                      maha, y(0), y(1), y(2));
    cur_state.mean = prev_state.mean;
    cur_state.covariance = prev_state.covariance;
    cur_state.key_frame_time_stamp = prev_state.key_frame_time_stamp;
    return;
  }
  Mat21x6 K = P_pred * H.transpose() * S.inverse();

  cur_state.mean = x_pred + K * y;
  Mat21x21 I_KH = Mat21x21::Identity() - K * H;
  cur_state.covariance = I_KH * P_pred * I_KH.transpose() + K * Rt2_ * K.transpose();

  cur_state.key_frame_time_stamp = prev_state.key_frame_time_stamp;
}

void EKFImuVision::changeAugmentedState(AugState& state) {
  // Drop the old keyframe slot, then re-augment by copying the current 15-state
  // pose into the keyframe slot. M_a_ creates the perfectly-correlated copy.
  Vec15    x15 = M_r_ * state.mean;
  Mat15x15 P15 = M_r_ * state.covariance * M_r_.transpose();
  state.mean       = M_a_ * x15;
  state.covariance = M_a_ * P15 * M_a_.transpose();
  state.key_frame_time_stamp = state.time_stamp;

  // Snapshot the new keyframe pose so updateVO can override the (potentially
  // stale) keyframe slot in prev_state during forward repropagation.
  current_keyframe_pose_ = state.mean.tail<6>();
  current_keyframe_cov_  = state.covariance.bottomRightCorner<6,6>();
  current_keyframe_time_ = state.time_stamp;
}

bool EKFImuVision::processNewState(AugState& new_state, bool change_keyframe) {
    // README's prescribed processing flow per call:
    //   1. insertNewState
    //   2. initFilter (if not yet initialized)
    //   3. repropagate from the affected point onward
    //   4. removeOldState
    //   5. publishFusedOdom
    bool dummy = false;

    (void)change_keyframe;

    // The first PnP is the absolute anchor of the EKF history.  Delayed
    // messages that belong before that anchor must not be inserted after
    // initialization: a stale VO key_stamp would otherwise create a synthetic
    // zero keyframe before the initialized state and corrupt repropagation.
    if (filter_initialized_ && new_state.time_stamp <= init_time_) {
        return false;
    }

    ros::Time reprop_from_time = new_state.time_stamp;
    if (new_state.type == vo && !new_state.measurement_key_frame_time_stamp.isZero()) {
        ros::Time kf_time = new_state.measurement_key_frame_time_stamp;
        if (filter_initialized_ && kf_time < init_time_) {
            return false;
        }
        if (keyframe_times_.insert(kf_time).second) {
            AugState kf_state;
            kf_state.type = keyframe;
            kf_state.time_stamp = kf_time;
            kf_state.key_frame_time_stamp = ros::Time(0);
            kf_state.measurement_key_frame_time_stamp = ros::Time(0);
            kf_state.mean.setZero();
            kf_state.covariance.setZero();
            kf_state.ut.setZero();
            insertNewState(kf_state);
            if (kf_time < reprop_from_time) reprop_from_time = kf_time;
        }
    }

    // Step 1: insert into time-ordered queue
    insertNewState(new_state);

    // Step 2: init on the first PnP
    if (!filter_initialized_ && new_state.type == pnp) {
        initFilter();
        if (!filter_initialized_) return false;

        AugState init_state = aug_state_hist_[latest_idx[pnp]];
        init_state.mean = state_;
        init_state.covariance = cov_;
        init_state.key_frame_time_stamp = init_state.time_stamp;
        init_state.measurement_key_frame_time_stamp = ros::Time(0);

        deque<AugState> trimmed_hist;
        trimmed_hist.push_back(init_state);
        for (const auto& state : aug_state_hist_) {
            if (state.time_stamp > init_state.time_stamp) {
                trimmed_hist.push_back(state);
            }
        }
        aug_state_hist_.swap(trimmed_hist);

        keyframe_times_.clear();
        current_keyframe_time_ = init_state.time_stamp;
        current_keyframe_pose_ = state_.tail<6>();
        current_keyframe_cov_  = cov_.bottomRightCorner<6,6>();
        keyframe_times_.insert(current_keyframe_time_);

        // Step 3a: propagate forward through any post-init zero-mean states
        auto front_it = aug_state_hist_.begin();
        repropagate(front_it, dummy);

        for (int i = 0; i < 4; i++) latest_idx[i] = -1;
        latest_idx[pnp] = 0;

        // Steps 4–5 still run below.
    } else if (filter_initialized_) {
        // Step 3b: out-of-order handling — repropagate from the inserted state
        // onward. The state before `it` already has a correct mean/covariance
        // from when it was last processed; this fills in `it` and any later
        // states that need updating.
        auto first_affected = aug_state_hist_.begin();
        while (first_affected != aug_state_hist_.end() &&
               first_affected->time_stamp < reprop_from_time) {
            ++first_affected;
        }
        if (first_affected == aug_state_hist_.end()) {
            if (!aug_state_hist_.empty()) {
                auto seed = std::prev(aug_state_hist_.end());
                repropagate(seed, dummy);
            }
        } else if (first_affected != aug_state_hist_.begin()) {
            auto seed = std::prev(first_affected);
            repropagate(seed, dummy);
        } else {
            repropagate(first_affected, dummy);
        }
    }

    // Step 4: bound the history queue
    removeOldState();

    // Step 5: publish the fused odometry
    if (filter_initialized_) {
        publishFusedOdom();
    }
    return true;
}

deque<AugState>::iterator EKFImuVision::insertNewState(AugState& new_state) {
  // TODO
  // insert the new state to the queue
  // update the latest_idx of the type of the new state
  // return the iterator point to the new state in the queue
  ros::Time time = new_state.time_stamp;
  deque<AugState>::iterator state_it = aug_state_hist_.end();
  while (state_it != aug_state_hist_.begin() && (state_it -1)->time_stamp > time) {
    state_it--;
  }
  
  state_it = aug_state_hist_.insert(state_it, new_state);
  
  if (new_state.type == imu) {
    latest_idx[imu] = state_it - aug_state_hist_.begin();
  } else if (new_state.type == pnp) {
    latest_idx[pnp] = state_it - aug_state_hist_.begin();
  } else if (new_state.type == vo) {
    latest_idx[vo] = state_it - aug_state_hist_.begin();
  } else if (new_state.type == keyframe) {
    latest_idx[keyframe] = state_it - aug_state_hist_.begin();
  }

  return state_it;
}

void EKFImuVision::repropagate(deque<AugState>::iterator& start_it, bool& init) {
    if (aug_state_hist_.empty()) return;

    // 从 start_it 的下一个元素开始
    auto it = start_it;
    ++it;
    while (it != aug_state_hist_.end()) {
        auto prev = std::prev(it);
        if (it->type == imu) {
            predictIMU(*it, *prev, it->ut);
        } else if (it->type == pnp) {
            updatePnP(*it, *prev);
        } else if (it->type == keyframe) {
            it->mean = prev->mean;
            it->covariance = prev->covariance;
            it->key_frame_time_stamp = prev->key_frame_time_stamp;
            changeAugmentedState(*it);
        } else if (it->type == vo) {
            updateVO(*it, *prev);
        }
        ++it;
    }
    if (!aug_state_hist_.empty()) {
        const AugState& last = aug_state_hist_.back();
        current_keyframe_time_ = last.key_frame_time_stamp;
        current_keyframe_pose_ = last.mean.tail<6>();
        current_keyframe_cov_ = last.covariance.bottomRightCorner<6,6>();
    }
}

void EKFImuVision::removeOldState() {
  // Bound the history queue. Keep ~5 s of history but never drop the active
  // keyframe state — Phase 2 of processNewState looks it up by timestamp.
  if (aug_state_hist_.size() < 500) return;
  ros::Time cutoff = aug_state_hist_.back().time_stamp - ros::Duration(5.0);
  if (!current_keyframe_time_.isZero() && current_keyframe_time_ < cutoff) {
    cutoff = current_keyframe_time_;
  }
  while (aug_state_hist_.size() > 1 && aug_state_hist_.front().time_stamp < cutoff) {
    if (aug_state_hist_.front().type == keyframe) {
      keyframe_times_.erase(aug_state_hist_.front().time_stamp);
    }
    aug_state_hist_.pop_front();
  }
  for (int i = 0; i < 4; i++) latest_idx[i] = -1;
}

void EKFImuVision::publishFusedOdom() {
  if (aug_state_hist_.empty()) {
    ROS_WARN_THROTTLE(1.0, "publishFusedOdom: state history empty");
    return;
  }

  AugState last_state = aug_state_hist_.back();

  if (last_state.mean.head(3).norm() > 20) {
    ROS_ERROR_STREAM("error state: " << last_state.mean.head(3).transpose());
    return;
  }

  // Publish on every EKF update. No throttle, no rate-limiting smoother.
  double phi   = last_state.mean(3);
  double theta = last_state.mean(4);
  double psi   = last_state.mean(5);

  ROS_INFO_THROTTLE(0.2,
      "t=%.3f p=[%+.3f %+.3f %+.3f] v=[%+.3f %+.3f %+.3f] euler=[%+.4f %+.4f %+.4f] ba=[%+.3f %+.3f %+.3f]",
      last_state.time_stamp.toSec(),
      last_state.mean(0), last_state.mean(1), last_state.mean(2),
      last_state.mean(6), last_state.mean(7), last_state.mean(8),
      phi, theta, psi,
      last_state.mean(12), last_state.mean(13), last_state.mean(14));

  Eigen::Quaterniond q = Eigen::AngleAxisd(psi, Eigen::Vector3d::UnitZ()) *
                         Eigen::AngleAxisd(phi, Eigen::Vector3d::UnitX()) *
                         Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitY());
  nav_msgs::Odometry odom;
  odom.header.frame_id = "world";
  odom.header.stamp = last_state.time_stamp;
  odom.child_frame_id = "base_link";
  odom.pose.pose.position.x = last_state.mean(0);
  odom.pose.pose.position.y = last_state.mean(1);
  odom.pose.pose.position.z = last_state.mean(2);
  odom.pose.pose.orientation.w = q.w();
  odom.pose.pose.orientation.x = q.x();
  odom.pose.pose.orientation.y = q.y();
  odom.pose.pose.orientation.z = q.z();
  odom.twist.twist.linear.x = last_state.mean(6);
  odom.twist.twist.linear.y = last_state.mean(7);
  odom.twist.twist.linear.z = last_state.mean(8);
  fuse_odom_pub_.publish(odom);

  // Throttle path-point appending to ~20 Hz so the Path topic doesn't blow up
  // with 25k+ points at 400+Hz publish (heavy on RViz).
  static ros::Time last_path_t(0);
  if (last_path_t.isZero() || (last_state.time_stamp - last_path_t).toSec() >= 0.05) {
    geometry_msgs::PoseStamped path_pose;
    path_pose.header.frame_id = path_.header.frame_id = "world";
    path_.header.stamp = last_state.time_stamp;
    path_pose.header.stamp = last_state.time_stamp;
    path_pose.pose.position.x = last_state.mean(0);
    path_pose.pose.position.y = last_state.mean(1);
    path_pose.pose.position.z = last_state.mean(2);
    path_.poses.push_back(path_pose);
    path_pub_.publish(path_);
    last_path_t = last_state.time_stamp;
  }
}

bool EKFImuVision::initFilter() {
  // TODO
  // Initial the filter when a keyframe after marker PnP measurements is available
    if (latest_idx[pnp] < 0 ||
        static_cast<size_t>(latest_idx[pnp]) >= aug_state_hist_.size()) {
      return false;
    }
    AugState &pnp_state = aug_state_hist_[latest_idx[pnp]];
    if (pnp_state.type != pnp) return false;  // 关键保护

  Vec6 pnp_meas = pnp_state.ut;
  state_.head<3>() = pnp_meas.head<3>();
  state_.segment<3>(3) = pnp_meas.segment<3>(3);
  state_.segment<3>(6).setZero();  // vel
  state_.segment<3>(9).setZero();  // bg
  state_.segment<3>(12).setZero(); // ba

  // 关键帧状态也初始化为相同的位姿
  state_.segment<3>(15) = pnp_meas.head<3>();
  state_.segment<3>(18) = pnp_meas.segment<3>(3);

  // 设置初始化时间戳
  init_time_ = pnp_state.time_stamp;
  current_time_ = init_time_;
  filter_initialized_ = true;

  Mat15x15 P15 = Mat15x15::Identity() * 0.1;
  cov_ = M_a_ * P15 * M_a_.transpose();

  ROS_INFO("Filter initialized at time %.3f with body p=[%.3f %.3f %.3f] euler=[%.3f %.3f %.3f]",
           init_time_.toSec(),
           state_(0), state_(1), state_(2),
           state_(3), state_(4), state_(5));
  return true;
}

bool EKFImuVision::initUsingPnP(deque<AugState>::iterator start_it) {
  if (start_it == aug_state_hist_.end() || start_it->type != pnp) return false;

  Vec15 x15 = Vec15::Zero();
  x15.head<3>() = start_it->ut.head<3>();
  x15.segment<3>(3) = start_it->ut.segment<3>(3);

  start_it->mean = M_a_ * x15;
  Mat15x15 P15 = Mat15x15::Identity() * 0.1;
  start_it->covariance = M_a_ * P15 * M_a_.transpose();
  start_it->key_frame_time_stamp = start_it->time_stamp;
  start_it->measurement_key_frame_time_stamp = ros::Time(0);
  return true;
}

Vec3 EKFImuVision::rotation2Euler(const Mat3x3& R) {
  double s = R(2, 1);
  if (s > 1.0) s = 1.0;
  if (s < -1.0) s = -1.0;
  double phi = asin(s);
  double theta = atan2(-R(2, 0), R(2, 2));
  double psi = atan2(-R(0, 1), R(1, 1));
  return Vec3(phi, theta, psi);
}
Vec3 EKFImuVision::quaternion2Euler(double w, double x, double y, double z) {
    // 先转成旋转矩阵，再调用已有的 rotation2Euler
    Eigen::Quaterniond q(w, x, y, z);
    Mat3x3 R = q.toRotationMatrix();
    return rotation2Euler(R);   // 这个在 ekf_model.cpp 中已经实现
}

}  // namespace ekf_imu_vision
