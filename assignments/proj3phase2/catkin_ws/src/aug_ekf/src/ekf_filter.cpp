#include <ekf_filter.h>

namespace ekf_imu_vision {
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
  
    /* ===== 增广矩阵 M_a_ 和缩减矩阵 M_r_ =====
   *  M_a_ ∈ ℝ^{21×15} 用于将 15 维状态扩展为 21 维:
   *  x_aug = M_a_ * x_orig
   *  M_a_ = [ I_15 ]
   *         [ B    ]   其中 B 选出位姿部分 (p,q) 复制到增广状态
   *  M_r_ ∈ ℝ^{15×21} 用于移除增广部分:
   *  x_orig = M_r_ * x_aug
   *  M_r_ = [ I_15  0 ]
   */
  M_a_ = Mat(21, 15)::Zero();
  M_a_.topLeftCorner<15,15>() = Mat15x15::Identity();  // 前15维不变
  // 后6维复制当前姿态（将状态中第3~5维的姿态复制到增广部分）
  M_a_.block<3,3>(15, 3) = Mat3x3::Identity();        // 位置部分不复制，只关心姿态
  // 实际上关键帧状态包含位置和姿态，因此需要复制前6维：
  // 修正：关键帧是 [pK; qK]，所以应复制 x[0..2] 和 x[3..5]
  M_a_.block<3,3>(15, 0) = Mat3x3::Identity();        // 位置复制
  M_a_.block<3,3>(18, 3) = Mat3x3::Identity();        // 姿态复制

  // M_r_ 将 21 维缩减为 15 维，直接丢弃后6维
  M_r_ = Mat(15, 21)::Zero();
  M_r_.topLeftCorner<15,15>() = Mat15x15::Identity();

  for (int i = 0; i < 3; i++) {
    /* process noise */
    node_.param("aug_ekf/ng", Qt_(i, i), 200.0);
    node_.param("aug_ekf/na", Qt_(i + 3, i + 3), 500.0);
    node_.param("aug_ekf/nbg", Qt_(i + 6, i + 6), 0.5);
    node_.param("aug_ekf/nba", Qt_(i + 9, i + 9), 5.0);
    node_.param("aug_ekf/pnp_p", Rt1_(i,i),      0.002);  // x,y
    if (i == 2) Rt1_(2,2) = 0.025;   // z 单独设
    node_.param("aug_ekf/pnp_q", Rt1_(i+3,i+3),  0.004);  // roll,pitch
    if (i == 2) Rt1_(5,5) = 0.025;  // yaw
    node_.param("aug_ekf/vo_pos", Rt2_(i, i), 9e-4);
    node_.param("aug_ekf/vo_rot", Rt2_(i + 3, i + 3), 9e-4);
  }

  init_ = false;
  current_imu_ut.setZero();
  path_.poses.clear();

  for (int i = 0; i < 4; i++) latest_idx[i] = 0;

  /* ---------- subscribe and publish ---------- */
  imu_sub_ =
      node_.subscribe<sensor_msgs::Imu>("/dji_sdk_1/dji_sdk/imu", 100, &EKFImuVision::imuCallback, this);
  pnp_sub_ = node_.subscribe<nav_msgs::Odometry>("tag_odom", 10, &EKFImuVision::PnPCallback, this);
  // opti_tf_sub_ = node_.subscribe<geometry_msgs::PointStamped>("opti_tf_odom", 10,
  // &EKFImuVision::opticalCallback, this);
  // stereo_sub_ = node_.subscribe<stereo_vo::relative_pose>("/vo/Relative_pose", 10,
  //                                                         &EKFImuVision::stereoVOCallback, this);
  fuse_odom_pub_ = node_.advertise<nav_msgs::Odometry>("ekf_fused_odom", 10);
  path_pub_ = node_.advertise<nav_msgs::Path>("/aug_ekf/Path", 100);

  ros::Duration(0.5).sleep();

  ROS_INFO("Start ekf.");
}

void EKFImuVision::PnPCallback(const nav_msgs::OdometryConstPtr& msg) {
  AugState new_state;
  new_state.time_stamp = msg->header.stamp;
  new_state.key_frame_time_stamp = ros::Time(0);
  new_state.type = pnp;
  new_state.mean.setZero();
  new_state.covariance.setZero();
  new_state.ut.setZero();

  const auto& pose = msg->pose.pose;
  Vec3 p(pose.position.x, pose.position.y, pose.position.z);
  Eigen::Quaterniond q(pose.orientation.w,
                       pose.orientation.x,
                       pose.orientation.y,
                       pose.orientation.z);
  Vec3 euler = rotation2Euler(q.toRotationMatrix());
  new_state.ut << p, euler;

  processNewState(new_state, false);
}

void EKFImuVision::stereoVOCallback(const stereo_vo::relative_poseConstPtr& msg) {
  AugState new_state;
  new_state.time_stamp = msg->header.stamp;
  new_state.key_frame_time_stamp = msg->key_stamp;
  new_state.type = vo;
  new_state.mean.setZero();
  new_state.covariance.setZero();
  new_state.ut.setZero();

  const auto& pose = msg->relative_pose;
  Vec3 dp(pose.position.x, pose.position.y, pose.position.z);
  Eigen::Quaterniond dq(pose.orientation.w,
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z);
  Vec3 deuler = rotation2Euler(dq.toRotationMatrix());
  new_state.ut << dp, deuler;

  processNewState(new_state, false);
}

void EKFImuVision::imuCallback(const sensor_msgs::ImuConstPtr& imu_msg) {
  // TODO
  // construct a new state using the IMU input and insert the new state into the queue

  AugState new_state;
  new_state.time_stamp = imu_msg->header.stamp;
  new_state.key_frame_time_stamp = ros::Time(0);
  new_state.type = imu;
  new_state.mean.setZero();
  new_state.covariance.setZero();
  new_state.ut.setZero();

  // 存储 IMU 测量到 ut
  new_state.ut << imu_msg->angular_velocity.x,
                  imu_msg->angular_velocity.y,
                  imu_msg->angular_velocity.z,
                  imu_msg->linear_acceleration.x,
                  imu_msg->linear_acceleration.y,
                  imu_msg->linear_acceleration.z;

  if (!processNewState(new_state, false)) {
    return;
  }
}

void EKFImuVision::predictIMU(AugState& cur_state, AugState& prev_state, Vec6 ut) {
  // prev_state 是上一个时刻的状态（15维或21维？当前状态可能是15维或21维）
  // 但由于滤波器可能已经增广，所以均值向量长度是21。预测只作用于前15维，
  // 增广部分保持不变。
  
  double dt = (cur_state.time_stamp - prev_state.time_stamp).toSec();
  if (dt <= 0.0) {
    // 如果时间倒退或相同，不做预测
    cur_state.mean = prev_state.mean;
    cur_state.covariance = prev_state.covariance;
    cur_state.key_frame_time_stamp = prev_state.key_frame_time_stamp;
    return;
  }
  
  // 提取原始15维状态和协方差
  Vec15 x_prev = prev_state.mean.head<15>();
  Mat15x15 P_prev = prev_state.covariance.topLeftCorner<15,15>();
  
  // 模型函数调用 (使用 n=0)
  Vec12 n_zero = Vec12::Zero();
  Vec15 xdot = modelF(x_prev, ut, n_zero);
  Mat15x15 A = jacobiFx(x_prev, ut, n_zero);
  Mat15x12 U = jacobiFn(x_prev, ut, n_zero);
  
  // 离散化
  Mat15x15 F = Mat15x15::Identity() + dt * A;
  Mat15x12 V = dt * U;
  
  // 预测均值
  Vec15 x_pred = x_prev + dt * xdot;
  
  // 预测协方差
  Mat15x15 P_pred = F * P_prev * F.transpose() + V * Qt_ * V.transpose();
  
  // 更新当前状态（前15维）
  cur_state.mean.head<15>() = x_pred;
  cur_state.covariance.topLeftCorner<15,15>() = P_pred;
  
  // 保持增广部分不变
  if (prev_state.mean.size() == 21) {
    cur_state.mean.tail<6>() = prev_state.mean.tail<6>();
    cur_state.covariance.topRightCorner<15,6>() = F * prev_state.covariance.topRightCorner<15,6>();
    cur_state.covariance.bottomLeftCorner<6,15>() = cur_state.covariance.topRightCorner<15,6>().transpose();
    cur_state.covariance.bottomRightCorner<6,6>() = prev_state.covariance.bottomRightCorner<6,6>();
  }
}

void EKFImuVision::updatePnP(AugState& cur_state, AugState& prev_state) {
  // 测量值从 prev_state.ut 中获取
  Vec6 z_meas = prev_state.ut;
  
  // 预测观测：使用当前预测的状态（注意此时状态已经是 predict 后的结果）
  Vec15 x_pred = prev_state.mean.head<15>();  // 因为类型是 pnp，此时 mean 已包含预测
  Vec6 v_zero = Vec6::Zero();
  Vec6 z_pred = modelG1(x_pred, v_zero);
  Mat6x15 C = jacobiG1x(x_pred, v_zero);
  Mat6x6 W = jacobiG1v(x_pred, v_zero);
  
  // 取当前协方差（可能是21维，但只需前15×15）
  Mat15x15 P = prev_state.covariance.topLeftCorner<15,15>();
  
  // 卡尔曼增益
  Mat6x6 S = C * P * C.transpose() + W * Rt1_ * W.transpose();
  Mat15x6 K = P * C.transpose() * S.inverse();
  
  // 更新均值（前15维）
  Vec15 x_updated = x_pred + K * (z_meas - z_pred);
  // 更新协方差
  Mat15x15 P_updated = (Mat15x15::Identity() - K * C) * P;
  
  // 写入 cur_state
  cur_state.mean.head<15>() = x_updated;
  cur_state.covariance.topLeftCorner<15,15>() = P_updated;
  
  // 增广部分保持不变（如果有的话）
  if (prev_state.mean.size() == 21) {
    cur_state.mean.tail<6>() = prev_state.mean.tail<6>();
    // 协方差交叉项也需要更新（因为状态更新影响互协方差）
    Mat15x6 P_cross = prev_state.covariance.topRightCorner<15,6>();
    Mat15x6 P_cross_new = (Mat15x15::Identity() - K * C) * P_cross;
    cur_state.covariance.topRightCorner<15,6>() = P_cross_new;
    cur_state.covariance.bottomLeftCorner<6,15>() = P_cross_new.transpose();
    // 关键帧自身方差不变，因为未更新
    cur_state.covariance.bottomRightCorner<6,6>() = prev_state.covariance.bottomRightCorner<6,6>();
  }
}

void EKFImuVision::updateVO(AugState& cur_state, AugState& prev_state) {
  // 测量值
  Vec6 z_meas = prev_state.ut;
  
  // 当前状态必须是21维（增广状态）
  Vec21 x_pred = prev_state.mean;
  Vec6 v_zero = Vec6::Zero();
  Vec6 z_pred = modelG2(x_pred, v_zero);
  Mat6x21 C = jacobiG2x(x_pred, v_zero);
  Mat6x6 W = jacobiG2v(x_pred, v_zero);
  
  Mat21x21 P = prev_state.covariance;
  
  Mat6x6 S = C * P * C.transpose() + W * Rt2_ * W.transpose();
  Mat21x6 K = P * C.transpose() * S.inverse();
  
  Vec21 x_updated = x_pred + K * (z_meas - z_pred);
  Mat21x21 P_updated = (Mat21x21::Identity() - K * C) * P;
  
  cur_state.mean = x_updated;
  cur_state.covariance = P_updated;
  cur_state.key_frame_time_stamp = prev_state.key_frame_time_stamp;
}

void EKFImuVision::changeAugmentedState(AugState& state) {
  ROS_ERROR("----------------change keyframe------------------------");
  // 找到与新 keyframe 时间戳匹配的状态（队列中已存在的某个历史状态）
  ros::Time kf_time = state.key_frame_time_stamp;
  auto it = aug_state_hist_.begin();
  for (; it != aug_state_hist_.end(); ++it) {
    if (it->time_stamp == kf_time) break;
  }
  if (it == aug_state_hist_.end()) {
    ROS_WARN("Keyframe timestamp not found in history, cannot change augmented state.");
    return;
  }
  
  // 复制该历史状态的位姿到增广部分
  Vec15 x_hist = it->mean.head<15>();
  state.mean.tail<6>() = x_hist.head<6>();  // [p_kf; q_kf]
  
  // 协方差更新：需要从历史状态的协方差中提取与当前状态的互相关
  // 简化处理：将增广部分的协方差设为历史状态协方差对应块，交叉项通过 M_a 传播
  // 但在实际中更严谨的做法是重新通过增广操作计算，这里采用近似：
  // 取出历史时刻的协方差中位姿块，作为增广部分的方差
  Mat15x15 P_hist = it->covariance.topLeftCorner<15,15>();
  Mat6x6 P_kf = P_hist.topLeftCorner<6,6>();
  state.covariance.bottomRightCorner<6,6>() = P_kf;
  
  // 交叉项：历史状态与当前状态的互协方差。由于我们很难直接获得，可近似设为零，
  // 但更好的做法是通过线性变换从历史状态传播到现在的协方差得到。
  // 简单起见，先置零，然后在后续的VO更新中通过滤波修正。
  state.covariance.topRightCorner<15,6>().setZero();
  state.covariance.bottomLeftCorner<6,15>().setZero();
}

bool EKFImuVision::processNewState(AugState& new_state, bool change_keyframe) {
  // 1. 插入队列
  auto it = insertNewState(new_state);
  if (it == aug_state_hist_.end()) return false;
  
  // 2. 尝试初始化
  bool init_before = init_;
  if (!init_) {
    initFilter();
    if (!init_) return false;
  }
  if (!init_before && init_) {
    it = aug_state_hist_.begin();
  }
  
  // 3. 重新传播
  bool dummy_init = init_;
  repropagate(it, dummy_init);
  
  // 4. 删除旧状态
  removeOldState();
  
  return true;
}

deque<AugState>::iterator EKFImuVision::insertNewState(AugState& new_state) {
  // 按时间顺序插入
  auto it = aug_state_hist_.begin();
  while (it != aug_state_hist_.end() && it->time_stamp < new_state.time_stamp) {
    ++it;
  }
  it = aug_state_hist_.insert(it, new_state);
  
  // 更新最新索引
  switch (new_state.type) {
    case imu: latest_idx[0] = it - aug_state_hist_.begin(); break;
    case pnp: latest_idx[1] = it - aug_state_hist_.begin(); break;
    case vo:  latest_idx[2] = it - aug_state_hist_.begin(); break;
  }
  
  return it;
}
void EKFImuVision::repropagate(deque<AugState>::iterator& new_input_it, bool& init) {
  // 如果尚未初始化，尝试初始化
  if (!init_ && !initFilter()) {
    return;
  }
  
  // 从 new_input_it 开始，向前传播
  for (auto it = new_input_it; it != aug_state_hist_.end(); ++it) {
    if (it == aug_state_hist_.begin()) continue; // 第一个无法预测
    auto prev_it = it - 1;
    
    // 保存新插入状态的原始信息
    int original_type = it->type;
    Vec6 original_ut = it->ut;
    ros::Time original_time = it->time_stamp;
    ros::Time original_kf_time = it->key_frame_time_stamp;
    
    // 复制上一个状态作为初始，然后根据类型进行 predict/update
    *it = *prev_it;
    
    // 恢复新插入状态的所有原始信息
    it->type = original_type;
    it->ut = original_ut;
    it->time_stamp = original_time;
    it->key_frame_time_stamp = original_kf_time;
    
    switch (it->type) {
      case imu: {
        predictIMU(*it, *prev_it, it->ut);
        break;
      }
      case pnp: {
        updatePnP(*it, *prev_it);
        break;
      }
      case vo: {
        // 检查是否需要切换关键帧
        if (it->key_frame_time_stamp != prev_it->key_frame_time_stamp) {
          auto kf_it = aug_state_hist_.begin();
          for (; kf_it != aug_state_hist_.end(); ++kf_it) {
            if (kf_it->time_stamp == it->key_frame_time_stamp) break;
          }
          if (kf_it != aug_state_hist_.end()) {
            it->mean.tail<6>() = kf_it->mean.head<6>();
            Mat6x6 P_kf = kf_it->covariance.topLeftCorner<6,6>();
            it->covariance.bottomRightCorner<6,6>() = P_kf;
            it->covariance.topRightCorner<15,6>().setZero();
            it->covariance.bottomLeftCorner<6,15>().setZero();
          }
        }
        updateVO(*it, *prev_it);
        break;
      }
    }
  }
  
  publishFusedOdom();
}

void EKFImuVision::removeOldState() {
  // 只保留最近一定数量的状态，或只保留每个类型的最新一个
  if (aug_state_hist_.size() > 200) {
    aug_state_hist_.pop_front();
    // 调整 latest_idx
    for (int i = 0; i < 4; ++i) if (latest_idx[i] > 0) latest_idx[i]--;
  }
}

void EKFImuVision::publishFusedOdom() {
  if (aug_state_hist_.empty()) return;

  AugState last_state = aug_state_hist_.back();

  double phi, theta, psi;
  phi = last_state.mean(3);
  theta = last_state.mean(4);
  psi = last_state.mean(5);

  if (last_state.mean.head(3).norm() > 20) {
    ROS_ERROR_STREAM("error state: " << last_state.mean.head(3).transpose());
    return;
  }

  // using the zxy euler angle
  Eigen::Quaterniond q = Eigen::AngleAxisd(psi, Eigen::Vector3d::UnitZ()) *
                         Eigen::AngleAxisd(phi, Eigen::Vector3d::UnitX()) *
                         Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitY());
  nav_msgs::Odometry odom;
  odom.header.frame_id = "world";
  odom.header.stamp = last_state.time_stamp;

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

  geometry_msgs::PoseStamped path_pose;
  path_pose.header.frame_id = path_.header.frame_id = "world";
  path_pose.pose.position.x = last_state.mean(0);
  path_pose.pose.position.y = last_state.mean(1);
  path_pose.pose.position.z = last_state.mean(2);
  path_.poses.push_back(path_pose);
  path_pub_.publish(path_);
}

bool EKFImuVision::initFilter() {
  // 检查是否已存在 PnP 和 VO 关键帧
  if (aug_state_hist_.empty()) return false;
  
  // 寻找第一个 PnP 测量
  auto pnp_it = aug_state_hist_.begin();
  for (; pnp_it != aug_state_hist_.end(); ++pnp_it) {
    if (pnp_it->type == pnp) break;
  }
  if (pnp_it == aug_state_hist_.end()) return false;
  
  // 用 PnP 初始化队列中所有后续状态
  if (!initUsingPnP(pnp_it)) return false;
  
  // 寻找第一个 VO 测量
  auto vo_it = aug_state_hist_.begin();
  for (; vo_it != aug_state_hist_.end(); ++vo_it) {
    if (vo_it->type == vo && vo_it->key_frame_time_stamp != ros::Time(0)) break;
  }
  if (vo_it == aug_state_hist_.end()) return false;

  // 将队列中所有状态统一扩展为21维，避免传播时维度不一致
  for (auto iter = aug_state_hist_.begin(); iter != aug_state_hist_.end(); ++iter) {
    if (iter->mean.size() == 21) continue;

    Vec21 new_mean = Vec21::Zero();
    Mat21x21 new_cov = Mat21x21::Zero();
    new_mean.head<15>() = iter->mean.head<15>();
    new_cov.topLeftCorner<15,15>() = iter->covariance.topLeftCorner<15,15>();

    Vec6 kf_pose = iter->mean.head<6>();
    Mat6x6 kf_cov = Mat6x6::Identity() * 100.0;
    if (iter->key_frame_time_stamp != ros::Time(0)) {
      auto kf_it = aug_state_hist_.begin();
      for (; kf_it != aug_state_hist_.end(); ++kf_it) {
        if (kf_it->time_stamp == iter->key_frame_time_stamp) break;
      }
      if (kf_it != aug_state_hist_.end()) {
        kf_pose = kf_it->mean.head<6>();
        kf_cov = kf_it->covariance.topLeftCorner<6,6>();
      }
    }

    new_mean.tail<6>() = kf_pose;
    new_cov.bottomRightCorner<6,6>() = kf_cov;
    new_cov.topRightCorner<15,6>().setZero();
    new_cov.bottomLeftCorner<6,15>().setZero();

    iter->mean = new_mean;
    iter->covariance = new_cov;
  }
  
  ROS_INFO("EKF filter initialized successfully!");
  init_ = true;
  return true;
}

bool EKFImuVision::initUsingPnP(deque<AugState>::iterator start_it) {
  // 寻找从 start_it 开始第一个类型为 pnp 的状态
  auto it = start_it;
  while (it != aug_state_hist_.end() && it->type != pnp) ++it;
  if (it == aug_state_hist_.end()) return false;
  
  Vec6 pnp_meas = it->ut;  // [p, q]
  Vec15 init_mean = Vec15::Zero();
  init_mean.head<6>() = pnp_meas;   // 位置和姿态直接用 PnP 测量初始化

  // --- 初始协方差：参照已验证的简单 EKF ---
  Mat15x15 init_cov = Mat15x15::Zero();

  // 位置协方差 = PnP 测量噪声 (Rt1_ 的对应部分)
  init_cov.block<3,3>(0,0) = Rt1_.block<3,3>(0,0);   // 0.002, 0.002, 0.025

  // 姿态协方差 = PnP 测量噪声
  init_cov.block<3,3>(3,3) = Rt1_.block<3,3>(3,3);   // 0.004, 0.004, 0.025

  // 速度完全未知，设大方差（标准差 2 m/s → 方差 4）
  init_cov.block<3,3>(6,6) = Mat3x3::Identity() * 4.0;

  // 陀螺仪 bias 初始猜测为 0，留一定不确定性（方差 0.01）
  init_cov.block<3,3>(9,9) = Mat3x3::Identity() * 0.01;

  // 加速度计 bias 类似（方差 1.0）
  init_cov.block<3,3>(12,12) = Mat3x3::Identity() * 1.0;

  // 将队列中所有当前及之后的未初始化状态赋初始均值与协方差
  for (auto iter = start_it; iter != aug_state_hist_.end(); ++iter) {
    iter->mean.head<15>() = init_mean;
    iter->covariance.topLeftCorner<15,15>() = init_cov;
  }
  return true;
}

Vec3 EKFImuVision::rotation2Euler(const Mat3x3& R) {
  double phi = asin(R(2, 1));
  double theta = atan2(-R(2, 0), R(2, 2));
  double psi = atan2(-R(0, 1), R(1, 1));
  return Vec3(phi, theta, psi);
}

}  // namespace ekf_imu_vision
