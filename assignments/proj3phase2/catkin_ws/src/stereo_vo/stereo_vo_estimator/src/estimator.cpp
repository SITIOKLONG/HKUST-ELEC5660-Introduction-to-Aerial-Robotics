#include "estimator.h"

void drawImage(const cv::Mat &img, const vector<cv::Point2f> &pts, string name)
{
  auto draw = img.clone();
  for (unsigned int i = 0; i < pts.size(); i++)
  {
    cv::circle(draw, pts[i], 2, cv::Scalar(0, 255, 0), -1, 8);
  }
  cv::imshow(name, draw);
  cv::waitKey(1);
}

Estimator::Estimator()
{
  ROS_INFO("Estimator init begins.");
  prev_frame.frame_time = ros::Time(0.0);
  prev_frame.w_t_c = Eigen::Vector3d(0, 0, 0);
  prev_frame.w_R_c = Eigen::Matrix3d::Identity();
  fail_cnt = 0;
  init_finish = false;
}

void Estimator::reset()
{
  ROS_ERROR("Lost, reset!");
  key_frame = prev_frame;
  fail_cnt = 0;
  init_finish = false;
}

void Estimator::setParameter()
{
  for (int i = 0; i < 2; i++)
  {
    tic[i] = TIC[i];
    ric[i] = RIC[i];
    cout << " exitrinsic cam " << i << endl
         << ric[i] << endl
         << tic[i].transpose() << endl;
  }

  prev_frame.frame_time = ros::Time(0.0);
  prev_frame.w_t_c = tic[0];
  prev_frame.w_R_c = ric[0];
  key_frame = prev_frame;

  readIntrinsicParameter(CAM_NAMES);

  // transform between left and right camera
  Matrix4d Tl, Tr;
  Tl.setIdentity();
  Tl.block(0, 0, 3, 3) = ric[0];
  Tl.block(0, 3, 3, 1) = tic[0];
  Tr.setIdentity();
  Tr.block(0, 0, 3, 3) = ric[1];
  Tr.block(0, 3, 3, 1) = tic[1];
  Tlr = Tl.inverse() * Tr;
}

void Estimator::readIntrinsicParameter(const vector<string> &calib_file)
{
  for (size_t i = 0; i < calib_file.size(); i++)
  {
    ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());
    camodocal::CameraPtr camera =
        camodocal::CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
    m_camera.push_back(camera);
  }
}

bool Estimator::inputImage(ros::Time time_stamp, const cv::Mat &_img, const cv::Mat &_img1)
{

  if (fail_cnt > 20)
  {
    reset();
  }
  std::cout << "receive new image===========================" << std::endl;

  Estimator::frame cur_frame;
  cur_frame.frame_time = time_stamp;
  cur_frame.img = _img;

  // cv::imshow("img", _img);
  // cv::waitKey(1);

  vector<cv::Point2f> left_pts_2d, right_pts_2d;
  vector<cv::Point3f> key_pts_3d;
  bool pose_estimated = false;

  c_R_k.setIdentity();
  c_t_k.setZero();

  if (init_finish)
  {
    // Track 2D-3D correspondences from keyframe to current left image.
    if (trackFeatureBetweenFrames(key_frame, _img, key_pts_3d, left_pts_2d))
    {
      vector<cv::Point2f> cur_un_pts = undistortedPts(left_pts_2d, m_camera[0]);
      pose_estimated = estimateTBetweenFrames(key_pts_3d, cur_un_pts, c_R_k, c_t_k);
    }

    if (pose_estimated)
    {
      fail_cnt = 0;
    }
    else
    {
      fail_cnt++;
      key_pts_3d.clear();
    }

  }

  // Extract new features in current left image and match to right image.
  left_pts_2d.clear();
  right_pts_2d.clear();
  extractNewFeatures(_img, left_pts_2d);
  bool stereo_matched = trackFeatureLeftRight(_img, _img1, left_pts_2d, right_pts_2d);

  // Compute current camera pose in world frame.
  if (!init_finish)
  {
    cur_frame.w_R_c = key_frame.w_R_c;
    cur_frame.w_t_c = key_frame.w_t_c;
  }
  else if (pose_estimated)
  {
    // p_c = c_R_k * p_k + c_t_k
    cur_frame.w_R_c = key_frame.w_R_c * c_R_k.transpose();
    cur_frame.w_t_c = key_frame.w_t_c - cur_frame.w_R_c * c_t_k;
  }
  else
  {
    // Keep continuity when PnP fails temporarily.
    cur_frame.w_R_c = prev_frame.w_R_c;
    cur_frame.w_t_c = prev_frame.w_t_c;
  }

  // Triangulate new 3D points from stereo correspondences.
  cur_frame.xyz.clear();
  cur_frame.uv.clear();
  if (stereo_matched)
  {
    vector<cv::Point2f> left_un_pts = undistortedPts(left_pts_2d, m_camera[0]);
    vector<cv::Point2f> right_un_pts = undistortedPts(right_pts_2d, m_camera[1]);
    cur_frame.uv = left_pts_2d; // keep pixel coordinates for future LK tracking
    generate3dPoints(left_un_pts, right_un_pts, cur_frame.xyz, cur_frame.uv);
  }

  // Change key frame
  double rot_w = Quaterniond(c_R_k).w();
  if (rot_w > 1.0)
    rot_w = 1.0;
  if (rot_w < -1.0)
    rot_w = -1.0;
  double rot_angle = acos(rot_w) * 2.0;

  bool switch_key_frame = false;
  size_t tracked_cnt = key_pts_3d.size();
  size_t stereo_cnt = cur_frame.xyz.size();
  if (!init_finish)
  {
    switch_key_frame = true;
  }
  else
  {
    double key_age = (cur_frame.frame_time - key_frame.frame_time).toSec();
    bool motion_trigger = c_t_k.norm() > TRANSLATION_THRESHOLD || rot_angle > ROTATION_THRESHOLD;
    bool low_feature_trigger = tracked_cnt < static_cast<size_t>(FEATURE_THRESHOLD);
    bool critical_low_track = tracked_cnt < static_cast<size_t>(MIN_CNT);
    const double MIN_KEYFRAME_DT = 0.20;

    if (pose_estimated)
    {
      // Avoid keyframe chatter unless motion is significant or tracked features are critically low.
      if ((motion_trigger || low_feature_trigger) && (key_age > MIN_KEYFRAME_DT || critical_low_track))
      {
        switch_key_frame = true;
      }
    }
    else
    {
      // If pose fails for a few frames, re-bootstrap with current stereo points.
      if (fail_cnt > 3 && stereo_cnt >= static_cast<size_t>(MIN_CNT))
      {
        switch_key_frame = true;
      }
    }
  }

  ROS_INFO_THROTTLE(0.5, "VO stats tracked:%lu stereo:%lu pose:%d fail:%d",
                    static_cast<unsigned long>(tracked_cnt),
                    static_cast<unsigned long>(stereo_cnt),
                    pose_estimated,
                    fail_cnt);

  if (switch_key_frame)
  {
    key_frame = cur_frame;
    ROS_INFO("Change key frame to current frame.");
  }

  prev_frame = cur_frame;

  updateLatestStates(cur_frame);

  init_finish = true;

  return true;
}

bool Estimator::trackFeatureBetweenFrames(const Estimator::frame &keyframe, const cv::Mat &cur_img,
                                          vector<cv::Point3f> &key_pts_3d,
                                          vector<cv::Point2f> &cur_pts_2d)
{

  key_pts_3d.clear();
  cur_pts_2d.clear();

  if (keyframe.img.empty() || cur_img.empty() || keyframe.uv.empty() || keyframe.xyz.empty())
  {
    return false;
  }

  size_t pt_num = std::min(keyframe.uv.size(), keyframe.xyz.size());
  if (pt_num == 0)
  {
    return false;
  }

  vector<cv::Point2f> key_pts_2d(keyframe.uv.begin(), keyframe.uv.begin() + pt_num);
  key_pts_3d.assign(keyframe.xyz.begin(), keyframe.xyz.begin() + pt_num);
  cur_pts_2d = key_pts_2d;

  vector<uchar> status;
  vector<float> err;
  cv::calcOpticalFlowPyrLK(keyframe.img, cur_img, key_pts_2d, cur_pts_2d, status, err, cv::Size(21, 21), 3);

  if (FLOW_BACK)
  {
    vector<cv::Point2f> reverse_pts = key_pts_2d;
    vector<uchar> reverse_status;
    vector<float> reverse_err;
    cv::calcOpticalFlowPyrLK(cur_img, keyframe.img, cur_pts_2d, reverse_pts, reverse_status, reverse_err, cv::Size(21, 21), 3);

    for (size_t i = 0; i < status.size(); i++)
    {
      status[i] = status[i] && reverse_status[i] && distance(key_pts_2d[i], reverse_pts[i]) < 1.0;
    }
  }

  for (size_t i = 0; i < status.size(); i++)
  {
    if (status[i] && !inBorder(cur_pts_2d[i], cur_img.rows, cur_img.cols))
      status[i] = 0;
  }

  reduceVector<cv::Point3f>(key_pts_3d, status);
  reduceVector<cv::Point2f>(cur_pts_2d, status);

  return cur_pts_2d.size() >= static_cast<size_t>(MIN_CNT);
}

bool Estimator::estimateTBetweenFrames(vector<cv::Point3f> &key_pts_3d,
                                       vector<cv::Point2f> &cur_pts_2d, Matrix3d &R, Vector3d &t)
{

  R.setIdentity();
  t.setZero();

  if (key_pts_3d.size() < 4 || cur_pts_2d.size() < 4 || key_pts_3d.size() != cur_pts_2d.size())
  {
    return false;
  }

  cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
  cv::Mat rvec, tvec;
  vector<int> inliers;

  const double ransac_reproj_err = 2.0 / 380.0;
  bool pnp_ok = cv::solvePnPRansac(key_pts_3d, cur_pts_2d, K, cv::Mat(), rvec, tvec, false,
                                   100, ransac_reproj_err, 0.99, inliers, cv::SOLVEPNP_EPNP);

  if (!pnp_ok || inliers.size() < 4)
  {
    return false;
  }

  vector<uchar> status(key_pts_3d.size(), 0);
  for (size_t i = 0; i < inliers.size(); i++)
  {
    status[inliers[i]] = 1;
  }
  reduceVector<cv::Point3f>(key_pts_3d, status);
  reduceVector<cv::Point2f>(cur_pts_2d, status);

  if (key_pts_3d.size() < 4)
  {
    return false;
  }

  // Refine on inliers.
  cv::solvePnP(key_pts_3d, cur_pts_2d, K, cv::Mat(), rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);

  cv::Mat cv_R;
  cv::Rodrigues(rvec, cv_R);
  cv::cv2eigen(cv_R, R);
  cv::cv2eigen(tvec, t);

  // Additional reprojection filtering in normalized plane.
  vector<uchar> reproj_status(key_pts_3d.size(), 1);
  const double reproj_th = 3.0 / 380.0;
  for (size_t i = 0; i < key_pts_3d.size(); i++)
  {
    if (reprojectionError(R, t, key_pts_3d[i], cur_pts_2d[i]) > reproj_th)
      reproj_status[i] = 0;
  }

  reduceVector<cv::Point3f>(key_pts_3d, reproj_status);
  reduceVector<cv::Point2f>(cur_pts_2d, reproj_status);

  const size_t MIN_PNP_INLIERS = static_cast<size_t>(std::max(8, MIN_CNT / 2));
  if (key_pts_3d.size() < MIN_PNP_INLIERS)
  {
    return false;
  }

  cv::solvePnP(key_pts_3d, cur_pts_2d, K, cv::Mat(), rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);
  cv::Rodrigues(rvec, cv_R);
  cv::cv2eigen(cv_R, R);
  cv::cv2eigen(tvec, t);

  return true;
}

void Estimator::extractNewFeatures(const cv::Mat &img, vector<cv::Point2f> &uv)
{

  if (img.empty())
  {
    uv.clear();
    return;
  }

  if (static_cast<int>(uv.size()) >= MAX_CNT)
  {
    return;
  }

  cv::Mat mask(img.rows, img.cols, CV_8UC1, cv::Scalar(255));
  for (size_t i = 0; i < uv.size(); i++)
  {
    cv::circle(mask, uv[i], MIN_DIST, 0, -1);
  }

  vector<cv::Point2f> new_pts;
  int need_cnt = MAX_CNT - static_cast<int>(uv.size());
  cv::goodFeaturesToTrack(img, new_pts, need_cnt, 0.01, MIN_DIST, mask);

  if (!new_pts.empty())
  {
    cv::cornerSubPix(img, new_pts, cv::Size(5, 5), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.03));
  }

  for (size_t i = 0; i < new_pts.size(); i++)
  {
    if (inBorder(new_pts[i], img.rows, img.cols))
      uv.push_back(new_pts[i]);
  }
}

bool Estimator::trackFeatureLeftRight(const cv::Mat &_img, const cv::Mat &_img1,
                                      vector<cv::Point2f> &left_pts, vector<cv::Point2f> &right_pts)
{

  right_pts.clear();

  if (_img.empty() || _img1.empty() || left_pts.empty())
  {
    return false;
  }

  right_pts = left_pts;
  vector<uchar> status;
  vector<float> err;
  cv::calcOpticalFlowPyrLK(_img, _img1, left_pts, right_pts, status, err, cv::Size(21, 21), 3);

  if (FLOW_BACK)
  {
    vector<cv::Point2f> reverse_pts = left_pts;
    vector<uchar> reverse_status;
    vector<float> reverse_err;
    cv::calcOpticalFlowPyrLK(_img1, _img, right_pts, reverse_pts, reverse_status, reverse_err, cv::Size(21, 21), 3);

    for (size_t i = 0; i < status.size(); i++)
    {
      status[i] = status[i] && reverse_status[i] && distance(left_pts[i], reverse_pts[i]) < 1.0;
    }
  }

  for (size_t i = 0; i < status.size(); i++)
  {
    if (!status[i])
      continue;

    if (!inBorder(right_pts[i], _img1.rows, _img1.cols))
      status[i] = 0;

    if (fabs(left_pts[i].y - right_pts[i].y) > 5.0)
      status[i] = 0;
  }

  reduceVector<cv::Point2f>(left_pts, status);
  reduceVector<cv::Point2f>(right_pts, status);

  return !left_pts.empty();
}

void Estimator::generate3dPoints(const vector<cv::Point2f> &left_pts,
                                 const vector<cv::Point2f> &right_pts,
                                 vector<cv::Point3f> &cur_pts_3d,
                                 vector<cv::Point2f> &cur_pts_2d)
{

  Eigen::Matrix<double, 3, 4> P1, P2;

  P1 << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0;
  P2.block(0, 0, 3, 3) = (Tlr.block(0, 0, 3, 3).transpose());
  P2.block(0, 3, 3, 1) = -P2.block(0, 0, 3, 3) * Tlr.block(0, 3, 3, 1);

  vector<uchar> status;

  for (unsigned int i = 0; i < left_pts.size(); ++i)
  {
    Vector2d pl(left_pts[i].x, left_pts[i].y);
    Vector2d pr(right_pts[i].x, right_pts[i].y);
    Vector3d pt3;
    triangulatePoint(P1, P2, pl, pr, pt3);

    if (pt3[2] > 0)
    {
      cur_pts_3d.push_back(cv::Point3f(pt3[0], pt3[1], pt3[2]));
      status.push_back(1);
    }
    else
    {
      status.push_back(0);
    }
  }

  reduceVector<cv::Point2f>(cur_pts_2d, status);
}

bool Estimator::inBorder(const cv::Point2f &pt, const int &row, const int &col)
{
  const int BORDER_SIZE = 1;
  int img_x = cvRound(pt.x);
  int img_y = cvRound(pt.y);
  return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y &&
         img_y < row - BORDER_SIZE;
}

double Estimator::distance(cv::Point2f pt1, cv::Point2f pt2)
{
  double dx = pt1.x - pt2.x;
  double dy = pt1.y - pt2.y;
  return sqrt(dx * dx + dy * dy);
}

template <typename Derived>
void Estimator::reduceVector(vector<Derived> &v, vector<uchar> status)
{
  int j = 0;
  for (int i = 0; i < int(v.size()); i++)
    if (status[i])
      v[j++] = v[i];
  v.resize(j);
}

void Estimator::updateLatestStates(frame &latest_frame)
{

  latest_time = latest_frame.frame_time;
  rel_key_time = key_frame.frame_time;
  latest_pointcloud = latest_frame.xyz;

  // body_T_cam is provided; convert camera pose to body pose using cam_T_body = inverse(body_T_cam).
  Matrix3d R_cb = ric[0].transpose();
  Vector3d t_cb = -R_cb * tic[0];

  Matrix3d w_R_b = latest_frame.w_R_c * R_cb;
  Vector3d w_t_b = latest_frame.w_R_c * t_cb + latest_frame.w_t_c;

  latest_Q = Quaterniond(w_R_b);
  latest_Q.normalize();
  latest_P = w_t_b;

  Matrix3d w_R_bk = key_frame.w_R_c * R_cb;
  Vector3d w_t_bk = key_frame.w_R_c * t_cb + key_frame.w_t_c;

  // Relative pose of current body frame in key-body coordinates.
  Matrix3d k_R_b = w_R_bk.transpose() * w_R_b;
  Vector3d k_t_b = w_R_bk.transpose() * (w_t_b - w_t_bk);

  latest_rel_Q = Quaterniond(k_R_b);
  latest_rel_Q.normalize();
  latest_rel_P = k_t_b;
}

void Estimator::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                                 Eigen::Vector2d &point0, Eigen::Vector2d &point1,
                                 Eigen::Vector3d &point_3d)
{
  Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
  design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
  design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
  design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
  design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
  Eigen::Vector4d triangulated_point;
  triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
  point_3d(0) = triangulated_point(0) / triangulated_point(3);
  point_3d(1) = triangulated_point(1) / triangulated_point(3);
  point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

double Estimator::reprojectionError(Matrix3d &R, Vector3d &t, cv::Point3f &key_pts_3d, cv::Point2f &cur_pts_2d)
{
  Vector3d pt1(key_pts_3d.x, key_pts_3d.y, key_pts_3d.z);
  Vector3d pt2 = R * pt1 + t;
  pt2 = pt2 / pt2[2];
  return sqrt(pow(pt2[0] - cur_pts_2d.x, 2) + pow(pt2[1] - cur_pts_2d.y, 2));
}

vector<cv::Point2f> Estimator::undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
{
  vector<cv::Point2f> un_pts;
  for (unsigned int i = 0; i < pts.size(); i++)
  {
    Eigen::Vector2d a(pts[i].x, pts[i].y);
    Eigen::Vector3d b;
    cam->liftProjective(a, b);
    un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
  }
  return un_pts;
}
