// control_node: lateral/longitudinal controller + first version of the safety shield.
//
// Inputs : /lane_error (Float32, normalized image offset, +right), /target_speed, /target_steering
//          /ml/steering (optional learned steering), /scan (LaserScan, contract frame lidar_link)
// Output : /cmd_drive_raw (ackermann_msgs/AckermannDriveStamped, m/s, rad, +steering = LEFT, REP-103)
//          -> scripts/safety_shield.py -> /cmd_drive. The shield has the final word on speed.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/float32.hpp>

using std::placeholders::_1;
using Float32 = std_msgs::msg::Float32;

class ControlNode : public rclcpp::Node {
public:
  ControlNode() : Node("control_node") {
    k_stanley_ = declare_parameter("k_stanley", 3.5);
    v_max_ = declare_parameter("v_max", 20.0);
    max_steer_ = declare_parameter("max_steer", 0.45);
    steer_rate_ = declare_parameter("max_steer_step", 0.03);      // rad per control cycle
    ml_weight_ = declare_parameter("ml_weight", 0.6);
    ml_timeout_ = declare_parameter("ml_timeout", 0.5);           // s
    stop_dist_ = declare_parameter("stop_distance", 6.0);         // m, at standstill
    stop_time_gap_ = declare_parameter("stop_time_gap", 0.8);     // s, extra distance per m/s
    front_half_angle_ = declare_parameter("front_half_angle", 0.35);  // rad (~20 deg)

    sub_lane_ = create_subscription<Float32>("/lane_error", 10,
                                             [this](Float32::SharedPtr m) { lane_error_ = m->data; });
    sub_v_ = create_subscription<Float32>("/target_speed", 10,
                                          [this](Float32::SharedPtr m) { target_v_ = m->data; });
    sub_s_ = create_subscription<Float32>("/target_steering", 10,
                                          [this](Float32::SharedPtr m) { target_s_ = m->data; });
    sub_ml_ = create_subscription<Float32>("/ml/steering", 10, std::bind(&ControlNode::on_ml, this, _1));
    sub_scan_ = create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan", rclcpp::SensorDataQoS(), std::bind(&ControlNode::on_scan, this, _1));

    pub_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/cmd_drive_raw", 10);
    timer_ = create_wall_timer(std::chrono::milliseconds(33), std::bind(&ControlNode::loop, this));
    RCLCPP_INFO(get_logger(), "control_node: Stanley + shield active (k=%.2f, v_max=%.1f)", k_stanley_, v_max_);
  }

private:
  void on_ml(Float32::SharedPtr m) {
    ml_steer_ = m->data;
    last_ml_ = now().seconds();
    ml_active_ = true;
  }

  // Minimum range inside the forward cone only (the old code used all 360 degrees,
  // so anything beside or behind the car could freeze it).
  void on_scan(sensor_msgs::msg::LaserScan::SharedPtr s) {
    float best = std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < s->ranges.size(); ++i) {
      const double a = s->angle_min + i * s->angle_increment;
      if (std::fabs(std::remainder(a, 2 * M_PI)) > front_half_angle_) continue;
      const float r = s->ranges[i];
      if (std::isfinite(r) && r > s->range_min) best = std::min(best, r);
    }
    if (!std::isfinite(best)) best = s->range_max;
    front_dist_ = 0.7f * front_dist_ + 0.3f * best;
  }

  void loop() {
    // 1. Stanley-style lateral control. lane_error > 0 means the lane centre is to the
    //    right in the image, so we must steer right = negative angle (REP-103).
    const double v = std::max(0.5, target_v_);
    double steer = target_s_ - std::atan2(k_stanley_ * lane_error_, v);

    // 2. Optional learned steering with heartbeat.
    if (ml_active_ && now().seconds() - last_ml_ < ml_timeout_) {
      steer = ml_weight_ * ml_steer_ + (1.0 - ml_weight_) * steer;
    } else if (ml_active_) {
      ml_active_ = false;
      RCLCPP_WARN(get_logger(), "ML steering timeout: Stanley only");
    }

    // 3. Safety shield: braking distance grows with speed.
    double v_cmd = std::clamp(target_v_, 0.0, v_max_);
    const double needed = stop_dist_ + stop_time_gap_ * speed_;
    if (front_dist_ < needed) {
      v_cmd = 0.0;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "SHIELD: obstacle %.1f m ahead -> stop", front_dist_);
    }

    // 4. Smoothing + limits.
    speed_ = 0.9 * speed_ + 0.1 * v_cmd;
    if (v_cmd == 0.0 && speed_ < 0.05) speed_ = 0.0;
    steer_ += std::clamp(steer - steer_, -steer_rate_, steer_rate_);
    steer_ = std::clamp(steer_, -max_steer_, max_steer_);

    ackermann_msgs::msg::AckermannDriveStamped cmd;
    cmd.header.stamp = now();
    cmd.header.frame_id = "base_link";
    cmd.drive.speed = static_cast<float>(speed_);
    cmd.drive.steering_angle = static_cast<float>(steer_);
    pub_->publish(cmd);
  }

  double k_stanley_, v_max_, max_steer_, steer_rate_, ml_weight_, ml_timeout_;
  double stop_dist_, stop_time_gap_, front_half_angle_;
  double lane_error_ = 0.0, target_v_ = 0.0, target_s_ = 0.0, ml_steer_ = 0.0, last_ml_ = 0.0;
  double speed_ = 0.0, steer_ = 0.0;
  float front_dist_ = 100.0f;
  bool ml_active_ = false;

  rclcpp::Subscription<Float32>::SharedPtr sub_lane_, sub_v_, sub_s_, sub_ml_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_scan_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ControlNode>());
  rclcpp::shutdown();
  return 0;
}
