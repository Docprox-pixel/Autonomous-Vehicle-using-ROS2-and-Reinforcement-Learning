#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <cmath>
#include <algorithm>

/**
 * 🏎️ TESLA-STYLE STANLEY CONTROL NODE
 * 1. Implementation: Stanley Controller (delta = psi + atan(k*e/v))
 * 2. Integration: 0.6*ML + 0.4*Stanley 
 * 3. Goal: Smooth, high-fidelity lane centering.
 */

class ControlNode : public rclcpp::Node
{
public:
    ControlNode() : Node("control_node")
    {
        // Subscribers from Perception & Planning
        sub_lane_err_ = create_subscription<std_msgs::msg::Float32>(
            "/lane_error", 10, [this](const std_msgs::msg::Float32::SharedPtr msg){ this->lane_error_ = msg->data; });
            
        sub_target_v_ = create_subscription<std_msgs::msg::Float32>(
            "/target_speed", 10, [this](const std_msgs::msg::Float32::SharedPtr msg){ this->target_v_ = msg->data; });

        sub_target_s_ = create_subscription<std_msgs::msg::Float32>(
            "/target_steering", 10, [this](const std_msgs::msg::Float32::SharedPtr msg){ this->target_s_ = msg->data; });

        sub_ml_ = create_subscription<std_msgs::msg::Float32>(
            "/ml/steering", 10, std::bind(&ControlNode::ml_cb, this, std::placeholders::_1));

        lidar_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
            "/lidar/scan", 10, std::bind(&ControlNode::lidarCallback, this, std::placeholders::_1));

        pub_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
        
        // Init state
        lane_error_ = 0.0; target_v_ = 0.0; target_s_ = 0.0;
        ml_steer_ = 0.0;
        prev_steer_ = 0.0; prev_speed_ = 3.0;
        filtered_dist_ = 30.0;
        k_gain_ = 3.5; // Stanley gain
        last_ml_stamp_ = 0.0;
        ml_active_ = false;
        
        timer_ = create_timer(std::chrono::milliseconds(33), std::bind(&ControlNode::control_loop, this));
        RCLCPP_INFO(this->get_logger(), "Tesla Control: Stanley Engine Active.");
    }

private:
    void ml_cb(const std_msgs::msg::Float32::SharedPtr msg) {
        ml_steer_ = msg->data;
        last_ml_stamp_ = this->get_clock()->now().seconds();
        ml_active_ = true;
    }

    void lidarCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        auto& ranges = msg->ranges;
        if (ranges.empty()) return;
        float raw_min = 30.0;
        for (auto r : ranges) if (r < raw_min && r > 0.1) raw_min = r;
        filtered_dist_ = (0.85 * filtered_dist_) + (0.15 * raw_min);
    }

    void control_loop() {
        // 1. Stanley Algorithm Implementation
        double current_v = std::max(0.5, target_v_); 
        double stanley_cross_track = std::atan2(k_gain_ * lane_error_, current_v);
        double stanley_steer = target_s_ + stanley_cross_track;

        // 2. Hybrid ML Blending with Robust Heartbeat
        double blended_steer = stanley_steer;
        double now_stamp = this->get_clock()->now().seconds();
        
        // Timeout check (0.5s) using double timestamps to avoid clock type mismatch
        if (ml_active_ && (now_stamp - last_ml_stamp_) < 0.5) {
            blended_steer = (0.6 * ml_steer_) + (0.4 * stanley_steer);
        } else if (ml_active_) {
            ml_active_ = false; // Heartbeat lost
            RCLCPP_WARN(this->get_logger(), "ML Signal Timeout: Reverting to 100%% Stanley.");
        }

        // 3. Safety & Smoothing
        if (filtered_dist_ < 6.0) target_v_ = 0.0;
        prev_speed_ = (0.9 * prev_speed_) + (0.1 * target_v_);
        
        double diff = blended_steer - prev_steer_;
        double limit = 0.03;
        prev_steer_ += std::max(std::min(diff, limit), -limit);
        prev_steer_ = std::max(std::min(prev_steer_, 0.45), -0.45);

        // 4. Output
        geometry_msgs::msg::Twist cmd;
        cmd.linear.x = std::max(0.0, prev_speed_);
        cmd.angular.z = prev_steer_;
        pub_->publish(cmd);
    }

    double lane_error_, target_v_, target_s_, ml_steer_;
    double prev_steer_, prev_speed_, k_gain_;
    double last_ml_stamp_;
    bool ml_active_;
    float filtered_dist_;

    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_lane_err_, sub_target_v_, sub_target_s_, sub_ml_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr lidar_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ControlNode>());
    rclcpp::shutdown();
    return 0;
}
