#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <string>
#include <vector>

/**
 * 🗺️ TESLA-STYLE BEHAVIORAL PLANNER
 * States: NORMAL, APPROACH, WAIT, DECIDE, TURNING.
 * Logic: Deliberates for 10 frames at intersections.
 */

enum class DriveState { NORMAL, APPROACH_INTERSECTION, WAIT, DECIDE, TURNING, AVOID_OBSTACLE };

class PlanningNode : public rclcpp::Node {
public:
    PlanningNode() : Node("planning_node") {
        // Subscribers
        sub_inter_ = create_subscription<std_msgs::msg::Bool>(
            "/intersection_flag", 10, [this](const std_msgs::msg::Bool::SharedPtr msg){ intersection_flag_ = msg->data; });
            
        sub_lidar_ = create_subscription<sensor_msgs::msg::LaserScan>(
            "/lidar/scan", 10, std::bind(&PlanningNode::lidar_callback, this, std::placeholders::_1));

        // Publishers
        pub_state_ = create_publisher<std_msgs::msg::String>("/drive_state", 10);
        pub_steer_ = create_publisher<std_msgs::msg::Float32>("/target_steering", 10);
        pub_speed_ = create_publisher<std_msgs::msg::Float32>("/target_speed", 10);

        // State Init
        state_ = DriveState::NORMAL;
        state_counter_ = 0;
        current_speed_ = 3.0; 
        current_steering_ = 0.0;
        
        timer_ = create_timer(std::chrono::milliseconds(50), std::bind(&PlanningNode::planning_loop, this));
        RCLCPP_INFO(this->get_logger(), "Tesla Planner: Behavioral Engine Active.");
    }

private:
    void planning_loop() {
        double target_v = 15.0;
        double target_s = 0.0;
        std::string state_str = "NORMAL";

        switch (state_) {
            case DriveState::NORMAL:
                target_v = 16.66;
                target_s = 0.0; // Kinematic baseline: Straight
                if (intersection_flag_) {
                    state_ = DriveState::APPROACH_INTERSECTION;
                    state_counter_ = 0;
                } else if (obstacle_distance_ < 28.0) {
                    state_ = DriveState::AVOID_OBSTACLE;
                    state_counter_ = 0;
                }
                state_str = "NORMAL";
                break;

            case DriveState::APPROACH_INTERSECTION:
                target_v = 5.0; 
                target_s = 0.0;
                state_counter_++;
                if (state_counter_ > 20) { // Near enough to stop
                    state_ = DriveState::WAIT;
                    state_counter_ = 0;
                }
                state_str = "APPROACH";
                break;

            case DriveState::WAIT:
                target_v = 0.0; // Deliberate
                target_s = 0.0;
                state_counter_++;
                if (state_counter_ >= 10) { 
                    state_ = DriveState::DECIDE;
                    state_counter_ = 0;
                }
                state_str = "WAIT";
                break;

            case DriveState::DECIDE:
                target_v = 0.0;
                // Heuristic: Always turn left for this demo pathing
                state_ = DriveState::TURNING;
                state_counter_ = 0;
                state_str = "DECIDE";
                break;

            case DriveState::TURNING:
                target_v = 8.0;
                target_s = 0.45; // Hard left kinematic turn
                state_counter_++;
                if (state_counter_ > 60) { // Exit turn after 3s
                    state_ = DriveState::NORMAL;
                    state_counter_ = 0;
                }
                state_str = "TURNING";
                break;

            case DriveState::AVOID_OBSTACLE:
                if (state_counter_ < 10) {
                    target_v = 2.0;
                    target_s = 0.0;
                    state_str = "AVOID (BRAKE)";
                } else if (state_counter_ < 35) {
                    target_v = 5.0;
                    target_s = 0.45;
                    state_str = "AVOID (SWERVE LEFT)";
                } else if (state_counter_ < 60) {
                    target_v = 5.0;
                    target_s = -0.45;
                    state_str = "AVOID (RECOVER RIGHT)";
                } else {
                    state_ = DriveState::NORMAL;
                    state_counter_ = 0;
                }
                state_counter_++;
                break;
        }

        // --- Kinematic Smoothing ---
        current_speed_ = (0.8 * current_speed_) + (0.2 * target_v);
        current_steering_ = (0.8 * current_steering_) + (0.2 * target_s);

        // --- Publish Outputs ---
        auto msg_state = std_msgs::msg::String(); msg_state.data = state_str;
        pub_state_->publish(msg_state);

        auto msg_v = std_msgs::msg::Float32(); msg_v.data = current_speed_;
        auto msg_s = std_msgs::msg::Float32(); msg_s.data = current_steering_;
        pub_speed_->publish(msg_v);
        pub_steer_->publish(msg_s);
    }

    void lidar_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        float raw_min = 100.0;
        int sz = msg->ranges.size();
        if (sz == 0) return;
        
        // Scan the front approx 30 degrees (15 left, 15 right)
        int center_idx = sz / 2;
        int spread = sz / 12; 
        
        for (int i = center_idx - spread; i <= center_idx + spread; i++) {
            int idx = i;
            if (idx < 0) idx += sz;
            if (idx >= sz) idx -= sz;
            if (msg->ranges[idx] < raw_min && msg->ranges[idx] > 0.1) {
                raw_min = msg->ranges[idx];
            }
        }
        obstacle_distance_ = (0.8 * obstacle_distance_) + (0.2 * raw_min);
    }

    DriveState state_;
    int state_counter_;
    bool intersection_flag_ = false;
    double current_speed_, current_steering_;
    float obstacle_distance_ = 30.0;

    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_inter_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_lidar_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_state_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pub_steer_, pub_speed_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PlanningNode>());
    rclcpp::shutdown();
    return 0;
}
