#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>

/**
 * 🏎️ THE ULTIMATE UNIFIED CONTROLLER
 * Includes: 360° LiDAR, Phase-1 Lane Following, Traffic Light Detection, and High-Speed Tuning.
 */

class CarController : public rclcpp::Node
{
public:
    CarController() : Node("car_controller")
    {
        camera_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/front_camera/image_raw", 10,
            std::bind(&CarController::imageCallback, this, std::placeholders::_1));

        lidar_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
            "/lidar/scan", 10,
            std::bind(&CarController::lidarCallback, this, std::placeholders::_1));

        cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

        RCLCPP_INFO(this->get_logger(), "Ultimate Unified Controller Started");
    }

private:
    void lidarCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        auto& ranges = msg->ranges;
        if (ranges.empty()) return;

        int n = ranges.size();
        
        // --- 1. 360-Degree Sector Analysis ---
        // Front (±30°), Left (60°-120°), Right (240°-300°), Rear (150°-210°)
        // Assuming 0 is Behind, 180 is Front (as per Webots Lidar defaults)
        
        auto get_min = [&](int start_deg, int end_deg) {
            float m = 30.0;
            int s = n * start_deg / 360;
            int e = n * end_deg / 360;
            for (int i = s; i < e; i++) {
                if (ranges[i] < m) m = ranges[i];
            }
            return m;
        };

        front_dist_ = get_min(150, 210); // Front cone
        left_dist_  = get_min(210, 270); // Left side
        right_dist_ = get_min(90, 150);  // Right side
        rear_dist_  = std::min(get_min(0, 30), get_min(330, 360)); // Behind

        // Safety Flag: Emergency stop if anything is too close in ANY direction
        is_collision_risk_ = (front_dist_ < 0.8 || left_dist_ < 0.5 || right_dist_ < 0.5 || rear_dist_ < 0.5);
    }

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        cv::Mat frame;
        try {
            frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (...) { return; }

        int height = frame.rows;
        int width  = frame.cols;

        // --- 1. Perception (HSV) ---
        cv::Mat hsv, lane_mask, red_mask;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        // Advanced Lane Masking
        cv::Mat m1, m2;
        cv::inRange(hsv, cv::Scalar(0, 0, 180),  cv::Scalar(180, 50, 255), m1); // White
        cv::inRange(hsv, cv::Scalar(15, 80, 80), cv::Scalar(40, 255, 255), m2); // Yellow
        lane_mask = m1 | m2;

        // Traffic Light Detection (Strictly Upper half)
        cv::inRange(hsv, cv::Scalar(0, 120, 100), cv::Scalar(10, 255, 255), red_mask);
        int red_pixels = cv::countNonZero(red_mask(cv::Range(0, height/2), cv::Range::all()));

        // --- 2. Planning (Lane Centering) ---
        // We look at the lower ROI for immediate steering
        cv::Mat roi = lane_mask(cv::Range(height * 0.6, height), cv::Range::all());
        cv::Moments mo = cv::moments(roi, true);
        
        double steering = 0.0;
        if (mo.m00 > 0) {
            double target_x = mo.m10 / mo.m00;
            double error = (width / 2.0) - target_x;

            // PID Tuning for High Speed
            double kp = 0.007;
            double kd = 0.015;
            steering = (kp * error) + (kd * (error - last_error_));
            last_error_ = error;
            
            // Limit steering to prevent over-correction at 40m/s
            steering = std::max(std::min(steering, 0.4), -0.4);
        }

        // --- 3. Speed Control ---
        float target_speed = 40.0; // User target: 40 m/s

        // Smooth Brake for Obstacles
        if (front_dist_ < 20.0) {
            target_speed *= (front_dist_ - 2.0) / 18.0; 
        }

        // Stop for Red Light
        if (red_pixels > 150) {
            target_speed = 0.0;
        }

        // Emergency Stop (360° Risk)
        if (is_collision_risk_) {
            target_speed = 0.0;
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "EMERGENCY STOP: Obstacle too close!");
        }

        if (target_speed < 0) target_speed = 0;

        // --- 4. Actuation ---
        geometry_msgs::msg::Twist cmd;
        cmd.linear.x = target_speed;
        cmd.angular.z = steering;
        cmd_pub_->publish(cmd);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr lidar_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr   cmd_pub_;

    float front_dist_ = 30.0;
    float left_dist_  = 30.0;
    float right_dist_ = 30.0;
    float rear_dist_  = 30.0;
    bool  is_collision_risk_ = false;
    double last_error_ = 0.0;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CarController>();
    rclcpp::spin(node);
    rclcpp::shutdown();
}
