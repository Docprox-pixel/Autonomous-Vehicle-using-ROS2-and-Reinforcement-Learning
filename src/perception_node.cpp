#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <opencv2/opencv.hpp>

/**
 * 👁️ PERCEPTION NODE (V3: Scene Awareness)
 * Features: Lane segmentation + Yellow crosswalk/intersection detection.
 */
class PerceptionNode : public rclcpp::Node {
public:
    PerceptionNode() : Node("perception_node") {
        this->declare_parameter("h_min", 0);
        this->declare_parameter("s_min", 0);
        this->declare_parameter("v_min", 180);

        sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/front_camera/image_raw", 10,
            std::bind(&PerceptionNode::callback, this, std::placeholders::_1));

        pub_ = create_publisher<sensor_msgs::msg::Image>("/lane_mask", 10);
        scene_pub_ = create_publisher<std_msgs::msg::Bool>("/is_intersection", 10);
        lane_err_pub_ = create_publisher<std_msgs::msg::Float32>("/lane_error", 10);
        
        RCLCPP_INFO(this->get_logger(), "Perception Node: Scene Awareness active.");
    }

private:
    void callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv::Mat frame;
        try {
            frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (...) { return; }

        int h = frame.rows;

        // 1. Preprocessing: ROI (Focus on road)
        cv::Mat roi = frame(cv::Range(h * 0.5, h), cv::Range::all());
        
        // 2. Segmentation (HSV) - Ready to be swapped for YOLOv8/UNet mask
        cv::Mat hsv, mask_white, mask_yellow, lane_mask;
        cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

        int hm = get_parameter("h_min").as_int();
        int sm = get_parameter("s_min").as_int();
        int vm = get_parameter("v_min").as_int();

        cv::inRange(hsv, cv::Scalar(hm, sm, vm), cv::Scalar(180, 50, 255), mask_white);
        cv::inRange(hsv, cv::Scalar(15, 80, 80), cv::Scalar(40, 255, 255), mask_yellow);
        lane_mask = mask_white | mask_yellow;

        // 3. Morphological Cleanup (Stable geometry)
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(lane_mask, lane_mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(lane_mask, lane_mask, cv::MORPH_CLOSE, kernel);

        // 4. Scene Awareness (Intersection Detection)
        std_msgs::msg::Bool is_intersection;
        is_intersection.data = (cv::countNonZero(mask_yellow) > 500); 
        scene_pub_->publish(is_intersection);

        // 4.5 Calculate Lane Error using Moments (focused on bottom 40% of ROI)
        cv::Mat bottom_roi = lane_mask(cv::Range(lane_mask.rows * 0.6, lane_mask.rows), cv::Range::all());
        cv::Moments m = cv::moments(bottom_roi, true);
        
        float lane_error = 0.0;
        if (m.m00 > 100) {
            float cx = m.m10 / m.m00;
            // Normalized error from center. width is frame.cols
            lane_error = (cx - (frame.cols / 2.0)) / (frame.cols / 2.0); 
        }
        
        std_msgs::msg::Float32 err_msg;
        err_msg.data = lane_error;
        lane_err_pub_->publish(err_msg);

        // 5. Output
        auto out_msg = cv_bridge::CvImage(msg->header, "mono8", lane_mask).toImageMsg();
        pub_->publish(*out_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr scene_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr lane_err_pub_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PerceptionNode>());
    rclcpp::shutdown();
    return 0;
}
