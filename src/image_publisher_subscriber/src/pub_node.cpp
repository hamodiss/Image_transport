#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class ImagePublisher : public rclcpp::Node {
public:
    ImagePublisher() : Node("image_publisher") {
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("image_topic", 10);
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100), std::bind(&ImagePublisher::publish_image, this));
    }

private:
    void publish_image() {
        cv::Mat img = cv::imread("/home/momo/image_transport/abc.jpg");  // Path to your image
        if (img.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Could not open or find the image");
            return;
        }

        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", img).toImageMsg();
        publisher_->publish(*msg);
        RCLCPP_INFO(this->get_logger(), "Image published");
    }

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImagePublisher>());
    rclcpp::shutdown();
    return 0;
}

