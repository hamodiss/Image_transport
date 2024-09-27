cmake_minimum_required(VERSION 3.5)
project(image_publisher_subscriber)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Find ament_cmake and other dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(include)

# Add the publisher executable
add_executable(pub_node src/pub_node.cpp)
ament_target_dependencies(pub_node rclcpp sensor_msgs cv_bridge OpenCV)

# Add the subscriber executable
add_executable(sub_node src/sub_node.cpp)
ament_target_dependencies(sub_node rclcpp sensor_msgs cv_bridge OpenCV)

# Install targets
install(TARGETS
  pub_node
  sub_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()

