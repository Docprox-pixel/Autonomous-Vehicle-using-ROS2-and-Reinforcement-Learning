cmake_minimum_required(VERSION 3.8)
project(auto_car)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# Modular Production Stack
add_executable(perception_node src/perception_node.cpp)
add_executable(planning_node   src/planning_node.cpp)
add_executable(control_node    src/control_node.cpp)

ament_target_dependencies(perception_node rclcpp sensor_msgs cv_bridge)
target_link_libraries(perception_node ${OpenCV_LIBS})

ament_target_dependencies(planning_node rclcpp sensor_msgs geometry_msgs cv_bridge)
target_link_libraries(planning_node ${OpenCV_LIBS})

ament_target_dependencies(control_node rclcpp geometry_msgs sensor_msgs)

install(TARGETS
  perception_node
  planning_node
  control_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch worlds urdf
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/data_collection_node.py
  scripts/inference_node.py
  scripts/training_script.py
  scripts/perception_node.py
  scripts/perception_v8_node.py
  scripts/telemetry_node.py
  rl/run_rl.py
  rl/car_env.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  rl/bmw_rl_driver.zip
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  auto_car/monitor_traffic.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME monitor_traffic
)

ament_python_install_package(${PROJECT_NAME})
ament_package()
