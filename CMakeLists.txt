cmake_minimum_required(VERSION 3.8)
project(auto_car)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# ---- messages ----------------------------------------------------------------
rosidl_generate_interfaces(${PROJECT_NAME}
  msg/Track.msg
  msg/TrackArray.msg
  DEPENDENCIES std_msgs geometry_msgs)

# ---- C++ nodes -------------------------------------------------------------
add_executable(planning_node src/planning_node.cpp)
ament_target_dependencies(planning_node rclcpp std_msgs sensor_msgs)

add_executable(control_node src/control_node.cpp)
ament_target_dependencies(control_node rclcpp std_msgs sensor_msgs ackermann_msgs)

install(TARGETS planning_node control_node
  DESTINATION lib/${PROJECT_NAME})

# ---- Python ----------------------------------------------------------------
# Python module is auto_car_sim (not auto_car): rosidl already generates the 'auto_car'
# Python package for the messages, and two packages of the same name collide.
ament_python_install_package(auto_car_sim)   # auto_car_sim.vehicle_driver (Webots plugin)
ament_python_install_package(auto_car_core)  # simulator-agnostic geometry/safety/planning maths

install(PROGRAMS
  scripts/perception_node.py
  scripts/cloud_preprocess.py
  scripts/object_tracker.py
  scripts/safety_shield.py
  scripts/local_planner.py
  scripts/rl_planner.py
  scripts/driver_monitor.py
  scripts/road_perception.py
  scripts/object_detector.py
  scripts/route_planner.py
  scripts/traffic_rules.py
  scripts/telemetry_node.py
  scripts/data_collection_node.py
  rl/car_env.py
  rl/run_rl.py
  rl/train_rl.py
  DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS auto_car_sim/monitor_traffic.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME monitor_traffic)

# ---- Data ------------------------------------------------------------------
install(DIRECTORY launch worlds controllers urdf config models
  DESTINATION share/${PROJECT_NAME})

ament_package()
