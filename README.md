# Auto Car: Advanced Autonomous Driving Framework

`auto_car` is a robust ROS 2 (Jazzy) autonomous vehicle framework built for the Webots simulator. Designed around real-world Tesla-like control mechanics, it fuses traditional mathematical steering controllers with cutting-edge Machine Learning and Reinforcement Learning pipelines.

## 🚀 Core Features
*   **Computer Vision (YOLOv8 + HSV):** Real-time lane tracking using dynamic color segmentation, merged with YOLOv8 inference to track bounding boxes of vehicles and pedestrians.
*   **3D LiDAR Obstacle Avoidance:** Utilizes a custom state machine tied to a 16-layer volumetric LiDAR scanner. When a frontal collision is predicted, the car automatically transfers weight to the front brakes, swerves aggressively to change lanes, recovers lateral momentum, and resumes standard operations.
*   **Stanley Control Kinematics:** Math-perfect lane-centering logic tuned for high-speed stability at speeds up to 60 km/h (16.66 m/s), preventing understeer.
*   **Machine Learning Blending Engine:** Live fusion of Reinforcement Learning (PPO) outputs. Native ROS 2 `control_node` actively blends `(0.6 * ML + 0.4 * Stanley)` to merge learned obstacle avoidance reflexes with mathematically perfect lane-centering logic seamlessly.
*   **Live Telemetry Dashboard:** A built-in terminal dashboard intercepting the vision pipeline and propulsion veins to display speed, steering angles, lane offsets, and YOLO classifications continuously without log flooding.

## 📁 Repository Structure
*   **`/auto_car`**: Native ROS 2 Python nodes (Data visualizers).
*   **`/src`**: Core C++ control architectures (`control_node.cpp`, `planning_node.cpp`).
*   **`/scripts`**: Neural network hubs and perception nodes bridging YOLOv8 to the ROS 2 space.
*   **`/rl`**: Gymnasium Reinforcement Learning environment (`car_env.py`) and pre-trained PPO agent scripts.
*   **`/launch`**: Global initialization mapping (`sim.launch.py`).
*   **`/worlds`**: Webots custom physics configurations (`city.wbt`).

## 🛠️ Usage
1. Compile the workspace:
    ```bash
    cd auto_ws
    colcon build --packages-select auto_car
    source install/setup.bash
    ```
2. Spawn the global simulation (Webots + Perception + Planning):
    ```bash
    ros2 launch auto_car sim.launch.py
    ```
3. Activate the Visual Interface:
    ```bash
    ros2 run auto_car monitor_traffic
    ```
4. Activate the Live Telemetry Console:
    ```bash
    ros2 run auto_car telemetry_node.py
    ```
5. *(Optional)* Let the Reinforcement Learning AI take over:
    ```bash
    ros2 run auto_car run_rl.py
    ```
