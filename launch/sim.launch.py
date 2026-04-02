import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController


def generate_launch_description():
    package_dir = os.path.dirname(os.path.dirname(__file__))

    world = os.path.join(package_dir, 'worlds', 'city.wbt')
    robot_description_path = os.path.join(package_dir, 'urdf', 'vehicle.urdf')

    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='modular',
        description='Mode: modular stack'
    )

    webots = WebotsLauncher(world=world)

    vehicle = WebotsController(
        robot_name='vehicle',
        parameters=[
            {'robot_description': robot_description_path},
            {'robot_type': 'Driver'},
            {'use_sim_time': True},
        ]
    )

    # Tesla-like production stack
    perception = Node(
        package='auto_car',
        executable='perception_node.py',
        output='screen'
    )

    planning = Node(
        package='auto_car',
        executable='planning_node',
        output='screen'
    )

    control = Node(
        package='auto_car',
        executable='control_node',
        output='screen',
        parameters=[{'k_stanley': 0.8, 'v_max': 20.0}]
    )

    return LaunchDescription([
        mode_arg,
        webots,
        vehicle,
        perception,
        planning,
        control,
    ])