"""Webots simulation + localisation + current driving stack.

  ros2 launch auto_car sim.launch.py                            # city.wbt, full stack
  ros2 launch auto_car sim.launch.py world:=city_traffic.wbt    # any world in share/auto_car/worlds
  ros2 launch auto_car sim.launch.py stack:=false               # everything except the planner (car parked)
  ros2 launch auto_car sim.launch.py slam:=true                 # also build 2D (SLAM) + 3D (OctoMap) maps
  ros2 launch auto_car sim.launch.py rviz:=true                 # open RViz (map view if slam:=true)

Always started: Webots, vehicle bridge, EKF, 3D perception, object tracker, safety shield.
stack:=true (default) adds the local planner, which drives: it follows the road and takes turns;
give it a destination with RViz "2D Goal Pose" (/goal_pose).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, LogInfo, OpaqueFunction, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown, matches_action
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition
from webots_ros2_driver.webots_controller import WebotsController
from webots_ros2_driver.webots_launcher import WebotsLauncher


def generate_launch_description():
    pkg = get_package_share_directory('auto_car')
    cfg = os.path.join(pkg, 'config')
    # Must be the launch argument literally named 'world', used as the LAST element of a
    # PathJoinSubstitution: WebotsLauncher swaps it for its modified copy of the world
    # (the copy is what contains the Ros2Supervisor robot that publishes /clock).
    world = LaunchConfiguration('world')
    stack = LaunchConfiguration('stack')
    slam = LaunchConfiguration('slam')
    rviz = LaunchConfiguration('rviz')
    sim_time = {'use_sim_time': True}

    webots = WebotsLauncher(
        world=PathJoinSubstitution([pkg, 'worlds', world]),
        # The ROS 2 supervisor publishes /clock. Every node uses sim time, so without it
        # anything with a timer (EKF, SLAM, Nav2) waits forever for the clock to start.
        ros2_supervisor=True)

    vehicle = WebotsController(
        robot_name='vehicle',
        parameters=[{'robot_description': os.path.join(pkg, 'urdf', 'vehicle.urdf')}, sim_time],
        # Webots can take >50 s to load a world the first time (PROTO downloads),
        # after which the controller gives up; respawn makes it keep retrying.
        respawn=True)

    # ---- localisation (Phase 1) ---------------------------------------------
    ekf = Node(package='robot_localization', executable='ekf_node', name='ekf_filter_node',
               output='screen', parameters=[os.path.join(cfg, 'ekf.yaml')])

    # In Jazzy slam_toolbox is a lifecycle node: it starts 'unconfigured' and publishes
    # nothing until it is configured and then activated (same as its own launch file).
    slam_node = LifecycleNode(package='slam_toolbox', executable='async_slam_toolbox_node',
                              name='slam_toolbox', namespace='', output='screen',
                              parameters=[os.path.join(cfg, 'slam.yaml'), {'use_lifecycle_manager': False}],
                              condition=IfCondition(slam))
    slam_configure = EmitEvent(
        event=ChangeState(lifecycle_node_matcher=matches_action(slam_node),
                          transition_id=Transition.TRANSITION_CONFIGURE),
        condition=IfCondition(slam))
    slam_activate = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=slam_node, start_state='configuring', goal_state='inactive',
            entities=[LogInfo(msg='slam_toolbox configured -> activating'),
                      EmitEvent(event=ChangeState(lifecycle_node_matcher=matches_action(slam_node),
                                                  transition_id=Transition.TRANSITION_ACTIVATE))]),
        condition=IfCondition(slam))

    # ---- 3D perception + 3D map ---------------------------------------------
    cloud_pre = Node(package='auto_car', executable='cloud_preprocess.py', output='screen',
                     parameters=[sim_time])
    rules = Node(package='auto_car', executable='traffic_rules.py', output='screen', parameters=[sim_time])
    detect = LaunchConfiguration('detector')
    detector = Node(package='auto_car', executable='object_detector.py', output='screen',
                    parameters=[sim_time, {'model': PathJoinSubstitution([pkg, 'models', detect])}],
                    condition=IfCondition(PythonExpression(["'", detect, "' != 'none'"])))
    road = Node(package='auto_car', executable='road_perception.py', output='screen', parameters=[sim_time])
    # Lane map: worlds/<world name>_net/sumo.net.xml (Webots ships one per city world).
    # Resolved in an OpaqueFunction placed BEFORE WebotsLauncher in the description: once
    # WebotsLauncher executes it replaces the 'world' configuration with its temporary copy's path.
    def make_route(context):
        name = context.launch_configurations['world'].replace('.wbt', '')
        net = os.path.join(pkg, 'worlds', f'{name}_net', 'sumo.net.xml')
        actions = [Node(package='auto_car', executable='route_planner.py', output='screen',
                        parameters=[sim_time, {'net_file': net}])]
        # Our own traffic-light controllers (controllers/<name>/<name>.py) run as <extern>
        # controllers: WebotsLauncher loads a COPY of the world from /tmp, where Webots cannot find
        # project controllers. Each light robot with controller "<extern>" is started here by name.
        text = open(os.path.join(pkg, 'worlds', f'{name}.wbt')).read()
        import re
        for m in re.finditer(r'CrossRoadsTrafficLight \{(?:(?!\n\}).)*?controller "<extern>"', text, re.S):
            block = m.group(0)
            nm = re.search(r'name "([^"]+)"', block)
            robot = nm.group(1) if nm else 'cross road traffic light'
            ctrl = os.path.join(pkg, 'controllers', 'crossroads_traffic_lights_10s', 'crossroads_traffic_lights_10s.py')
            actions.append(ExecuteProcess(cmd=[os.path.join(os.environ.get('WEBOTS_HOME', ''), 'webots-controller'),
                                               '--protocol=ipc', f'--robot-name={robot}', ctrl],
                                          output='screen', name='traffic_light_controller'))
        # SUMO traffic: our patched supervisor (controllers/sumo_supervisor, heights from the 3-D
        # lane shapes so SUMO cars drive ON bridges and ramps) as the <extern> controller of the
        # world's SumoInterface (externController TRUE). Arguments as the PROTO would build them.
        si = re.search(r'(?ms)^SumoInterface \{\n(.*?)^\}', text)
        if si and re.search(r'(?m)^  externController TRUE', si.group(1)):
            b = si.group(1)
            fld = lambda k, d: (re.search(rf'(?m)^  {k} (\S+)', b).group(1) if re.search(rf'(?m)^  {k} (\S+)', b) else d)
            args = []
            if fld('gui', 'TRUE') == 'FALSE':
                args.append('--no-gui')
            if fld('useNetconvert', 'FALSE') == 'FALSE':
                args.append('--no-netconvert')
            if fld('enableTrafficLights', 'TRUE') == 'FALSE':
                args.append('--disable-traffic-lights')
            if fld('enableHeight', 'FALSE') == 'TRUE':
                args.append('--enable-height')
            args += [f'--max-vehicles={fld("maxVehicles", "100")}', f'--port={fld("port", "8873")}',
                     f'--seed={fld("seed", "1")}', f'--step={fld("step", "200")}', f'--radius={fld("radius", "-1")}',
                     f'--maximum-lateral-speed={fld("maximumLateralSpeed", "2.5")}',
                     f'--maximum-angular-speed={fld("maximumAngularSpeed", "3")}',
                     f'--lane-change-delay={fld("laneChangeDelay", "3")}',
                     '--directory=' + os.path.join(pkg, 'worlds', f'{name}_net')]
            robot = fld('name', '"sumo interface"').strip('"')
            ctrl = os.path.join(pkg, 'controllers', 'sumo_supervisor', 'sumo_supervisor.py')
            actions.append(ExecuteProcess(cmd=[os.path.join(os.environ.get('WEBOTS_HOME', ''), 'webots-controller'),
                                               '--protocol=ipc', f'--robot-name={robot}', ctrl] + args,
                                          output='screen', name='sumo_supervisor'))
        return actions
    route = OpaqueFunction(function=make_route)
    tracker = Node(package='auto_car', executable='object_tracker.py', output='screen',
                   parameters=[sim_time])
    octomap = Node(package='octomap_server', executable='octomap_server_node', name='octomap_server',
                   output='screen', parameters=[os.path.join(cfg, 'octomap.yaml')],
                   remappings=[('cloud_in', '/perception/static_points')],   # moving objects removed
                   condition=IfCondition(slam))

    # ---- safety shield: /cmd_drive_raw -> /cmd_drive (always on) -----------
    shield = Node(package='auto_car', executable='safety_shield.py', output='screen', parameters=[sim_time])
    # Driver monitoring: unresponsive driver -> warning, alarm, then drive to the nearest hospital
    driver_mon = Node(package='auto_car', executable='driver_monitor.py', output='screen',
                      parameters=[sim_time], condition=IfCondition(LaunchConfiguration('driver_monitor')))

    # ---- driving: local planner (Phase 2) -> /cmd_drive_raw -> shield ------
    legacy = LaunchConfiguration('legacy')
    # planner:=arcs (default, hand-written arc planner) or planner:=rl (trained SAC policy, Phase 5).
    # Both publish /cmd_drive_raw and sit behind the same safety shield and traffic rules.
    which = LaunchConfiguration('planner')
    drive_on = ["'", stack, "'.lower() in ('true','1') and '", legacy, "'.lower() not in ('true','1')"]
    planner = Node(package='auto_car', executable='local_planner.py', output='screen',
                   parameters=[sim_time], condition=IfCondition(PythonExpression(
                       drive_on + [" and '", which, "' != 'rl'"])))
    rl_planner = Node(package='auto_car', executable='rl_planner.py', output='screen',
                      parameters=[sim_time], condition=IfCondition(PythonExpression(
                          drive_on + [" and '", which, "' == 'rl'"])))
    # Old reactive stack (YOLO + HSV lanes + state machine + Stanley), kept for comparison only.
    legacy_nodes = [
        Node(package='auto_car', executable='perception_node.py', output='screen',
             parameters=[sim_time], condition=IfCondition(legacy)),
        Node(package='auto_car', executable='planning_node', output='screen',
             parameters=[sim_time], condition=IfCondition(legacy)),
        Node(package='auto_car', executable='control_node', output='screen',
             parameters=[sim_time, {'k_stanley': 3.5, 'v_max': 20.0}], condition=IfCondition(legacy)),
    ]

    # ---- visualisation ------------------------------------------------------
    rviz_cfg = PythonExpression(["'", os.path.join(cfg, 'slam.rviz'), "' if '", slam,
                                 "'.lower() in ('true', '1') else '", os.path.join(cfg, 'sensors.rviz'), "'"])
    rviz_node = Node(package='rviz2', executable='rviz2', output='log',
                     arguments=['-d', rviz_cfg], parameters=[sim_time], condition=IfCondition(rviz))

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='city.wbt', description='world file in share/auto_car/worlds'),
        DeclareLaunchArgument('stack', default_value='true', description='start the local planner (drives the car)'),
        DeclareLaunchArgument('driver_monitor', default_value='true',
                              description='driver attention monitor + hospital emergency mode'),
        DeclareLaunchArgument('planner', default_value='arcs',
                              description="'arcs' = hand-written planner, 'rl' = trained SAC policy"),
        DeclareLaunchArgument('legacy', default_value='false', description='old reactive stack instead of the planner'),
        DeclareLaunchArgument('slam', default_value='false', description='run slam_toolbox (build a map)'),
        DeclareLaunchArgument('rviz', default_value='false', description='open RViz'),
        DeclareLaunchArgument('detector', default_value='rtdetr-l.pt',
                              description="camera model in models/ (rtdetr-l.pt, yolo26s.pt) or 'none'"),
        route,                        # must stay before webots (see make_route)
        webots,
        webots._supervisor,
        vehicle,
        ekf,
        cloud_pre,
        tracker,
        road,
        detector,
        rules,
        shield,
        driver_mon,
        slam_node,
        octomap,
        slam_activate,
        slam_configure,
        planner,
        rl_planner,
        *legacy_nodes,
        rviz_node,
        RegisterEventHandler(OnProcessExit(target_action=webots,
                                           on_exit=[EmitEvent(event=Shutdown())])),
    ])
