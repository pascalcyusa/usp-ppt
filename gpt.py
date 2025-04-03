#!/usr/bin/env python3

import rclpy
import time
import math

from rclpy.node import Node
from geometry_msgs.msg import Twist
from irobot_create_msgs.msg import WheelTicks  # or Odom if you prefer
from irobot_create_msgs.srv import ResetPose
from irobot_create_msgs.srv import DriveDistance, RotateAngle

class PancakeRobot(Node):
    def __init__(self):
        super().__init__('pancake_robot')

        # Create a publisher to send velocity commands (if needed for manual driving)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Create service clients for built-in drive/rotate (if using iRobot's built-in services)
        self.drive_client = self.create_client(DriveDistance, '/drive_distance')
        self.rotate_client = self.create_client(RotateAngle, '/rotate_angle')
        self.reset_pose_client = self.create_client(ResetPose, '/reset_pose')

        # Wait until services are available
        self.get_logger().info("Waiting for services...")
        self.drive_client.wait_for_service()
        self.rotate_client.wait_for_service()
        self.reset_pose_client.wait_for_service()

        # You can store station positions or distances here
        # For example, assume each station is 2 meters apart along a straight line:
        self.station_distances = [0.0, 2.0, 4.0, 6.0, 8.0]  # meters from start
        # The time (in seconds) required at each station:
        # station 1 => Take Order, station 2 => Cook, station 3 => Stack, station 4 => Toppings
        self.station_times = [5, 10, 5, 5]  # example times

        # Reset the robot’s internal odometry to 0,0,0
        self.reset_pose()

        self.get_logger().info("Initialization complete. Ready to start pancake workflow.")

    def reset_pose(self):
        """Reset the internal pose to 0,0,0 if needed."""
        req = ResetPose.Request()
        future = self.reset_pose_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info("Pose reset successfully.")
        else:
            self.get_logger().warn("Failed to reset pose.")

    def move_distance(self, distance_m):
        """
        Use the built-in DriveDistance service to move the robot a given distance in meters.
        Positive distance moves forward, negative distance moves backward.
        """
        req = DriveDistance.Request()
        req.distance = float(distance_m)
        req.max_translation_speed = 0.3  # m/s, adjust as desired
        future = self.drive_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info(f"Moved {distance_m} meters.")
        else:
            self.get_logger().warn("DriveDistance service call failed.")

    def rotate_angle(self, angle_deg):
        """
        Rotate the robot a certain angle in degrees.
        Positive angle => rotate left (counter-clockwise).
        """
        req = RotateAngle.Request()
        req.angle = math.radians(angle_deg)
        req.max_rotation_speed = 1.0  # rad/s
        future = self.rotate_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info(f"Rotated {angle_deg} degrees.")
        else:
            self.get_logger().warn("RotateAngle service call failed.")

    def do_station_task(self, station_index):
        """
        Simulate performing the station's task by waiting the required time.
        E.g., station 1 => take order, station 2 => cook, etc.
        """
        task_time = self.station_times[station_index - 1]  # station_index starts at 1
        self.get_logger().info(f"Performing task at station {station_index} for {task_time}s...")
        time.sleep(task_time)
        self.get_logger().info(f"Station {station_index} task complete.")

    def run_pancake_workflow(self, num_pancakes=1):
        """
        Main logic to go station by station for each pancake.
        For each pancake:
          1. Move from station 1 -> 2 -> 3 -> 4.
          2. Wait the required time at each station.
          3. Optionally deliver the pancake to 'end' or back to start.
        """
        for p in range(num_pancakes):
            self.get_logger().info(f"Starting pancake #{p+1} workflow.")

            # Start at station 1 (assume we begin at distance 0).
            current_station = 1

            # 1) Station 1 task
            self.do_station_task(current_station)

            # Move to station 2
            self.move_distance(self.station_distances[1] - self.station_distances[0])
            current_station = 2
            self.do_station_task(current_station)

            # Move to station 3
            self.move_distance(self.station_distances[2] - self.station_distances[1])
            current_station = 3
            self.do_station_task(current_station)

            # Move to station 4
            self.move_distance(self.station_distances[3] - self.station_distances[2])
            current_station = 4
            self.do_station_task(current_station)

            # Optionally move to "end" (distance index 4 in station_distances)
            # or back to start.  Example: move to station_distances[4].
            self.move_distance(self.station_distances[4] - self.station_distances[3])
            self.get_logger().info(f"Completed pancake #{p+1} delivery!\n")

        self.get_logger().info("All pancake workflows complete.")

def main(args=None):
    rclpy.init(args=args)
    node = PancakeRobot()

    try:
        # Example: run the workflow for 3 pancakes
        node.run_pancake_workflow(num_pancakes=3)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
