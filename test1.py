#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from irobot_create_msgs.action import DriveDistance, RotateAngle
from geometry_msgs.msg import Twist
import RPi.GPIO as GPIO
import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls
import time
import asyncio


class IRLineFollowerWithStations(Node):
    def __init__(self):
        super().__init__('ir_line_follower_with_stations')

        # GPIO Setup
        GPIO.setmode(GPIO.BOARD)
        self.LEFT_IR_PIN = 16
        self.RIGHT_IR_PIN = 18
        GPIO.setup(self.LEFT_IR_PIN, GPIO.IN)
        GPIO.setup(self.RIGHT_IR_PIN, GPIO.IN)

        # Movement Configuration
        self.POLL_RATE = 0.001
        self.DRIVE_SPEED = 0.01
        self.ROTATE_SPEED = 0.2
        self.BASE_SPEED = 0.01
        self.TURN_FACTOR = 0.7
        self.MIN_ROTATION = 0.1

        # Camera Setup
        self.camera = Picamera2()
        config = self.camera.create_preview_configuration(
            main={"size": (640, 480)})
        self.camera.configure(config)
        self.camera.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        self.camera.start()
        time.sleep(2)

        # Station Colors (HSV format)
        self.station_colors = {
            "Green Station": {
                "lower": (35, 100, 100),    # Lower bound for green
                "upper": (85, 255, 255),    # Upper bound for green
                # BGR color for visualization (Green)
                "color_bgr": (0, 255, 0),
                "last_detected": 0          # Timestamp of last detection
            }
        }

        # Detection parameters
        self.color_detection_threshold = 2000
        self.is_at_station = False
        self.current_station = None
        self.station_wait_time = 3.0
        self.color_cooldown = 5.0  # Time in seconds before same color can be detected again
        self.debug_windows = True  # Enable debug windows

        # ROS2 Setup
        self.drive_client = ActionClient(
            self, DriveDistance, '/drive_distance')
        self.rotate_client = ActionClient(self, RotateAngle, '/rotate_angle')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Timers
        self.ir_timer = self.create_timer(
            self.POLL_RATE, self.ir_timer_callback)
        self.color_timer = self.create_timer(0.1, self.color_timer_callback)

        self.get_logger().info('IR Line Follower with Stations initialized')

    def color_timer_callback(self):
        """Handle color detection with visualization"""
        if self.is_at_station:
            return

        frame = self.camera.capture_array()
        frame = cv2.rotate(frame, cv2.ROTATE_180)  # Rotate camera image
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create debug visualization
        debug_frame = frame.copy()
        color_mask = np.zeros_like(frame)

        # Create white background mask
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)

        for station_name, params in self.station_colors.items():
            lower = np.array(params["lower"])
            upper = np.array(params["upper"])
            color_bgr = params["color_bgr"]

            # Create color mask excluding white background
            color_mask = cv2.inRange(hsv, lower, upper)
            color_mask = cv2.bitwise_and(
                color_mask, cv2.bitwise_not(white_mask))

            # Count detected pixels
            detected_pixels = cv2.countNonZero(color_mask)

            # Get average BGR values in detected area
            if detected_pixels > 0:
                detected_area = cv2.bitwise_and(frame, frame, mask=color_mask)
                bgr_values = cv2.mean(detected_area)
                self.get_logger().info(
                    f'{station_name} - BGR values: B:{bgr_values[0]:.1f}, G:{bgr_values[1]:.1f}, R:{bgr_values[2]:.1f} - Pixels: {detected_pixels}')

            # Add visualization
            detected_area = cv2.bitwise_and(frame, frame, mask=color_mask)
            debug_frame = cv2.addWeighted(
                debug_frame, 1, detected_area, 0.5, 0)

            # Add text showing detection info
            text = f"{station_name}: {detected_pixels} px"
            y_pos = 30 * (list(self.station_colors.keys()
                               ).index(station_name) + 1)
            cv2.putText(debug_frame, text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)

            # Check for station detection
            current_time = time.time()
            if detected_pixels > self.color_detection_threshold and current_time - params["last_detected"] > self.color_cooldown:
                if not self.is_at_station:
                    params["last_detected"] = current_time
                    self.handle_station_arrival(station_name)

        if self.debug_windows:
            # Show debug windows
            cv2.imshow("Camera View (Press 'q' to quit)", debug_frame)
            cv2.imshow("HSV View", hsv)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cleanup()
                rclpy.shutdown()

    def handle_station_arrival(self, station_name):
        """Handle arrival at a station"""
        self.is_at_station = True
        self.current_station = station_name
        self.get_logger().info(f'Stopping at {station_name}')

        # Stop the robot
        self.move_robot(0.0, 0.0)

        # Create timer for station wait
        self._station_timer = self.create_timer(
            self.station_wait_time,
            lambda: self._resume_movement_callback(station_name)
        )

    def _resume_movement_callback(self, station_name):
        """Resume normal operation after station stop"""
        if self.current_station == station_name:
            # Cleanup timer
            if hasattr(self, '_station_timer'):
                self._station_timer.cancel()
                delattr(self, '_station_timer')

            # Reset station status
            self.is_at_station = False
            self.current_station = None
            self.get_logger().info('Resuming line following')

    def ir_timer_callback(self):
        """Handle IR line following"""
        if self.is_at_station:
            return

        left_sensor, right_sensor = self.read_ir_sensors()
        linear_speed = self.BASE_SPEED
        angular_speed = 0.0

        # IR line following logic
        if left_sensor == GPIO.LOW and right_sensor == GPIO.LOW:
            self.get_logger().warn('Both sensors off line!')
            recovery_rotation = self.ROTATE_SPEED * 0.5
            self.move_robot(0.0, recovery_rotation)
        elif left_sensor == GPIO.HIGH and right_sensor == GPIO.HIGH:
            self.move_robot(linear_speed, 0.0)
        elif left_sensor == GPIO.HIGH and right_sensor == GPIO.LOW:
            self.move_robot(linear_speed * 0.8, -
                            self.ROTATE_SPEED * self.TURN_FACTOR)
        elif left_sensor == GPIO.LOW and right_sensor == GPIO.HIGH:
            self.move_robot(linear_speed * 0.8,
                            self.ROTATE_SPEED * self.TURN_FACTOR)

    def read_ir_sensors(self):
        """Read IR sensor values"""
        left_value = GPIO.input(self.LEFT_IR_PIN)
        right_value = GPIO.input(self.RIGHT_IR_PIN)
        return left_value, right_value

    def move_robot(self, linear_x, angular_z):
        """Send movement commands"""
        twist_msg = Twist()
        twist_msg.linear.x = float(linear_x)
        twist_msg.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(twist_msg)

    def cleanup(self):
        """Clean up resources"""
        self.camera.stop()
        GPIO.cleanup()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = IRLineFollowerWithStations()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
