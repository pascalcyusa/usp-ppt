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
        config = self.camera.create_preview_configuration(main={"size": (640, 480)})
        self.camera.configure(config)
        self.camera.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        self.camera.start()
        time.sleep(2)

        # Color Detection Configuration
        self.station_colors = {
            "Blue Station": {
                "lower": (100, 150, 150),  # Adjusted for better detection
                "upper": (130, 255, 255)
            },
            "Green Station": {
                "lower": (50, 150, 150),   # Adjusted for better detection
                "upper": (80, 255, 255)
            }
        }
        self.color_detection_threshold = 3000  # Adjusted threshold
        self.is_at_station = False
        self.current_station = None
        self.station_wait_time = 3.0  # seconds to wait at station
        
        # ROS2 Setup
        self.drive_client = ActionClient(self, DriveDistance, '/drive_distance')
        self.rotate_client = ActionClient(self, RotateAngle, '/rotate_angle')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Timers
        self.ir_timer = self.create_timer(self.POLL_RATE, self.ir_timer_callback)
        self.color_timer = self.create_timer(0.1, self.color_timer_callback)
        
        self.get_logger().info('IR Line Follower with Stations initialized')

    def color_timer_callback(self):
        """Handle color detection"""
        if self.is_at_station:
            return

        frame = self.camera.capture_array()
        # Rotate the frame 180 degrees to match IR sensor orientation
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        # Convert to HSV and create mask for white background
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        for station_name, color_range in self.station_colors.items():
            lower = np.array(color_range["lower"])
            upper = np.array(color_range["upper"])
            color_mask = cv2.inRange(hsv, lower, upper)
            
            # Exclude white background from color detection
            color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(white_mask))
            
            detected_pixels = cv2.countNonZero(color_mask)
            
            if detected_pixels > self.color_detection_threshold:
                self.handle_station_arrival(station_name)
                break

    def handle_station_arrival(self, station_name):
        """Handle arrival at a station"""
        if not self.is_at_station:
            self.is_at_station = True
            self.current_station = station_name
            self.get_logger().info(f'Stopping at {station_name}')
            self.move_robot(0.0, 0.0)  # Stop robot
            
            # Schedule resume after wait time
            self.create_timer(
                self.station_wait_time,
                lambda: self.resume_movement(station_name),
                oneshot=True
            )

    def resume_movement(self, station_name):
        """Resume movement after station stop"""
        if self.current_station == station_name:
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
            self.move_robot(linear_speed * 0.8, -self.ROTATE_SPEED * self.TURN_FACTOR)
        elif left_sensor == GPIO.LOW and right_sensor == GPIO.HIGH:
            self.move_robot(linear_speed * 0.8, self.ROTATE_SPEED * self.TURN_FACTOR)

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