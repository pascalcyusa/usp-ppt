#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from irobot_create_msgs.action import DriveDistance, RotateAngle
import RPi.GPIO as GPIO
import time

class IRLineFollower(Node):
    def __init__(self):
        super().__init__('ir_line_follower')
        
        # GPIO Setup
        GPIO.setmode(GPIO.BOARD) 
        
        # Define GPIO pins for IR sensors
        self.LEFT_IR_PIN = 16   
        self.RIGHT_IR_PIN = 18 
        
        # Setup GPIO pins as inputs
        GPIO.setup(self.LEFT_IR_PIN, GPIO.IN)
        GPIO.setup(self.RIGHT_IR_PIN, GPIO.IN)
        
        # Configuration
        self.POLL_RATE = 0.1  # seconds between readings
        self.DRIVE_SPEED = 0.1   # m/s
        self.ROTATE_SPEED = 0.5  # rad/s
        self.DRIVE_INCREMENT = 0.05  # meters
        
        # Action clients for movement
        self.drive_client = ActionClient(self, DriveDistance, '/drive_distance')
        self.rotate_client = ActionClient(self, RotateAngle, '/rotate_angle')
        
        # Create timer for control loop
        self.timer = self.create_timer(self.POLL_RATE, self.control_loop)
        self.get_logger().info('IR Line Follower initialized')

    def read_ir_sensors(self):
        """Read IR sensor values. Returns (left_value, right_value)
        Note: GPIO.LOW (0) means line detected (sensor over dark surface)
              GPIO.HIGH (1) means no line (sensor over light surface)
        """
        left_value = GPIO.input(self.LEFT_IR_PIN)
        right_value = GPIO.input(self.RIGHT_IR_PIN)
        self.get_logger().debug(f'IR Values - Left: {left_value}, Right: {right_value}')
        return left_value, right_value

    async def drive_distance(self, distance):
        goal_msg = DriveDistance.Goal()
        goal_msg.distance = float(distance)
        goal_msg.max_translation_speed = self.DRIVE_SPEED
        
        self.get_logger().info(f'Driving {distance}m')
        await self.drive_client.send_goal_async(goal_msg)

    async def rotate_angle(self, angle):
        goal_msg = RotateAngle.Goal()
        goal_msg.angle = float(angle)
        goal_msg.max_rotation_speed = self.ROTATE_SPEED
        
        self.get_logger().info(f'Rotating {angle} radians')
        await self.rotate_client.send_goal_async(goal_msg)

    def control_loop(self):
        # Read sensor values
        left_sensor, right_sensor = self.read_ir_sensors()
        
        # Both sensors off the line (HIGH)
        if left_sensor == GPIO.LOW and right_sensor == GPIO.LOW:
            self.get_logger().warn('Both sensors off line!')
            self.stop_moving()
            
        # Both sensors on the line (LOW)
        elif left_sensor == GPIO.HIGH and right_sensor == GPIO.HIGH:
            self.get_logger().info('On line - driving forward')
            self.drive_distance(self.DRIVE_INCREMENT)
            
        # Left sensor on line (LOW), right sensor off (HIGH)
        elif left_sensor == GPIO.LOW and right_sensor == GPIO.HIGH:
            self.get_logger().info('Turning left')
            self.rotate_angle(-0.1)
            
        # Right sensor on line (LOW), left sensor off (HIGH)
        elif left_sensor == GPIO.HIGH and right_sensor == GPIO.LOW:
            self.get_logger().info('Turning right')
            self.rotate_angle(0.1)

    def stop_moving(self):
        return self.drive_distance(0.0)

    def cleanup(self):
        """Clean up GPIO on shutdown"""
        GPIO.cleanup()

def main(args=None):
    rclpy.init(args=args)
    node = IRLineFollower()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()  # Clean up GPIO
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()