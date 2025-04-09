#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from irobot_create_msgs.action import DriveDistance, RotateAngle
from geometry_msgs.msg import Twist
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
        self.POLL_RATE = 0.001  # seconds between readings
        self.DRIVE_SPEED = 0.1   # m/s
        self.ROTATE_SPEED = 0.5  # rad/s
        self.DRIVE_INCREMENT = 0.05  # meters
        self.BASE_SPEED = 0.15        # Base forward speed (m/s)
        self.TURN_FACTOR = 0.5        # Factor to adjust turn intensity
        self.MIN_ROTATION = 0.1       # Minimum rotation speed (rad/s)
        
        # Action clients for movement
        self.drive_client = ActionClient(self, DriveDistance, '/drive_distance')
        self.rotate_client = ActionClient(self, RotateAngle, '/rotate_angle')
        
        # Add twist publisher for continuous movement
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        
        # Modify timer to use async control loop
        self.timer = self.create_timer(
            self.POLL_RATE, 
            lambda: rclpy.create_task(self.control_loop())
        )
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

    async def move_robot(self, linear_x, angular_z):
        """Send velocity commands to the robot"""
        twist_msg = Twist()
        twist_msg.linear.x = float(linear_x)
        twist_msg.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(twist_msg)

    async def control_loop(self):
        """Improved control loop with proportional line following"""
        # Read sensor values
        left_sensor, right_sensor = self.read_ir_sensors()
        
        # Calculate control values
        linear_speed = self.BASE_SPEED
        angular_speed = 0.0
        
        # Both sensors off the line (LOW)
        if left_sensor == GPIO.LOW and right_sensor == GPIO.LOW:
            self.get_logger().warn('Both sensors off line!')
            await self.move_robot(0.0, 0.0)  # Stop
            return
            
        # Both sensors on the line (HIGH) - move forward
        elif left_sensor == GPIO.HIGH and right_sensor == GPIO.HIGH:
            self.get_logger().info('On line - moving forward')
            angular_speed = 0.0
            
        # Left sensor on line (HIGH), right sensor off (LOW) - turn left
        elif left_sensor == GPIO.HIGH and right_sensor == GPIO.LOW:
            self.get_logger().info('Turning left')
            angular_speed = -self.ROTATE_SPEED * self.TURN_FACTOR
            linear_speed *= 0.8  # Slow down during turns
            
        # Right sensor on line (HIGH), left sensor off (LOW) - turn right
        elif left_sensor == GPIO.LOW and right_sensor == GPIO.HIGH:
            self.get_logger().info('Turning right')
            angular_speed = self.ROTATE_SPEED * self.TURN_FACTOR
            linear_speed *= 0.8  # Slow down during turns

        # Apply minimum rotation threshold
        if 0 < abs(angular_speed) < self.MIN_ROTATION:
            angular_speed = self.MIN_ROTATION * (1 if angular_speed > 0 else -1)

        # Send movement commands
        await self.move_robot(linear_speed, angular_speed)

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