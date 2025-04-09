#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from irobot_create_msgs.action import DriveDistance, RotateAngle
from geometry_msgs.msg import Twist
import RPi.GPIO as GPIO
import time
import asyncio

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
        self.POLL_RATE = 0.001  # Increased the sampling speed
        self.DRIVE_SPEED = 0.01
        self.ROTATE_SPEED = 0.5
        self.BASE_SPEED = 0.01
        self.TURN_FACTOR = 0.5
        self.MIN_ROTATION = 0.1
        
        # Action clients for movement
        self.drive_client = ActionClient(self, DriveDistance, '/drive_distance')
        self.rotate_client = ActionClient(self, RotateAngle, '/rotate_angle')
        
        # Add twist publisher for continuous movement
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        
        # Add turn history tracking
        self.turn_history = []  # Will store -1 for left turns, 1 for right turns
        self.MAX_HISTORY = 5    # Keep last 5 turns
        self.RECOVERY_TURN_SPEED = 0.3  # Slower speed for recovery turns
        self.lost_line_count = 0  # Counter for consecutive readings off the line
        self.MAX_LOST_COUNT = 3  # Number of consecutive readings before recovery

        # Create timer without async wrapper
        self.timer = self.create_timer(
            self.POLL_RATE,
            self.timer_callback
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

    def timer_callback(self):
        """Non-async timer callback that handles the control loop"""
        # Read sensor values
        left_sensor, right_sensor = self.read_ir_sensors()
        
        # Calculate control values
        linear_speed = -self.BASE_SPEED  # Invert forward speed
        angular_speed = 0.0
        
        # Both sensors off the line (HIGH) - potential recovery needed
        if left_sensor == GPIO.HIGH and right_sensor == GPIO.HIGH:
            self.lost_line_count += 1
            self.get_logger().warn(f'Both sensors off line! Count: {self.lost_line_count}')
            
            if self.lost_line_count >= self.MAX_LOST_COUNT:
                self.recover_line()
                return
            else:
                self.move_robot(0.0, 0.0)  # Stop remains same
                return
            
        # Reset lost line counter if we're on the line
        self.lost_line_count = 0
            
        # Both sensors on the line (LOW) - move backward (inverted forward)
        if left_sensor == GPIO.LOW and right_sensor == GPIO.LOW:
            self.get_logger().info('On line - moving backward')
            angular_speed = 0.0
            
        # Left sensor on line (LOW), right sensor off (HIGH) - turn right (inverted left)
        elif left_sensor == GPIO.LOW and right_sensor == GPIO.HIGH:
            self.get_logger().info('Turning right')
            angular_speed = self.ROTATE_SPEED * self.TURN_FACTOR  # Positive for right turn
            linear_speed *= 0.8  # Slow down during turns
            self.record_turn(1)  # Record right turn (inverted from left)
            
        # Right sensor on line (LOW), left sensor off (HIGH) - turn left (inverted right)
        elif left_sensor == GPIO.HIGH and right_sensor == GPIO.LOW:
            self.get_logger().info('Turning left')
            angular_speed = -self.ROTATE_SPEED * self.TURN_FACTOR  # Negative for left turn
            linear_speed *= 0.8  # Slow down during turns
            self.record_turn(-1)  # Record left turn (inverted from right)

        # Apply minimum rotation threshold
        if 0 < abs(angular_speed) < self.MIN_ROTATION:
            angular_speed = self.MIN_ROTATION * (1 if angular_speed > 0 else -1)

        # Send movement commands
        self.move_robot(linear_speed, angular_speed)

    def move_robot(self, linear_x, angular_z):
        """Non-async version of move_robot"""
        twist_msg = Twist()
        twist_msg.linear.x = float(linear_x)
        twist_msg.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(twist_msg)

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

    def stop_moving(self):
        return self.drive_distance(0.0)

    def record_turn(self, direction):
        """Record turn direction (-1 for left, 1 for right)"""
        self.turn_history.append(direction)
        if len(self.turn_history) > self.MAX_HISTORY:
            self.turn_history.pop(0)  # Remove oldest turn

    def recover_line(self):
        """Attempt to recover the line based on turn history (with inverted directions)"""
        if not self.turn_history:
            self.get_logger().warn('No turn history available for recovery')
            self.move_robot(0.0, 0.0)  # Stop remains same
            return

        # Get the last turn direction but DON'T reverse it (already inverted in recording)
        last_turn = self.turn_history[-1]
        recovery_direction = last_turn  # Same direction as last turn (inverted logic)
        
        self.get_logger().info(f'Attempting recovery: turning {"left" if recovery_direction < 0 else "right"}')
        
        # Make recovery turn with inverted direction
        angular_speed = self.RECOVERY_TURN_SPEED * recovery_direction
        self.move_robot(0.0, angular_speed)  # Turn in place
        
        # Clear turn history after recovery attempt
        self.turn_history = []
        self.lost_line_count = 0

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