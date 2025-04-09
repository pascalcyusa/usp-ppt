#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from irobot_create_msgs.action import DriveDistance, RotateAngle
from irobot_create_msgs.msg import IrIntensityVector

class IRLineFollower(Node):
    def __init__(self):
        super().__init__('ir_line_follower')
        
        # Configuration
        self.IR_THRESHOLD = 800  # Adjust based on your surface/line
        self.DRIVE_SPEED = 0.1   # m/s
        self.ROTATE_SPEED = 0.5  # rad/s
        self.DRIVE_INCREMENT = 0.05  # meters to drive forward
        
        # IR Sensor subscriptions
        self.left_ir_sub = self.create_subscription(
            IrIntensityVector,
            '/ir_intensity/left',
            self.left_ir_callback,
            10
        )
        self.right_ir_sub = self.create_subscription(
            IrIntensityVector,
            '/ir_intensity/right',
            self.right_ir_callback,
            10
        )
        
        # Action clients for movement
        self.drive_client = ActionClient(self, DriveDistance, '/drive_distance')
        self.rotate_client = ActionClient(self, RotateAngle, '/rotate_angle')
        
        # State variables
        self.left_ir_value = 0
        self.right_ir_value = 0
        
        # Control loop timer
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('IR Line Follower initialized')

    def left_ir_callback(self, msg):
        if msg.readings:
            self.left_ir_value = msg.readings[0].value
            self.get_logger().debug(f'Left IR: {self.left_ir_value}')

    def right_ir_callback(self, msg):
        if msg.readings:
            self.right_ir_value = msg.readings[0].value
            self.get_logger().debug(f'Right IR: {self.right_ir_value}')

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
        # Print current IR values for debugging
        self.get_logger().info(f'IR Values - Left: {self.left_ir_value}, Right: {self.right_ir_value}')
        
        # Both sensors off the line (high values)
        if self.left_ir_value > self.IR_THRESHOLD and self.right_ir_value > self.IR_THRESHOLD:
            self.get_logger().warn('Both sensors off line!')
            return
            
        # Both sensors on the line (low values)
        elif self.left_ir_value < self.IR_THRESHOLD and self.right_ir_value < self.IR_THRESHOLD:
            self.get_logger().info('On line - driving forward')
            self.drive_distance(self.DRIVE_INCREMENT)
            
        # Left sensor on line, right sensor off
        elif self.left_ir_value < self.IR_THRESHOLD:
            self.get_logger().info('Turning left')
            self.rotate_angle(-0.1)  # Small left turn
            
        # Right sensor on line, left sensor off
        elif self.right_ir_value < self.IR_THRESHOLD:
            self.get_logger().info('Turning right')
            self.rotate_angle(0.1)   # Small right turn

def main(args=None):
    rclpy.init(args=args)
    node = IRLineFollower()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()