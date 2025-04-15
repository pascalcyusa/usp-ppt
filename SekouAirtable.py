#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls
import requests
import json
import RPi.GPIO as GPIO
from geometry_msgs.msg import Twist
from irobot_create_msgs.action import DriveDistance, RotateAngle
from irobot_create_msgs.msg import AudioNoteVector, AudioNote
from enum import Enum, auto
from irobot_create_msgs.msg import InterfaceButtons

# --- Airtable Configuration ---
class AirtablePancake:
    def __init__(self):
        self.AIRTABLE_API_KEY = "pat9eVlOP9knFawW5.cfe50b9999314cff7122fc2ec4373338acb96d860b0d43aab09b619733fc1a23"
        self.AIRTABLE_BASE_ID = "app4CfINDWGkqlxrN"
        self.AIRTABLE_TABLE_NAME = "Orders"
        self.url = f"https://api.airtable.com/v0/{self.AIRTABLE_BASE_ID}/{self.AIRTABLE_TABLE_NAME}"
        self.headers = {
            "Authorization": f"Bearer {self.AIRTABLE_API_KEY}",
            "Content-Type": "application/json",
        }

    def check_value(self, station):
        params = {'sort[0][field]': 'Created', 'sort[0][direction]': 'asc'}
        response = requests.get(self.url, headers=self.headers, params=params)
        if response.status_code == 200:
            data = response.json()
            for record in data['records']:
                if record['fields'].get("Pickup Status") != 99:
                    return record['fields'].get(station)
        return None

    def change_value(self, station, value):
        current_order = 0
        params = {'sort[0][field]': 'Created', 'sort[0][direction]': 'asc'}
        response = requests.get(self.url, headers=self.headers, params=params)
        if response.status_code == 200:
            data = response.json()
            for record in data['records']:
                if record['fields'].get("Pickup Status") != 99:
                    current_order = record.get("id")
                    break
        patch_url = f'{self.url}/{current_order}'
        data = {"fields": {station: value}}
        response = requests.patch(patch_url, headers=self.headers, json=data)
        return response.status_code == 200

# --- Robot State Management ---
class RobotState(Enum):
    IDLE = auto()
    FETCHING_ORDER = auto()
    MOVING_TO_STATION = auto()
    ARRIVED_AT_STATION = auto()
    WAITING_FOR_COMPLETION = auto()
    ORDER_COMPLETE = auto()
    ERROR = auto()

# --- Main Robot Control Class ---
class PancakeRobotNode(Node):
    def __init__(self):
        super().__init__('pancake_robot_node')
        self.get_logger().info("Pancake Robot Node Initializing...")

        # Airtable setup
        self.airtable = AirtablePancake()

        # Camera setup
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"size": (640, 480)})
        self.picam2.configure(config)
        self.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        self.picam2.start()
        time.sleep(2)

        # ROS2 setup
        self.drive_client = ActionClient(self, DriveDistance, '/drive_distance')
        self.rotate_client = ActionClient(self, RotateAngle, '/rotate_angle')
        self.audio_publisher = self.create_publisher(AudioNoteVector, '/cmd_audio', 10)

        # PID setup
        self.Kp = 0.1
        self.Ki = 0
        self.Kd = 0
        self.last_error = 0
        self.integral = 0

        # State variables
        self.state = RobotState.IDLE
        self.current_order = None
        self.pancakes_made_count = 0
        self.station_sequence = []
        self.current_station_index = 0
        self.target_station_index = -1

    def move_forward(self, distance=0.05, speed=0.1):
        goal_msg = DriveDistance.Goal()
        goal_msg.distance = distance
        goal_msg.max_translation_speed = speed
        future = self.drive_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)

    def rotate(self, angle, speed=0.5):
        goal_msg = RotateAngle.Goal()
        goal_msg.angle = angle
        goal_msg.max_rotation_speed = speed
        future = self.rotate_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)

    def check_for_station_color(self, frame, target_idx):
        if target_idx not in STATION_COLORS_HSV:
            self.get_logger().warn(f"Invalid target index {target_idx} for color detection.")
            return False
        color_info = STATION_COLORS_HSV[target_idx]
        lower_bound = np.array(color_info["hsv_lower"])
        upper_bound = np.array(color_info["hsv_upper"])

        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        detected_pixels = cv2.countNonZero(color_mask)

        if detected_pixels > COLOR_DETECTION_THRESHOLD:
            self.get_logger().info(f"Detected color for station {target_idx}!")
            return True
        return False

    def fetch_order_from_airtable(self):
        order = self.airtable.check_value('Pickup Status')
        return order

    def control_loop(self):
        if self.state == RobotState.IDLE:
            self.get_logger().info("Fetching next order...")
            order = self.fetch_order_from_airtable()
            if order:
                self.current_order = order
                self.state = RobotState.MOVING_TO_STATION
            else:
                self.get_logger().info("No orders to process.")
                self.state = RobotState.ORDER_COMPLETE

        if self.state == RobotState.MOVING_TO_STATION:
            self.get_logger().info(f"Moving to station {self.target_station_index}...")
            detected_color = self.check_for_station_color(self.picam2.capture_array(), self.target_station_index)
            if detected_color:
                self.state = RobotState.ARRIVED_AT_STATION
            else:
                self.move_forward()

        if self.state == RobotState.ARRIVED_AT_STATION:
            self.get_logger().info(f"Arrived at station {self.target_station_index}")
            self.airtable.change_value(STATION_INDEX_TO_FIELD[self.target_station_index], STATUS_ARRIVED)
            self.state = RobotState.WAITING_FOR_COMPLETION

        if self.state == RobotState.WAITING_FOR_COMPLETION:
            self.get_logger().info("Waiting for completion status...")
            status = self.airtable.check_value('Pickup Status')
            if status == STATUS_DONE:
                self.state = RobotState.ORDER_COMPLETE

        if self.state == RobotState.ORDER_COMPLETE:
            self.get_logger().info("Order complete, resetting.")
            self.pancakes_made_count += 1
            self.state = RobotState.IDLE

def main(args=None):
    rclpy.init(args=args)
    pancake_robot_node = PancakeRobotNode()
    rclpy.spin(pancake_robot_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
