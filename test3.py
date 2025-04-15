#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

import os
from dotenv import load_dotenv
import time
from enum import Enum, auto
import math
import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls
import requests
import json
import RPi.GPIO as GPIO
from geometry_msgs.msg import Twist

from irobot_create_msgs.action import DriveDistance, RotateAngle
from builtin_interfaces.msg import Duration
from irobot_create_msgs.msg import AudioNoteVector, AudioNote

# Load environment variables from .env file
load_dotenv()

# --- Airtable Configuration ---
AIRTABLE_API_TOKEN = os.getenv('AIRTABLE_API_TOKEN')
AIRTABLE_BASE_ID = os.getenv('AIRTABLE_BASE_ID')
AIRTABLE_TABLE_NAME = os.getenv('AIRTABLE_TABLE_NAME')

if not all([AIRTABLE_API_TOKEN, AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME]):
    raise EnvironmentError(
        "Missing required Airtable environment variables. Please check your .env file.")

# --- Construct Airtable URL and Headers ---
AIRTABLE_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
AIRTABLE_HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_API_TOKEN}",
    "Content-Type": "application/json",
}

# --- Field names in Airtable base (MUST match exactly, case-sensitive) ---
AIRTABLE_ORDER_NAME_COLUMN = "Order Name"       # Column for the order identifier

# Station Status Fields (Numeric)
AIRTABLE_COOKING_1_STATUS_FIELD = "Cooking 1 Status"
AIRTABLE_COOKING_2_STATUS_FIELD = "Cooking 2 Status"
AIRTABLE_WHIPPED_CREAM_STATUS_FIELD = "Whipped Cream Status"
AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD = "Choco Chips Status"
AIRTABLE_SPRINKLES_STATUS_FIELD = "Sprinkles Status"
AIRTABLE_PICKUP_STATUS_FIELD = "Pickup Status"

# --- Airtable Status Codes (Numeric) ---
STATUS_WAITING = 0
STATUS_ARRIVED = 1
STATUS_DONE = 99

# --- Map Airtable Fields and Order Requirements to Station Indices ---
STATION_FIELD_TO_INDEX = {
    AIRTABLE_COOKING_1_STATUS_FIELD: 1,
    AIRTABLE_COOKING_2_STATUS_FIELD: 1,  # Same station as Cooking 1
    AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD: 3,
    AIRTABLE_WHIPPED_CREAM_STATUS_FIELD: 4,
    AIRTABLE_SPRINKLES_STATUS_FIELD: 5,
    AIRTABLE_PICKUP_STATUS_FIELD: 0
}
STATION_INDEX_TO_FIELD = {v: k for k, v in STATION_FIELD_TO_INDEX.items()}

# --- Hardware Configuration ---
LEFT_IR_PIN = 16
RIGHT_IR_PIN = 18
CAMERA_RESOLUTION = (640, 480)
CAMERA_ROTATION = cv2.ROTATE_180

# --- Color Detection Configuration ---
STATION_COLORS_HSV = {
    0: {"name": "Pickup Station", "hsv_lower": (35, 100, 100), "hsv_upper": (85, 255, 255), "color_bgr": (255, 0, 0)},
    1: {"name": "Cooking Station", "hsv_lower": (35, 100, 100), "hsv_upper": (85, 255, 255), "color_bgr": (255, 0, 0)},
    3: {"name": "Chocolate Chips", "hsv_lower": (35, 100, 100), "hsv_upper": (85, 255, 255), "color_bgr": (255, 0, 0)},
    4: {"name": "Whipped Cream", "hsv_lower": (35, 100, 100), "hsv_upper": (85, 255, 255), "color_bgr": (255, 0, 0)},
    5: {"name": "Sprinkles", "hsv_lower": (35, 100, 100), "hsv_upper": (85, 255, 255), "color_bgr": (255, 0, 0)},
}

# --- Navigation & Control Parameters ---
BASE_DRIVE_SPEED = 0.01
BASE_ROTATE_SPEED = 0.2
TURN_FACTOR = 0.7
LOST_LINE_ROTATE_SPEED = 0.1
COLOR_DETECTION_THRESHOLD = 2000
COLOR_COOLDOWN_SEC = 5.0
STATION_WAIT_TIMEOUT_SEC = 120.0

class RobotState(Enum):
    IDLE = auto()
    FETCHING_ORDER = auto()
    PLANNING_ROUTE = auto()
    MOVING_TO_STATION = auto()
    ARRIVED_AT_STATION = auto()
    WAITING_FOR_STATION_COMPLETION = auto()
    STATION_TIMED_OUT = auto()
    RETURNING_TO_PICKUP = auto()
    ARRIVED_AT_PICKUP = auto()
    ORDER_COMPLETE = auto()
    ALL_ORDERS_COMPLETE = auto()
    ERROR = auto()
    CAMERA_ERROR = auto()
    AIRTABLE_ERROR = auto()
    GPIO_ERROR = auto()

class PancakeRobotNode(Node):
    def __init__(self):
        super().__init__('pancake_robot_node')
        self.get_logger().info("Pancake Robot Node Initializing...")
        
        # Initialize robot state and variables
        self.state = RobotState.IDLE
        self.current_order = None
        self.station_sequence = []
        self.current_sequence_index = 0
        self.target_station_index = -1
        self.pancakes_made_count = 0
        self.last_color_detection_times = {idx: 0.0 for idx in STATION_COLORS_HSV.keys()}
        self.wait_start_time = 0.0
        
        # Initialize hardware
        self._init_hardware()
        
        # Initialize ROS2 publishers and clients
        self._init_ros2()
        
        # Initialize timers
        self.control_timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info("Pancake Robot Node Initialized and Ready.")
        self.play_sound([(440, 200), (550, 300)])

    def _init_hardware(self):
        """Initialize GPIO and Camera"""
        try:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(LEFT_IR_PIN, GPIO.IN)
            GPIO.setup(RIGHT_IR_PIN, GPIO.IN)
            self.get_logger().info(f"GPIO initialized (Pins: L={LEFT_IR_PIN}, R={RIGHT_IR_PIN}).")
        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to initialize GPIO: {e}")
            self.state = RobotState.GPIO_ERROR
            return

        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(main={"size": CAMERA_RESOLUTION})
            self.picam2.configure(config)
            self.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous, "LensPosition": 0.0})
            self.picam2.start()
            time.sleep(2)
            self.get_logger().info("Pi Camera initialized successfully.")
            self.debug_windows = True
        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to initialize Pi Camera: {e}")
            self.cleanup_gpio()
            self.state = RobotState.CAMERA_ERROR
            return

    def _init_ros2(self):
        """Initialize ROS2 publishers and action clients"""
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.audio_publisher = self.create_publisher(AudioNoteVector, '/cmd_audio', 10)
        self.drive_client = ActionClient(self, DriveDistance, '/drive_distance')
        self.rotate_client = ActionClient(self, RotateAngle, '/rotate_angle')

    def fetch_order_from_airtable(self):
        """Fetches the oldest order that needs processing."""
        try:
            params = {
                "maxRecords": 1,
                "filterByFormula": f"AND({{{AIRTABLE_COOKING_1_STATUS_FIELD}}}=0, {{{AIRTABLE_PICKUP_STATUS_FIELD}}}=0)",
                "sort[0][field]": AIRTABLE_ORDER_NAME_COLUMN,
                "sort[0][direction]": "asc"
            }
            
            response = requests.get(url=AIRTABLE_URL, headers=AIRTABLE_HEADERS, params=params)
            response.raise_for_status()
            data = response.json()
            
            records = data.get("records", [])
            if not records:
                self.get_logger().info("No pending orders found.")
                return None
                
            record = records[0]
            record_id = record.get("id")
            fields = record.get("fields", {})
            order_name = fields.get(AIRTABLE_ORDER_NAME_COLUMN)
            
            if not record_id or not order_name:
                self.get_logger().error(f"Invalid record format: {record}")
                return None
                
            return {
                "record_id": record_id,
                "order_name": order_name,
                "station_status": {
                    field: fields.get(field, 0) for field in [
                        AIRTABLE_COOKING_1_STATUS_FIELD,
                        AIRTABLE_COOKING_2_STATUS_FIELD,
                        AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD,
                        AIRTABLE_WHIPPED_CREAM_STATUS_FIELD,
                        AIRTABLE_SPRINKLES_STATUS_FIELD,
                        AIRTABLE_PICKUP_STATUS_FIELD
                    ]
                }
            }
            
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Airtable fetch error: {e}")
            return None

    def update_station_status(self, record_id, station_field_name, new_status_code):
        """Updates a specific station's status for an order."""
        if not record_id or not station_field_name:
            self.get_logger().error("Missing record_id or station_field_name")
            return False

        update_data = {"fields": {station_field_name: new_status_code}}
        
        try:
            response = requests.patch(
                url=f"{AIRTABLE_URL}/{record_id}",
                headers=AIRTABLE_HEADERS,
                json=update_data
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Failed to update station status: {e}")
            return False

    def wait_for_station_completion(self, record_id, station_field_name):
        """Waits for a station to complete its task (status becomes 99)."""
        try:
            response = requests.get(
                url=f"{AIRTABLE_URL}/{record_id}",
                headers=AIRTABLE_HEADERS
            )
            response.raise_for_status()
            data = response.json()
            return data.get('fields', {}).get(station_field_name) == STATUS_DONE
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Error checking station status: {e}")
            return False

    def move_robot(self, linear_x, angular_z):
        """Publishes Twist messages to control robot velocity."""
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.GPIO_ERROR, RobotState.AIRTABLE_ERROR]:
            self.stop_moving()
            return

        twist_msg = Twist()
        twist_msg.linear.x = float(linear_x)
        twist_msg.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(twist_msg)

    def stop_moving(self):
        """Stops the robot movement."""
        self.get_logger().info("Stopping robot...")
        self.move_robot(0.0, 0.0)
        time.sleep(0.1)

    def read_ir_sensors(self):
        """Reads the state of the IR line sensors."""
        try:
            left_val = GPIO.input(LEFT_IR_PIN)
            right_val = GPIO.input(RIGHT_IR_PIN)
            return left_val, right_val
        except Exception as e:
            self.get_logger().error(f"IR sensor read error: {e}")
            return GPIO.HIGH, GPIO.HIGH

    def check_for_station_color(self, frame, target_idx):
        """Detects station color markers in camera frame."""
        if target_idx not in STATION_COLORS_HSV:
            return False, None

        color_info = STATION_COLORS_HSV[target_idx]
        try:
            hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            white_mask = cv2.inRange(
                hsv_image,
                np.array([0, 0, 200]),
                np.array([180, 30, 255])
            )
            
            color_mask = cv2.inRange(
                hsv_image,
                np.array(color_info["hsv_lower"]),
                np.array(color_info["hsv_upper"])
            )
            color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(white_mask))
            
            detected_pixels = cv2.countNonZero(color_mask)
            current_time = time.time()
            
            if detected_pixels > COLOR_DETECTION_THRESHOLD and \
               (current_time - self.last_color_detection_times.get(target_idx, 0.0) > COLOR_COOLDOWN_SEC):
                self.last_color_detection_times[target_idx] = current_time
                return True, frame
                
            return False, frame
            
        except Exception as e:
            self.get_logger().error(f"Color detection error: {e}")
            return False, None

    def play_sound(self, notes):
        """Plays a sequence of notes through the robot's speaker."""
        note_msg = AudioNoteVector()
        for frequency, duration in notes:
            note = AudioNote()
            note.frequency = frequency
            note.max_runtime = Duration(sec=int(duration/1000))
            note_msg.notes.append(note)
        self.audio_publisher.publish(note_msg)

    def cleanup_gpio(self):
        """Cleanup GPIO pins on shutdown."""
        try:
            GPIO.cleanup()
        except Exception as e:
            self.get_logger().error(f"GPIO cleanup error: {e}")

    def control_loop(self):
        """Main control loop for the robot."""
        if self.state == RobotState.IDLE:
            self.current_order = self.fetch_order_from_airtable()
            if self.current_order:
                self.state = RobotState.PLANNING_ROUTE
            else:
                self.state = RobotState.ALL_ORDERS_COMPLETE

        elif self.state == RobotState.PLANNING_ROUTE:
            # Plan the sequence of stations to visit
            self.station_sequence = [
                STATION_FIELD_TO_INDEX[field]
                for field in [
                    AIRTABLE_COOKING_1_STATUS_FIELD,
                    AIRTABLE_COOKING_2_STATUS_FIELD,
                    AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD,
                    AIRTABLE_WHIPPED_CREAM_STATUS_FIELD,
                    AIRTABLE_SPRINKLES_STATUS_FIELD,
                    AIRTABLE_PICKUP_STATUS_FIELD
                ]
                if self.current_order["station_status"][field] == STATUS_WAITING
            ]
            self.current_sequence_index = 0
            self.state = RobotState.MOVING_TO_STATION

        elif self.state == RobotState.MOVING_TO_STATION:
            # Line following and color detection logic
            left_val, right_val = self.read_ir_sensors()
            frame = self.picam2.capture_array()
            
            target_idx = self.station_sequence[self.current_sequence_index]
            color_detected, _ = self.check_for_station_color(frame, target_idx)
            
            if color_detected:
                self.stop_moving()
                self.state = RobotState.ARRIVED_AT_STATION
            else:
                # Basic line following
                if not left_val and not right_val:  # Both sensors on line
                    self.move_robot(BASE_DRIVE_SPEED, 0.0)
                elif not left_val:  # Left sensor on line
                    self.move_robot(BASE_DRIVE_SPEED, -BASE_ROTATE_SPEED)
                elif not right_val:  # Right sensor on line
                    self.move_robot(BASE_DRIVE_SPEED, BASE_ROTATE_SPEED)
                else:  # Both sensors off line
                    self.move_robot(0.0, LOST_LINE_ROTATE_SPEED)

        elif self.state == RobotState.ARRIVED_AT_STATION:
            current_station = self.station_sequence[self.current_sequence_index]
            station_field = STATION_INDEX_TO_FIELD[current_station]
            
            if self.update_station_status(self.current_order["record_id"], station_field, STATUS_ARRIVED):
                self.wait_start_time = time.time()
                self.state = RobotState.WAITING_FOR_STATION_COMPLETION
            else:
                self.state = RobotState.ERROR

        elif self.state == RobotState.WAITING_FOR_STATION_COMPLETION:
            current_station = self.station_sequence[self.current_sequence_index]
            station_field = STATION_INDEX_TO_FIELD[current_station]
            
            if self.wait_for_station_completion(self.current_order["record_id"], station_field):
                self.current_sequence_index += 1
                if self.current_sequence_index >= len(self.station_sequence):
                    self.state = RobotState.ORDER_COMPLETE
                else:
                    self.state = RobotState.MOVING_TO_STATION
            elif time.time() - self.wait_start_time > STATION_WAIT_TIMEOUT_SEC:
                self.state = RobotState.STATION_TIMED_OUT

        elif self.state == RobotState.ORDER_COMPLETE:
            self.play_sound([(660, 200), (880, 300)])  # Success sound
            self.current_order = None
            self.station_sequence = []
            self.current_sequence_index = 0
            self.state = RobotState.IDLE

        elif self.state == RobotState.ALL_ORDERS_COMPLETE:
            self.play_sound([(440, 200), (440, 200)])  # Completion sound
            self.get_logger().info("All orders completed!")
            time.sleep(5)  # Wait before checking for new orders
            self.state = RobotState.IDLE

def main(args=None):
    rclpy.init(args=args)
    node = PancakeRobotNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.cleanup_gpio()
        rclpy.shutdown()

if __name__ == '__main__':
    main()