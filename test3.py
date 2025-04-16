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
from libcamera import controls, Transform # Import Transform for rotation
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
    print("FATAL: Missing required Airtable environment variables. Please check your .env file.")
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
    AIRTABLE_COOKING_2_STATUS_FIELD: 2, # NOTE: Index 2 still needs an entry in STATION_COLORS_HSV for visual stop
    AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD: 3,
    AIRTABLE_WHIPPED_CREAM_STATUS_FIELD: 4,
    AIRTABLE_SPRINKLES_STATUS_FIELD: 5,
    AIRTABLE_PICKUP_STATUS_FIELD: 0
}
STATION_INDEX_TO_FIELD = {v: k for k, v in STATION_FIELD_TO_INDEX.items()}

# --- Hardware Configuration ---
LEFT_IR_PIN = 16
RIGHT_IR_PIN = 18
IR_LINE_DETECT_SIGNAL = GPIO.LOW # Assume LOW signal means the sensor is OVER the line
IR_OFF_LINE_SIGNAL = GPIO.HIGH

CAMERA_RESOLUTION = (640, 480)
CAMERA_TRANSFORM = Transform(hflip=True, vflip=True) # Equivalent to cv2.ROTATE_180

# --- Color Detection Configuration (Using SAME GREEN for ALL stations) ---
COMMON_HSV_LOWER = (35, 100, 100)
COMMON_HSV_UPPER = (85, 255, 255)
COMMON_COLOR_BGR = (0, 255, 0) # BGR for Green

STATION_COLORS_HSV = {
    # Index 0: Pickup Station
    0: {"name": "Pickup Station", "hsv_lower": COMMON_HSV_LOWER, "hsv_upper": COMMON_HSV_UPPER, "color_bgr": COMMON_COLOR_BGR},
    # Index 1: Cooking Station 1
    1: {"name": "Cooking Station 1", "hsv_lower": COMMON_HSV_LOWER, "hsv_upper": COMMON_HSV_UPPER, "color_bgr": COMMON_COLOR_BGR},
    # --- Index 2: Cooking Station 2 - ENSURE THIS ENTRY IS PRESENT ---
    2: {"name": "Cooking Station 2", "hsv_lower": COMMON_HSV_LOWER, "hsv_upper": COMMON_HSV_UPPER, "color_bgr": COMMON_COLOR_BGR},
    # Index 3: Chocolate Chips
    3: {"name": "Chocolate Chips", "hsv_lower": COMMON_HSV_LOWER, "hsv_upper": COMMON_HSV_UPPER, "color_bgr": COMMON_COLOR_BGR},
    # Index 4: Whipped Cream
    4: {"name": "Whipped Cream", "hsv_lower": COMMON_HSV_LOWER, "hsv_upper": COMMON_HSV_UPPER, "color_bgr": COMMON_COLOR_BGR},
    # Index 5: Sprinkles
    5: {"name": "Sprinkles", "hsv_lower": COMMON_HSV_LOWER, "hsv_upper": COMMON_HSV_UPPER, "color_bgr": COMMON_COLOR_BGR},
}

# --- Navigation & Control Parameters (User Specified Values) ---
# NOTE: Polling rates below are descriptive targets. Actual checks depend on the main control loop frequency (20Hz)
# and specific logic (e.g., Airtable check uses modulo timing).
IR_POLL_RATE = 0.001            # Target: Seconds (1000 Hz) - How often to check IR sensors (Actual: every loop, ~50ms)
COLOR_POLL_RATE = 0.1             # Target: Seconds (10 Hz) - How often to check camera for colors (Actual: every loop, ~50ms)
AIRTABLE_POLL_RATE = 2.0          # Target: Seconds - How often to check Airtable for '99' status (Actual: approx this rate via modulo)

BASE_DRIVE_SPEED = 0.01         # m/s - Forward speed during line following
BASE_ROTATE_SPEED = 0.2         # rad/s - Turning speed during line following
TURN_FACTOR = 0.7               # Multiplier for speed reduction during turns
LOST_LINE_ROTATE_SPEED = 0.1    # rad/s - Speed for rotation when line is lost

COLOR_DETECTION_THRESHOLD = 2000  # Min pixels of target color to trigger detection
COLOR_COOLDOWN_SEC = 5.0        # Min seconds before detecting the *same* station color again
STATION_WAIT_TIMEOUT_SEC = 120.0  # Max seconds to wait for Airtable status 99 before failing


class RobotState(Enum):
    IDLE = auto()
    FETCHING_ORDER = auto()
    PLANNING_ROUTE = auto()
    MOVING_TO_STATION = auto()
    ARRIVED_AT_STATION = auto()
    WAITING_FOR_STATION_COMPLETION = auto()
    STATION_TIMED_OUT = auto()
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
        self.current_sequence_index = -1
        self.target_station_index = -1
        self.last_color_detection_times = {idx: 0.0 for idx in STATION_COLORS_HSV.keys()}
        self.wait_start_time = 0.0
        self.picam2 = None
        self.debug_windows = True # Control OpenCV display windows

        # Initialize hardware
        self._init_hardware()
        if self.state in [RobotState.GPIO_ERROR, RobotState.CAMERA_ERROR]:
            self.get_logger().fatal("Hardware initialization failed. Node cannot operate.")
            return

        # Initialize ROS2 publishers and clients
        self._init_ros2()
        if self.state == RobotState.ERROR: # Check if ROS init failed
             self.get_logger().fatal("ROS2 component initialization failed. Node cannot operate.")
             self.cleanup_hardware() # Cleanup GPIO/Camera if ROS failed
             return

        # Initialize timers (Control loop runs at 20Hz)
        self.control_timer = self.create_timer(0.05, self.control_loop) # 0.05 seconds = 20 Hz

        self.get_logger().info("Pancake Robot Node Initialized and Ready.")
        self.play_sound([(440, 150), (550, 200)])

    def _init_hardware(self):
        """Initialize GPIO and Camera"""
        # GPIO Setup
        try:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(LEFT_IR_PIN, GPIO.IN)
            GPIO.setup(RIGHT_IR_PIN, GPIO.IN)
            self.get_logger().info(f"GPIO initialized (Pins: L={LEFT_IR_PIN}, R={RIGHT_IR_PIN}). Expecting {IR_LINE_DETECT_SIGNAL} on line.")
        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to initialize GPIO: {e}")
            self.state = RobotState.GPIO_ERROR
            return

        # Camera Setup
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": CAMERA_RESOLUTION},
                transform=CAMERA_TRANSFORM
            )
            self.picam2.configure(config)
            self.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous, "LensPosition": 0.0})
            self.picam2.start()
            time.sleep(2) # Allow camera warmup
            self.get_logger().info("Pi Camera initialized successfully.")
            if self.debug_windows:
                cv2.namedWindow("Camera Feed")
                cv2.namedWindow("Color Detection Mask")
        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to initialize Pi Camera: {e}", exc_info=True)
            self.cleanup_gpio()
            self.state = RobotState.CAMERA_ERROR
            return

    def _init_ros2(self):
        """Initialize ROS2 publishers and action clients"""
        try:
            self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
            self.audio_publisher = self.create_publisher(AudioNoteVector, '/cmd_audio', 10)
            self.drive_client = ActionClient(self, DriveDistance, '/drive_distance')
            self.rotate_client = ActionClient(self, RotateAngle, '/rotate_angle')
            # TODO: Check if action servers are available using wait_for_server?
            self.get_logger().info("ROS2 publishers and action clients initialized.")
        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to initialize ROS2 components: {e}")
            self.state = RobotState.ERROR # Use general error for ROS issues


    def fetch_order_from_airtable(self):
        """Fetches the oldest order where Cooking 1 is WAITING and Pickup is WAITING."""
        self.get_logger().info("Attempting to fetch order from Airtable...")
        try:
            params = {
                "maxRecords": 1,
                "filterByFormula": f"AND({{{AIRTABLE_COOKING_1_STATUS_FIELD}}}=0, {{{AIRTABLE_PICKUP_STATUS_FIELD}}}=0)",
                "sort[0][field]": AIRTABLE_ORDER_NAME_COLUMN,
                "sort[0][direction]": "asc"
            }
            response = requests.get(url=AIRTABLE_URL, headers=AIRTABLE_HEADERS, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            records = data.get("records", [])

            if not records:
                self.get_logger().info("No pending orders found matching criteria.")
                return None

            record = records[0]
            record_id = record.get("id")
            fields = record.get("fields", {})
            order_name = fields.get(AIRTABLE_ORDER_NAME_COLUMN)

            if not record_id or not order_name:
                self.get_logger().error(f"Airtable record missing ID or Order Name: {record}")
                return None

            self.get_logger().info(f"Fetched order '{order_name}' (Record ID: {record_id}).")
            return {
                "record_id": record_id,
                "order_name": order_name,
                "station_status": {
                    field: fields.get(field, 0) for field in STATION_FIELD_TO_INDEX.keys()
                }
            }
        except requests.exceptions.Timeout:
            self.get_logger().error("Airtable request timed out.")
            self.state = RobotState.AIRTABLE_ERROR
            return None
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Airtable fetch error: {e}")
            self.state = RobotState.AIRTABLE_ERROR
            return None
        except Exception as e:
            self.get_logger().error(f"Unexpected error during Airtable fetch: {e}", exc_info=True)
            self.state = RobotState.AIRTABLE_ERROR
            return None

    def update_station_status(self, record_id, station_field_name, new_status_code):
        """Updates a specific station's status for an order in Airtable."""
        if not record_id or not station_field_name:
            self.get_logger().error("Cannot update Airtable status: Missing record_id or station_field_name")
            return False

        update_data = {"fields": {station_field_name: new_status_code}}
        url = f"{AIRTABLE_URL}/{record_id}"
        self.get_logger().info(f"Updating Airtable: {url} -> {station_field_name} = {new_status_code}")
        try:
            response = requests.patch(url=url, headers=AIRTABLE_HEADERS, json=update_data, timeout=10)
            response.raise_for_status()
            self.get_logger().info(f"Airtable update successful for {station_field_name}.")
            return True
        except requests.exceptions.Timeout:
            self.get_logger().error(f"Airtable update request timed out for {station_field_name}.")
            self.state = RobotState.AIRTABLE_ERROR
            return False
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Failed to update Airtable status for {station_field_name}: {e}")
            try: self.get_logger().error(f"Airtable response content: {response.text}")
            except: pass
            self.state = RobotState.AIRTABLE_ERROR
            return False
        except Exception as e:
            self.get_logger().error(f"Unexpected error during Airtable update: {e}", exc_info=True)
            self.state = RobotState.AIRTABLE_ERROR
            return False

    def wait_for_station_completion(self, record_id, station_field_name):
        """Checks if a station's status in Airtable is DONE (99). Returns True/False."""
        if not record_id or not station_field_name:
            self.get_logger().error("Cannot check Airtable status: Missing record_id or station_field_name")
            return False
        url = f"{AIRTABLE_URL}/{record_id}"
        try:
            response = requests.get(url=url, headers=AIRTABLE_HEADERS, timeout=10)
            response.raise_for_status()
            data = response.json()
            current_status = data.get('fields', {}).get(station_field_name)
            return current_status == STATUS_DONE
        except requests.exceptions.Timeout:
            self.get_logger().warning(f"Airtable status check request timed out for {station_field_name}.")
            return False # Don't assume done on timeout
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Error checking Airtable station status ({station_field_name}): {e}")
            return False # Assume not done if check fails
        except Exception as e:
            self.get_logger().error(f"Unexpected error during Airtable status check: {e}", exc_info=True)
            return False

    def move_robot(self, linear_x, angular_z):
        """Publishes Twist messages to control robot velocity."""
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.GPIO_ERROR, RobotState.AIRTABLE_ERROR] or not self.cmd_vel_pub:
            twist_msg = Twist()
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            if self.cmd_vel_pub: self.cmd_vel_pub.publish(twist_msg)
            return

        twist_msg = Twist()
        twist_msg.linear.x = float(linear_x)
        twist_msg.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(twist_msg)

    def stop_moving(self):
        """Stops the robot movement reliably."""
        # self.get_logger().info("Stopping robot...") # Reduced verbosity
        for _ in range(3):
            self.move_robot(0.0, 0.0)
            time.sleep(0.02)

    def read_ir_sensors(self):
        """Reads the state of the IR line sensors. Returns (left_on_line, right_on_line)."""
        try:
            left_val = GPIO.input(LEFT_IR_PIN)
            right_val = GPIO.input(RIGHT_IR_PIN)
            left_on_line = (left_val == IR_LINE_DETECT_SIGNAL)
            right_on_line = (right_val == IR_LINE_DETECT_SIGNAL)
            return left_on_line, right_on_line
        except Exception as e:
            self.get_logger().error(f"IR sensor read error: {e}")
            return False, False # Assume off line on error

    def check_for_station_color(self, frame, target_idx):
        """
        Detects station color markers (now common green) in camera frame.
        Returns: (detected_flag, display_frame, mask_frame)
        """
        detected_flag = False
        display_frame = frame.copy()
        mask_frame = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        cv2.putText(display_frame, f"State: {self.state.name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if target_idx not in STATION_COLORS_HSV:
            cv2.putText(display_frame, f"Target: Invalid ({target_idx})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return False, display_frame, mask_frame

        # Get specific station name, but use common color info
        color_info = STATION_COLORS_HSV[target_idx]
        target_name = color_info['name']
        # Use common color definitions directly
        target_bgr = COMMON_COLOR_BGR
        lower_bound = np.array(COMMON_HSV_LOWER)
        upper_bound = np.array(COMMON_HSV_UPPER)

        try:
            hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            color_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
            mask_frame = color_mask
            detected_pixels = cv2.countNonZero(color_mask)

            # Visualization
            text = f"Target: {target_name} ({detected_pixels} px)"
            cv2.putText(display_frame, text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, target_bgr, 2) # Use target_bgr (green)

            # Detection Logic
            current_time = time.time()
            if detected_pixels > COLOR_DETECTION_THRESHOLD and \
               (current_time - self.last_color_detection_times.get(target_idx, 0.0) > COLOR_COOLDOWN_SEC):
                self.last_color_detection_times[target_idx] = current_time
                detected_flag = True
                cv2.putText(display_frame, "DETECTED!", (frame.shape[1] // 2 - 50, frame.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                self.get_logger().info(f"Color detected for station {target_idx} ({target_name})")

            return detected_flag, display_frame, mask_frame

        except cv2.error as cv2_e:
             self.get_logger().error(f"OpenCV error during color detection: {cv2_e}")
             return False, display_frame, np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        except Exception as e:
            self.get_logger().error(f"Unexpected error in color detection: {e}", exc_info=True)
            return False, display_frame, np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    def play_sound(self, notes):
        """Plays a sequence of notes through the robot's speaker."""
        if not self.audio_publisher:
            self.get_logger().warning("Audio publisher not available, cannot play sound.")
            return
        note_msg = AudioNoteVector()
        for frequency, duration_ms in notes:
            note = AudioNote()
            note.frequency = frequency
            duration_sec = int(duration_ms / 1000)
            duration_nsec = int((duration_ms % 1000) * 1e6)
            note.max_runtime = Duration(sec=duration_sec, nanosec=duration_nsec)
            note_msg.notes.append(note)
        self.audio_publisher.publish(note_msg)

    def cleanup_gpio(self):
        """Cleanup GPIO pins."""
        self.get_logger().info("Cleaning up GPIO pins...")
        try:
            GPIO.cleanup()
            self.get_logger().info("GPIO cleanup successful.")
        except Exception as e:
            self.get_logger().error(f"Error during GPIO cleanup: {e}")

    def cleanup_hardware(self):
        """Clean up GPIO and Camera"""
        self.cleanup_gpio()
        # Cleanup Camera
        if self.picam2:
            try:
                self.get_logger().info("Stopping camera...")
                self.picam2.stop()
            except Exception as cam_e:
                self.get_logger().error(f"Error stopping camera: {cam_e}")

    def control_loop(self):
        """Main state machine and control logic for the robot."""
        # --- Check for fatal errors first ---
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.GPIO_ERROR]:
            self.stop_moving()
            return

        display_frame = None
        mask_frame = None
        color_detected = False

        # --- Camera Capture and Processing ---
        if self.picam2 and self.state != RobotState.CAMERA_ERROR:
            try:
                raw_frame = self.picam2.capture_array()
                if self.target_station_index != -1:
                    color_detected, display_frame, mask_frame = self.check_for_station_color(raw_frame, self.target_station_index)
                else:
                    display_frame = raw_frame.copy()
                    cv2.putText(display_frame, f"State: {self.state.name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_frame, "Target: None", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            except Exception as e:
                self.get_logger().error(f"Camera error in control loop: {e}", exc_info=True)
                self.state = RobotState.CAMERA_ERROR
                self.stop_moving()
                return
        else:
            # No camera, cannot do visual stop
            if self.state != RobotState.CAMERA_ERROR:
                 self.get_logger().warning("Camera not available, cannot perform visual checks.")


        # --- State Machine Logic ---
        try:
            # State Transition based on Color Detection
            if color_detected and self.state == RobotState.MOVING_TO_STATION:
                self.get_logger().info(f"Color marker detected for station {self.target_station_index}. Arriving.")
                self.play_sound([(523, 100), (659, 150)])
                self.stop_moving()
                self.state = RobotState.ARRIVED_AT_STATION
                # Skip rest of logic this cycle

            # --- Handle States ---
            elif self.state == RobotState.IDLE:
                # self.get_logger().info("State: IDLE") # Less verbose
                self.stop_moving()
                self.current_order = None
                self.station_sequence = []
                self.current_sequence_index = -1
                self.target_station_index = -1
                self.current_order = self.fetch_order_from_airtable()
                if self.current_order:
                    if self.state == RobotState.AIRTABLE_ERROR:
                         self.get_logger().error("Airtable error during fetch. Halting.")
                    else:
                        self.get_logger().info(f"Order '{self.current_order['order_name']}' received. Planning route...")
                        self.state = RobotState.PLANNING_ROUTE
                else:
                    if self.state != RobotState.AIRTABLE_ERROR:
                        # self.get_logger().info("No pending orders found. Entering ALL_ORDERS_COMPLETE state.") # Less verbose
                        self.state = RobotState.ALL_ORDERS_COMPLETE

            elif self.state == RobotState.PLANNING_ROUTE:
                self.get_logger().info("State: PLANNING_ROUTE")
                if not self.current_order:
                    self.get_logger().error("PLANNING_ROUTE: No current order! Returning to IDLE.")
                    self.state = RobotState.IDLE
                    return

                self.station_sequence = []
                order_status = self.current_order["station_status"]

                # --- Determine Sequence ---
                # 1. Cooking 1 (Mandatory Start)
                if order_status.get(AIRTABLE_COOKING_1_STATUS_FIELD) == STATUS_WAITING:
                    self.station_sequence.append(STATION_FIELD_TO_INDEX[AIRTABLE_COOKING_1_STATUS_FIELD])
                else:
                    self.get_logger().error(f"Order '{self.current_order['order_name']}' fetched, but '{AIRTABLE_COOKING_1_STATUS_FIELD}' != WAITING. Aborting.")
                    self.state = RobotState.IDLE
                    self.current_order = None
                    return

                # 2. Cooking 2
                if order_status.get(AIRTABLE_COOKING_2_STATUS_FIELD) == STATUS_WAITING:
                    idx2 = STATION_FIELD_TO_INDEX[AIRTABLE_COOKING_2_STATUS_FIELD]
                    if idx2 not in STATION_COLORS_HSV: # Check if added to color map
                         self.get_logger().warning(f"Station {idx2} ({AIRTABLE_COOKING_2_STATUS_FIELD}) required but has no color definition. Visual stop impossible.")
                    self.station_sequence.append(idx2) # Add regardless for sequence

                # 3. Toppings
                topping_fields = [AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD, AIRTABLE_WHIPPED_CREAM_STATUS_FIELD, AIRTABLE_SPRINKLES_STATUS_FIELD]
                for field in topping_fields:
                    if order_status.get(field) == STATUS_WAITING:
                        if field in STATION_FIELD_TO_INDEX:
                            idx = STATION_FIELD_TO_INDEX[field]
                            if idx not in STATION_COLORS_HSV:
                                self.get_logger().warning(f"Station {idx} ({field}) required but has no color definition. Visual stop impossible.")
                            self.station_sequence.append(idx) # Add regardless for sequence
                        else:
                             self.get_logger().warning(f"Topping field '{field}' required but not in STATION_FIELD_TO_INDEX. Skipping.")

                # 4. Pickup Station (Mandatory End)
                pickup_idx = STATION_FIELD_TO_INDEX[AIRTABLE_PICKUP_STATUS_FIELD]
                if pickup_idx not in STATION_COLORS_HSV:
                     self.get_logger().error(f"Pickup station {pickup_idx} ({AIRTABLE_PICKUP_STATUS_FIELD}) has no color definition! Visual stop impossible.")
                self.station_sequence.append(pickup_idx) # Add regardless

                # --- Finalize ---
                if not self.station_sequence or self.station_sequence[0] != STATION_FIELD_TO_INDEX[AIRTABLE_COOKING_1_STATUS_FIELD]:
                    self.get_logger().error("Route planning failed (empty or doesn't start with Cooking 1)! Aborting order.")
                    self.state = RobotState.IDLE
                    self.current_order = None
                else:
                    self.current_sequence_index = 0
                    self.target_station_index = self.station_sequence[self.current_sequence_index]
                    self.get_logger().info(f"Route planned: {self.station_sequence}. Next target: Station {self.target_station_index}")
                    self.state = RobotState.MOVING_TO_STATION
                    self.play_sound([(440,100), (550,100), (660, 100)])

            elif self.state == RobotState.MOVING_TO_STATION:
                # Line following
                if self.target_station_index == -1 or self.current_sequence_index >= len(self.station_sequence):
                     self.get_logger().error(f"MOVING_TO_STATION: Invalid target/index. Stopping.")
                     self.stop_moving(); self.state = RobotState.ERROR; return

                left_on, right_on = self.read_ir_sensors()

                if left_on and right_on:
                    self.move_robot(BASE_DRIVE_SPEED, 0.0)
                elif left_on and not right_on:
                    self.move_robot(BASE_DRIVE_SPEED * TURN_FACTOR, -BASE_ROTATE_SPEED)
                elif not left_on and right_on:
                    self.move_robot(BASE_DRIVE_SPEED * TURN_FACTOR, BASE_ROTATE_SPEED)
                else: # Lost line
                    current_time = time.time()
                    if int(current_time) % 2 == 0: self.move_robot(0.0, LOST_LINE_ROTATE_SPEED)
                    else: self.move_robot(0.0, -LOST_LINE_ROTATE_SPEED)

            elif self.state == RobotState.ARRIVED_AT_STATION:
                self.get_logger().info(f"State: ARRIVED_AT_STATION ({self.target_station_index})")
                self.stop_moving()

                if self.current_sequence_index < 0 or self.current_sequence_index >= len(self.station_sequence):
                    self.get_logger().error(f"ARRIVED_AT_STATION: Invalid sequence index {self.current_sequence_index}. Stopping."); self.state = RobotState.ERROR; return

                current_station_idx = self.station_sequence[self.current_sequence_index]
                if current_station_idx not in STATION_INDEX_TO_FIELD:
                    self.get_logger().error(f"ARRIVED_AT_STATION: No Airtable field for index {current_station_idx}. Stopping."); self.state = RobotState.ERROR; return

                station_field = STATION_INDEX_TO_FIELD[current_station_idx]
                self.get_logger().info(f"At {station_field}. Updating Airtable to ARRIVED ({STATUS_ARRIVED}).")

                if self.update_station_status(self.current_order["record_id"], station_field, STATUS_ARRIVED):
                    self.wait_start_time = time.time()
                    self.state = RobotState.WAITING_FOR_STATION_COMPLETION
                    self.get_logger().info(f"Status updated. Waiting for {station_field} to be DONE.")
                else: # update status already set state to AIRTABLE_ERROR
                    self.get_logger().error(f"Failed to update Airtable status for {station_field}.")

            elif self.state == RobotState.WAITING_FOR_STATION_COMPLETION:
                elapsed_wait_time = time.time() - self.wait_start_time
                if elapsed_wait_time > STATION_WAIT_TIMEOUT_SEC:
                    self.get_logger().warning(f"WAITING state TIMEOUT ({elapsed_wait_time:.1f}s > {STATION_WAIT_TIMEOUT_SEC}s) for station {self.target_station_index}. Moving to TIMED_OUT state.")
                    self.play_sound([(330, 500), (220, 500)])
                    self.state = RobotState.STATION_TIMED_OUT
                    return

                # Check Airtable periodically using AIRTABLE_POLL_RATE approx
                if time.time() - getattr(self, '_last_airtable_check_time', 0) >= AIRTABLE_POLL_RATE:
                    self._last_airtable_check_time = time.time() # Update last check time

                    if self.current_sequence_index < 0 or self.current_sequence_index >= len(self.station_sequence):
                        self.get_logger().error(f"WAITING_FOR_STATION: Invalid sequence index. Stopping."); self.state = RobotState.ERROR; return
                    current_station_idx = self.station_sequence[self.current_sequence_index]
                    if current_station_idx not in STATION_INDEX_TO_FIELD:
                        self.get_logger().error(f"WAITING_FOR_STATION: No Airtable field for index {current_station_idx}. Stopping."); self.state = RobotState.ERROR; return

                    station_field = STATION_INDEX_TO_FIELD[current_station_idx]
                    # self.get_logger().debug(f"Checking Airtable if {station_field} is DONE...") # Verbose

                    if self.wait_for_station_completion(self.current_order["record_id"], station_field):
                        self.get_logger().info(f"Station {current_station_idx} ({station_field}) reported DONE ({STATUS_DONE}).")
                        self.play_sound([(659, 150), (784, 200)])

                        self.current_sequence_index += 1 # Advance sequence
                        if self.current_sequence_index >= len(self.station_sequence):
                            self.get_logger().info("Completed all stations in sequence.")
                            self.state = RobotState.ORDER_COMPLETE
                        else:
                            self.target_station_index = self.station_sequence[self.current_sequence_index]
                            self.get_logger().info(f"Moving to next station: {self.target_station_index} ({STATION_INDEX_TO_FIELD.get(self.target_station_index, 'Unknown')})")
                            self.state = RobotState.MOVING_TO_STATION
                    # else: Stay waiting

            elif self.state == RobotState.ORDER_COMPLETE:
                self.get_logger().info(f"State: ORDER_COMPLETE for '{self.current_order['order_name']}'")
                self.play_sound([(784, 150), (880, 150), (1047, 250)])
                self.stop_moving()
                self.current_order = None # Reset for next
                self.station_sequence = []
                self.current_sequence_index = -1
                self.target_station_index = -1
                self.state = RobotState.IDLE # Go back to look for more work
                self.get_logger().info("Order finished. Returning to IDLE state.")

            elif self.state == RobotState.ALL_ORDERS_COMPLETE:
                self.get_logger().info("State: ALL_ORDERS_COMPLETE - No orders found. Waiting...")
                self.play_sound([(440, 200), (440, 200)])
                self.stop_moving()
                time.sleep(5.0) # Wait before checking again
                self.state = RobotState.IDLE

            elif self.state == RobotState.STATION_TIMED_OUT:
                self.get_logger().error("State: STATION_TIMED_OUT - Station did not complete in time.")
                self.stop_moving()
                self.get_logger().error(f"Aborting order '{self.current_order['order_name']}' due to station timeout.")
                self.current_order = None; self.station_sequence = []; self.current_sequence_index = -1; self.target_station_index = -1
                self.state = RobotState.IDLE # Go back to fetch next order

            elif self.state == RobotState.AIRTABLE_ERROR:
                 self.get_logger().error("State: AIRTABLE_ERROR - Halting due to Airtable communication issue.")
                 self.play_sound([(330, 300), (330, 300), (330, 300)])
                 self.stop_moving() # Remain in this state

        except Exception as e:
            self.get_logger().error(f"Unhandled exception in state machine logic: {e}", exc_info=True)
            self.state = RobotState.ERROR
            self.stop_moving()

        # --- Display Update ---
        finally:
            if self.debug_windows:
                try:
                    if display_frame is not None: cv2.imshow("Camera Feed", display_frame)
                    if mask_frame is not None: cv2.imshow("Color Detection Mask", mask_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.get_logger().info("Quit key 'q' pressed. Initiating shutdown.")
                        self.state = RobotState.ERROR
                        self.stop_moving()
                        # Request ROS shutdown (cleaner way)
                        self.get_logger().info("Requesting ROS shutdown...")
                        rclpy.try_shutdown()

                except Exception as display_e:
                    self.get_logger().error(f"Error updating OpenCV windows: {display_e}")
                    # self.debug_windows = False # Optionally disable if problematic
                    # cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = None
    try:
        node = PancakeRobotNode()
        if node.state not in [RobotState.GPIO_ERROR, RobotState.CAMERA_ERROR, RobotState.ERROR]:
            executor = SingleThreadedExecutor()
            executor.add_node(node)
            node.get_logger().info("Starting ROS2 executor spin...")
            executor.spin()
        else:
             node.get_logger().fatal(f"Node initialization failed in state {node.state.name}. Shutting down.")

    except KeyboardInterrupt:
        if node: node.get_logger().info("KeyboardInterrupt received.")
        else: print("KeyboardInterrupt received during node initialization.")
    except Exception as e:
        if node: node.get_logger().fatal(f"Unhandled exception: {e}", exc_info=True)
        else: print(f"FATAL Exception during node initialization: {e}")
    finally:
        if node:
            node.get_logger().info("Initiating final cleanup...")
            node.stop_moving()
            if executor: executor.shutdown()
            node.cleanup_hardware() # Cleans GPIO and Camera
            if node.debug_windows:
                 node.get_logger().info("Closing OpenCV windows...")
                 cv2.destroyAllWindows()
            node.get_logger().info("Destroying node...")
            node.destroy_node()
        else:
            print("Node object not fully created, skipping node-specific cleanup.")

        if rclpy.ok():
             print("Shutting down rclpy...")
             rclpy.shutdown()
        print("Shutdown complete.")

if __name__ == '__main__':
    main()