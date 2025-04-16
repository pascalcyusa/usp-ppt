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
    AIRTABLE_COOKING_2_STATUS_FIELD: 2,
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
    0: {"name": "Pickup Station", "hsv_lower": COMMON_HSV_LOWER, "hsv_upper": COMMON_HSV_UPPER, "color_bgr": COMMON_COLOR_BGR},
    1: {"name": "Cooking Station 1", "hsv_lower": COMMON_HSV_LOWER, "hsv_upper": COMMON_HSV_UPPER, "color_bgr": COMMON_COLOR_BGR},
    2: {"name": "Cooking Station 2", "hsv_lower": COMMON_HSV_LOWER, "hsv_upper": COMMON_HSV_UPPER, "color_bgr": COMMON_COLOR_BGR},
    3: {"name": "Chocolate Chips", "hsv_lower": COMMON_HSV_LOWER, "hsv_upper": COMMON_HSV_UPPER, "color_bgr": COMMON_COLOR_BGR},
    4: {"name": "Whipped Cream", "hsv_lower": COMMON_HSV_LOWER, "hsv_upper": COMMON_HSV_UPPER, "color_bgr": COMMON_COLOR_BGR},
    5: {"name": "Sprinkles", "hsv_lower": COMMON_HSV_LOWER, "hsv_upper": COMMON_HSV_UPPER, "color_bgr": COMMON_COLOR_BGR},
}

# --- Navigation & Control Parameters ---
AIRTABLE_POLL_RATE = 2.0          # Seconds - Approx rate to check Airtable status

# --- Increased speeds slightly for potentially better line following ---
# --- TUNING REQUIRED: Adjust these values based on your robot's behavior ---
BASE_DRIVE_SPEED = 0.03         # m/s - Forward speed (Increased from 0.01)
BASE_ROTATE_SPEED = 0.3         # rad/s - Turning speed (Increased from 0.2)
# --- End Tuning Section ---

TURN_FACTOR = 0.7               # Multiplier for speed reduction during turns (0.0-1.0)
LOST_LINE_ROTATE_SPEED = 0.15   # rad/s - Speed for rotation when line is lost (Slightly increased)

COLOR_DETECTION_THRESHOLD = 2000  # Min pixels of target color to trigger detection
COLOR_COOLDOWN_SEC = 5.0        # Min seconds before detecting the *same* station color again
STATION_WAIT_TIMEOUT_SEC = 120.0  # Max seconds to wait for Airtable status 99 before failing
LEAVING_STATION_DURATION_SEC = 2.0 # Seconds to drive forward after leaving a station


class RobotState(Enum):
    IDLE = auto()
    FETCHING_ORDER = auto() # Not strictly needed, could merge into IDLE
    PLANNING_ROUTE = auto()
    LEAVING_STATION = auto()  # Drive away from previous station marker
    MOVING_TO_STATION = auto() # Actively line following and looking for color
    ARRIVED_AT_STATION = auto() # Stopped at color marker
    WAITING_FOR_STATION_COMPLETION = auto() # Waiting for Airtable status 99
    STATION_TIMED_OUT = auto() # Waiting took too long
    ORDER_COMPLETE = auto() # Finished last station in sequence
    ALL_ORDERS_COMPLETE = auto() # No orders found
    ERROR = auto() # General unrecoverable error
    CAMERA_ERROR = auto()
    AIRTABLE_ERROR = auto()
    GPIO_ERROR = auto()

class PancakeRobotNode(Node):
    def __init__(self):
        super().__init__('pancake_robot_node')
        self.get_logger().info("Pancake Robot Node Initializing...")

        # State and Order Variables
        self.state = RobotState.IDLE
        self.current_order = None
        self.station_sequence = []
        self.current_sequence_index = -1
        self.target_station_index = -1

        # Timers and Cooldowns
        self.last_color_detection_times = {idx: 0.0 for idx in STATION_COLORS_HSV.keys()}
        self.wait_start_time = 0.0
        self.leaving_station_start_time = 0.0
        self._last_airtable_check_time = 0.0 # For rate-limiting Airtable checks

        # Hardware and Display
        self.picam2 = None
        self.debug_windows = True

        # ROS Publishers/Clients
        self.cmd_vel_pub = None
        self.audio_publisher = None
        self.drive_client = None
        self.rotate_client = None

        # Initialization
        self._init_hardware()
        if self.state in [RobotState.GPIO_ERROR, RobotState.CAMERA_ERROR]:
            self.get_logger().fatal("Hardware initialization failed. Node cannot operate.")
            return

        self._init_ros2()
        if self.state == RobotState.ERROR:
             self.get_logger().fatal("ROS2 component initialization failed. Node cannot operate.")
             self.cleanup_hardware()
             return

        # Control Loop Timer (20 Hz)
        self.control_timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info("Pancake Robot Node Initialized and Ready.")
        self.play_sound([(440, 150), (550, 200)]) # Initial sound

    def _init_hardware(self):
        """Initialize GPIO and Camera"""
        # GPIO
        try:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(LEFT_IR_PIN, GPIO.IN)
            GPIO.setup(RIGHT_IR_PIN, GPIO.IN)
            self.get_logger().info(f"GPIO initialized (Pins: L={LEFT_IR_PIN}, R={RIGHT_IR_PIN}). Expecting {IR_LINE_DETECT_SIGNAL} on line.")
        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to initialize GPIO: {e}", exc_info=True)
            self.state = RobotState.GPIO_ERROR
            return
        # Camera
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(main={"size": CAMERA_RESOLUTION}, transform=CAMERA_TRANSFORM)
            self.picam2.configure(config)
            self.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous, "LensPosition": 0.0})
            self.picam2.start()
            time.sleep(2)
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
            # Check if audio publisher was created
            if not self.audio_publisher:
                 raise RuntimeError("Audio publisher creation failed.")
            self.get_logger().info("ROS2 publishers and action clients initialized.")
        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to initialize ROS2 components: {e}", exc_info=True)
            self.state = RobotState.ERROR

    # --- Airtable Functions (fetch, update, wait) ---
    # (These functions remain unchanged from the previous version)
    def fetch_order_from_airtable(self):
        """Fetches the oldest order where Cooking 1 is WAITING and Pickup is WAITING."""
        # self.get_logger().info("Attempting to fetch order from Airtable...") # Reduced verbosity
        try:
            params = { "maxRecords": 1, "filterByFormula": f"AND({{{AIRTABLE_COOKING_1_STATUS_FIELD}}}=0, {{{AIRTABLE_PICKUP_STATUS_FIELD}}}=0)", "sort[0][field]": AIRTABLE_ORDER_NAME_COLUMN, "sort[0][direction]": "asc" }
            response = requests.get(url=AIRTABLE_URL, headers=AIRTABLE_HEADERS, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            records = data.get("records", [])
            if not records:
                # self.get_logger().info("No pending orders found matching criteria.") # Reduced verbosity
                return None
            record = records[0]; record_id = record.get("id"); fields = record.get("fields", {}); order_name = fields.get(AIRTABLE_ORDER_NAME_COLUMN)
            if not record_id or not order_name:
                self.get_logger().error(f"Airtable record missing ID or Order Name: {record}")
                return None
            self.get_logger().info(f"Fetched order '{order_name}' (Record ID: {record_id}).")
            return { "record_id": record_id, "order_name": order_name, "station_status": { field: fields.get(field, 0) for field in STATION_FIELD_TO_INDEX.keys() } }
        except requests.exceptions.Timeout: self.get_logger().error("Airtable fetch timed out."); self.state = RobotState.AIRTABLE_ERROR; return None
        except requests.exceptions.RequestException as e: self.get_logger().error(f"Airtable fetch error: {e}"); self.state = RobotState.AIRTABLE_ERROR; return None
        except Exception as e: self.get_logger().error(f"Unexpected error during Airtable fetch: {e}", exc_info=True); self.state = RobotState.AIRTABLE_ERROR; return None

    def update_station_status(self, record_id, station_field_name, new_status_code):
        """Updates a specific station's status for an order in Airtable."""
        if not record_id or not station_field_name: self.get_logger().error("Airtable update error: Missing record_id or station_field_name"); return False
        update_data = {"fields": {station_field_name: new_status_code}}; url = f"{AIRTABLE_URL}/{record_id}"
        # self.get_logger().info(f"Updating Airtable: {url} -> {station_field_name} = {new_status_code}") # Reduced verbosity
        try:
            response = requests.patch(url=url, headers=AIRTABLE_HEADERS, json=update_data, timeout=10); response.raise_for_status()
            # self.get_logger().info(f"Airtable update successful for {station_field_name}.") # Reduced verbosity
            return True
        except requests.exceptions.Timeout: self.get_logger().error(f"Airtable update timed out for {station_field_name}."); self.state = RobotState.AIRTABLE_ERROR; return False
        except requests.exceptions.RequestException as e: self.get_logger().error(f"Airtable update error for {station_field_name}: {e}"); self.state = RobotState.AIRTABLE_ERROR; return False
        except Exception as e: self.get_logger().error(f"Unexpected error during Airtable update: {e}", exc_info=True); self.state = RobotState.AIRTABLE_ERROR; return False

    def wait_for_station_completion(self, record_id, station_field_name):
        """Checks if a station's status in Airtable is DONE (99). Returns True/False."""
        if not record_id or not station_field_name: self.get_logger().error("Airtable status check error: Missing record_id or station_field_name"); return False
        url = f"{AIRTABLE_URL}/{record_id}"
        try:
            response = requests.get(url=url, headers=AIRTABLE_HEADERS, timeout=10); response.raise_for_status()
            data = response.json(); current_status = data.get('fields', {}).get(station_field_name)
            return current_status == STATUS_DONE
        except requests.exceptions.Timeout: self.get_logger().warning(f"Airtable status check timed out for {station_field_name}."); return False
        except requests.exceptions.RequestException as e: self.get_logger().error(f"Airtable status check error ({station_field_name}): {e}"); return False
        except Exception as e: self.get_logger().error(f"Unexpected error during Airtable status check: {e}", exc_info=True); return False

    # --- Robot Movement and Sensors ---
    def move_robot(self, linear_x, angular_z):
        """Publishes Twist messages."""
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.GPIO_ERROR, RobotState.AIRTABLE_ERROR] or not self.cmd_vel_pub:
            # Ensure stopped in error states
            twist_msg = Twist(); twist_msg.linear.x = 0.0; twist_msg.angular.z = 0.0
            if self.cmd_vel_pub: self.cmd_vel_pub.publish(twist_msg)
            return
        twist_msg = Twist(); twist_msg.linear.x = float(linear_x); twist_msg.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(twist_msg)

    def stop_moving(self):
        """Stops the robot movement reliably."""
        for _ in range(3): self.move_robot(0.0, 0.0); time.sleep(0.02)

    def read_ir_sensors(self):
        """Reads IR sensors. Returns (left_on_line, right_on_line)."""
        try:
            left_val = GPIO.input(LEFT_IR_PIN); right_val = GPIO.input(RIGHT_IR_PIN)
            return (left_val == IR_LINE_DETECT_SIGNAL), (right_val == IR_LINE_DETECT_SIGNAL)
        except Exception as e:
            self.get_logger().error(f"IR sensor read error: {e}", exc_info=True) # Log traceback
            return False, False

    # --- Color Detection ---
    def check_for_station_color(self, frame, target_idx):
        """Detects station color markers. Returns (detected_flag, display_frame, mask_frame)."""
        # (This function remains unchanged from the previous version)
        detected_flag = False
        display_frame = frame.copy()
        mask_frame = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        cv2.putText(display_frame, f"State: {self.state.name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if target_idx not in STATION_COLORS_HSV:
            cv2.putText(display_frame, f"Target: Invalid ({target_idx})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return False, display_frame, mask_frame
        color_info = STATION_COLORS_HSV[target_idx]; target_name = color_info['name']
        target_bgr = COMMON_COLOR_BGR; lower_bound = np.array(COMMON_HSV_LOWER); upper_bound = np.array(COMMON_HSV_UPPER)
        try:
            hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            color_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
            mask_frame = color_mask; detected_pixels = cv2.countNonZero(color_mask)
            text = f"Target: {target_name} ({detected_pixels} px)"
            cv2.putText(display_frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, target_bgr, 2)
            current_time = time.time()
            if detected_pixels > COLOR_DETECTION_THRESHOLD and (current_time - self.last_color_detection_times.get(target_idx, 0.0) > COLOR_COOLDOWN_SEC):
                self.last_color_detection_times[target_idx] = current_time; detected_flag = True
                cv2.putText(display_frame, "DETECTED!", (frame.shape[1] // 2 - 50, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                # self.get_logger().info(f"Color detected for station {target_idx} ({target_name})") # Reduced verbosity
            return detected_flag, display_frame, mask_frame
        except cv2.error as cv2_e: self.get_logger().error(f"OpenCV error during color detection: {cv2_e}"); return False, display_frame, np.zeros_like(mask_frame)
        except Exception as e: self.get_logger().error(f"Unexpected error in color detection: {e}", exc_info=True); return False, display_frame, np.zeros_like(mask_frame)

    # --- Sound ---
    def play_sound(self, notes):
        """Plays a sequence of notes."""
        # Check if publisher exists and is valid
        if not hasattr(self, 'audio_publisher') or self.audio_publisher is None:
            self.get_logger().warning("Audio publisher not initialized, cannot play sound.")
            return
        if not self.audio_publisher.is_activated:
             self.get_logger().warning("Audio publisher exists but is not activated, cannot play sound.")
             return

        note_msg = AudioNoteVector()
        note_list_str = [] # For logging
        for frequency, duration_ms in notes:
            if frequency <= 0 or duration_ms <= 0:
                 self.get_logger().warning(f"Skipping invalid note: freq={frequency}, dur={duration_ms}")
                 continue
            note = AudioNote()
            note.frequency = int(frequency)
            duration_sec = int(duration_ms / 1000)
            duration_nsec = int((duration_ms % 1000) * 1e6)
            note.max_runtime = Duration(sec=duration_sec, nanosec=duration_nsec)
            note_msg.notes.append(note)
            note_list_str.append(f"({frequency}Hz,{duration_ms}ms)")

        if not note_msg.notes:
             self.get_logger().warning("No valid notes provided to play_sound.")
             return

        self.get_logger().info(f"Publishing sound: {', '.join(note_list_str)}") # Log the notes being played
        try:
            self.audio_publisher.publish(note_msg)
        except Exception as e:
             self.get_logger().error(f"Failed to publish audio command: {e}", exc_info=True)


    # --- Cleanup ---
    def cleanup_gpio(self):
        """Cleanup GPIO pins."""
        # self.get_logger().info("Cleaning up GPIO pins...") # Reduced verbosity
        try: GPIO.cleanup()
        except Exception as e: self.get_logger().error(f"Error during GPIO cleanup: {e}")

    def cleanup_hardware(self):
        """Clean up GPIO and Camera"""
        self.cleanup_gpio()
        if self.picam2:
            try:
                # self.get_logger().info("Stopping camera...") # Reduced verbosity
                self.picam2.stop()
            except Exception as cam_e: self.get_logger().error(f"Error stopping camera: {cam_e}")

    # --- Main Control Loop ---
    def control_loop(self):
        """Main state machine and control logic for the robot."""
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.GPIO_ERROR]:
            self.stop_moving(); return

        display_frame = None; mask_frame = None; color_detected = False
        process_color = (self.state == RobotState.MOVING_TO_STATION)

        # --- Camera Handling ---
        if self.picam2 and self.state != RobotState.CAMERA_ERROR:
            try:
                raw_frame = self.picam2.capture_array()
                if self.target_station_index != -1:
                    _detected_flag, display_frame, mask_frame = self.check_for_station_color(raw_frame, self.target_station_index)
                    if process_color: color_detected = _detected_flag
                else:
                    display_frame = raw_frame.copy()
                    cv2.putText(display_frame, f"State: {self.state.name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_frame, "Target: None", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            except Exception as e:
                self.get_logger().error(f"Camera error in control loop: {e}", exc_info=True); self.state = RobotState.CAMERA_ERROR; self.stop_moving(); return
        # --- End Camera Handling ---

        # --- State Machine ---
        try:
            # --- Handle States ---
            if self.state == RobotState.IDLE:
                self.stop_moving(); self.current_order = None; self.station_sequence = []; self.current_sequence_index = -1; self.target_station_index = -1
                self.current_order = self.fetch_order_from_airtable()
                if self.current_order:
                    if self.state != RobotState.AIRTABLE_ERROR:
                        self.get_logger().info(f"Order '{self.current_order['order_name']}' received. Planning route...")
                        self.state = RobotState.PLANNING_ROUTE
                else:
                    if self.state != RobotState.AIRTABLE_ERROR: self.state = RobotState.ALL_ORDERS_COMPLETE

            elif self.state == RobotState.PLANNING_ROUTE:
                # self.get_logger().info("State: PLANNING_ROUTE") # Reduced verbosity
                if not self.current_order: self.get_logger().error("PLANNING_ROUTE: No order!"); self.state = RobotState.IDLE; return
                self.station_sequence = []; order_status = self.current_order["station_status"]
                # 1. Cooking 1
                if order_status.get(AIRTABLE_COOKING_1_STATUS_FIELD) == STATUS_WAITING: self.station_sequence.append(STATION_FIELD_TO_INDEX[AIRTABLE_COOKING_1_STATUS_FIELD])
                else: self.get_logger().error(f"Order '{self.current_order['order_name']}' invalid start state. Aborting."); self.state = RobotState.IDLE; self.current_order = None; return
                # 2. Cooking 2
                if order_status.get(AIRTABLE_COOKING_2_STATUS_FIELD) == STATUS_WAITING: idx = STATION_FIELD_TO_INDEX[AIRTABLE_COOKING_2_STATUS_FIELD]; self.station_sequence.append(idx)
                # 3. Toppings
                for field in [AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD, AIRTABLE_WHIPPED_CREAM_STATUS_FIELD, AIRTABLE_SPRINKLES_STATUS_FIELD]:
                    if order_status.get(field) == STATUS_WAITING:
                        if field in STATION_FIELD_TO_INDEX: idx = STATION_FIELD_TO_INDEX[field]; self.station_sequence.append(idx)
                        else: self.get_logger().warning(f"Topping '{field}' required but not in index map. Skipping.")
                # 4. Pickup
                self.station_sequence.append(STATION_FIELD_TO_INDEX[AIRTABLE_PICKUP_STATUS_FIELD])
                # Finalize
                if not self.station_sequence: self.get_logger().error("Route planning failed (empty)!"); self.state = RobotState.IDLE; self.current_order = None
                else:
                    self.current_sequence_index = 0; self.target_station_index = self.station_sequence[0]
                    self.get_logger().info(f"Route planned: {self.station_sequence}. Next target: Station {self.target_station_index}")
                    self.state = RobotState.MOVING_TO_STATION
                    self.play_sound([(440,100), (550,100), (660, 100)]) # Planning complete sound

            elif self.state == RobotState.LEAVING_STATION:
                elapsed_leaving_time = time.time() - self.leaving_station_start_time
                if elapsed_leaving_time < LEAVING_STATION_DURATION_SEC:
                    # self.get_logger().debug(f"Leaving station... {elapsed_leaving_time:.1f}s / {LEAVING_STATION_DURATION_SEC}s") # Verbose
                    self.move_robot(BASE_DRIVE_SPEED, 0.0) # Drive straight
                else:
                    self.get_logger().info(f"Finished leaving station. Now moving towards station {self.target_station_index}.")
                    self.stop_moving()
                    self.state = RobotState.MOVING_TO_STATION

            elif self.state == RobotState.MOVING_TO_STATION:
                # Check for color arrival *first*
                if color_detected:
                    self.get_logger().info(f"Color marker detected for station {self.target_station_index}. Arriving.")
                    self.play_sound([(523, 100), (659, 150)]) # Arrival sound
                    self.stop_moving(); self.state = RobotState.ARRIVED_AT_STATION; return

                # If no color, perform line following
                if self.target_station_index == -1 or self.current_sequence_index >= len(self.station_sequence):
                     self.get_logger().error(f"MOVING_TO_STATION: Invalid target/index ({self.target_station_index}/{self.current_sequence_index}). Stopping."); self.stop_moving(); self.state = RobotState.ERROR; return

                # --- Line Following Logic ---
                left_on, right_on = self.read_ir_sensors()
                if left_on and right_on: # On line -> Drive straight
                    self.move_robot(BASE_DRIVE_SPEED, 0.0)
                elif left_on and not right_on: # Veer right -> Turn Left (Negative Z)
                    self.move_robot(BASE_DRIVE_SPEED * TURN_FACTOR, -BASE_ROTATE_SPEED)
                elif not left_on and right_on: # Veer left -> Turn Right (Positive Z)
                    self.move_robot(BASE_DRIVE_SPEED * TURN_FACTOR, BASE_ROTATE_SPEED)
                else: # Lost line -> Search pattern
                    current_time = time.time()
                    if int(current_time) % 2 == 0: self.move_robot(0.0, LOST_LINE_ROTATE_SPEED) # Turn right
                    else: self.move_robot(0.0, -LOST_LINE_ROTATE_SPEED) # Turn left
                # --- End Line Following Logic ---

            elif self.state == RobotState.ARRIVED_AT_STATION:
                # self.get_logger().info(f"State: ARRIVED_AT_STATION ({self.target_station_index})") # Reduced verbosity
                self.stop_moving() # Ensure stopped
                if self.current_sequence_index < 0 or self.current_sequence_index >= len(self.station_sequence): self.get_logger().error(f"ARRIVED: Invalid index {self.current_sequence_index}."); self.state = RobotState.ERROR; return
                current_station_idx = self.station_sequence[self.current_sequence_index]
                if current_station_idx not in STATION_INDEX_TO_FIELD: self.get_logger().error(f"ARRIVED: No field for index {current_station_idx}."); self.state = RobotState.ERROR; return
                station_field = STATION_INDEX_TO_FIELD[current_station_idx]
                # self.get_logger().info(f"At {station_field}. Updating Airtable to ARRIVED.") # Reduced verbosity
                if self.update_station_status(self.current_order["record_id"], station_field, STATUS_ARRIVED):
                    self.wait_start_time = time.time(); self.state = RobotState.WAITING_FOR_STATION_COMPLETION
                    self.get_logger().info(f"Status updated. Waiting for {station_field} to be DONE.")
                # else: Error state already set by update_station_status

            elif self.state == RobotState.WAITING_FOR_STATION_COMPLETION:
                elapsed_wait_time = time.time() - self.wait_start_time
                if elapsed_wait_time > STATION_WAIT_TIMEOUT_SEC:
                    self.get_logger().warning(f"WAITING TIMEOUT ({elapsed_wait_time:.1f}s) for station {self.target_station_index}."); self.play_sound([(330, 500), (220, 500)]); self.state = RobotState.STATION_TIMED_OUT; return

                if time.time() - self._last_airtable_check_time >= AIRTABLE_POLL_RATE:
                    self._last_airtable_check_time = time.time()
                    if self.current_sequence_index < 0 or self.current_sequence_index >= len(self.station_sequence): self.get_logger().error(f"WAITING: Invalid index."); self.state = RobotState.ERROR; return
                    current_station_idx = self.station_sequence[self.current_sequence_index]
                    if current_station_idx not in STATION_INDEX_TO_FIELD: self.get_logger().error(f"WAITING: No field for index {current_station_idx}."); self.state = RobotState.ERROR; return
                    station_field = STATION_INDEX_TO_FIELD[current_station_idx]

                    if self.wait_for_station_completion(self.current_order["record_id"], station_field):
                        self.get_logger().info(f"Station {current_station_idx} ({station_field}) reported DONE.")
                        self.play_sound([(659, 150), (784, 200)]) # Station complete sound

                        last_station_index = self.current_sequence_index
                        self.current_sequence_index += 1
                        if self.current_sequence_index >= len(self.station_sequence):
                            self.get_logger().info("Completed all stations."); self.state = RobotState.ORDER_COMPLETE
                        else:
                            self.target_station_index = self.station_sequence[self.current_sequence_index]
                            self.get_logger().info(f"Station {last_station_index} finished. Leaving station before moving to {self.target_station_index}.")
                            self.leaving_station_start_time = time.time(); self.state = RobotState.LEAVING_STATION
                            self.last_color_detection_times[last_station_index] = 0.0 # Reset cooldown

            elif self.state == RobotState.ORDER_COMPLETE:
                self.get_logger().info(f"Order '{self.current_order['order_name']}' COMPLETE.")
                self.play_sound([(784, 150), (880, 150), (1047, 250)]) # Order success sound
                self.stop_moving(); self.current_order = None; self.station_sequence = []; self.current_sequence_index = -1; self.target_station_index = -1
                self.state = RobotState.IDLE # Look for next order

            elif self.state == RobotState.ALL_ORDERS_COMPLETE:
                self.get_logger().info("No pending orders. Waiting...")
                self.play_sound([(440, 200), (440, 200)]) # Idle sound
                self.stop_moving(); time.sleep(5.0); self.state = RobotState.IDLE

            elif self.state == RobotState.STATION_TIMED_OUT:
                self.get_logger().error("STATION TIMED OUT.")
                self.stop_moving()
                if self.current_order: self.get_logger().error(f"Aborting order '{self.current_order['order_name']}'.")
                self.current_order = None; self.station_sequence = []; self.current_sequence_index = -1; self.target_station_index = -1
                self.state = RobotState.IDLE # Go back to fetch next order

            elif self.state == RobotState.AIRTABLE_ERROR:
                 self.get_logger().error("AIRTABLE COMMUNICATION ERROR. Halting.")
                 self.play_sound([(330, 300), (330, 300), (330, 300)]) # Error sound
                 self.stop_moving() # Remain in this state

        except Exception as e:
            self.get_logger().error(f"Unhandled exception in state machine: {e}", exc_info=True)
            self.state = RobotState.ERROR; self.stop_moving()
        # --- End State Machine ---

        # --- Display Update ---
        finally:
            if self.debug_windows:
                try:
                    if display_frame is not None: cv2.imshow("Camera Feed", display_frame)
                    if mask_frame is not None: cv2.imshow("Color Detection Mask", mask_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.get_logger().info("Quit key 'q' pressed. Initiating shutdown.")
                        self.state = RobotState.ERROR; self.stop_moving(); rclpy.try_shutdown()
                except Exception as display_e: self.get_logger().error(f"Error updating OpenCV windows: {display_e}")
        # --- End Display Update ---

# --- Main Function ---
def main(args=None):
    # (Main function remains unchanged)
    rclpy.init(args=args)
    node = None; executor = None
    try:
        node = PancakeRobotNode()
        if node.state not in [RobotState.GPIO_ERROR, RobotState.CAMERA_ERROR, RobotState.ERROR]:
            executor = SingleThreadedExecutor(); executor.add_node(node)
            node.get_logger().info("Starting ROS2 executor spin...")
            executor.spin()
        else: node.get_logger().fatal(f"Node initialization failed: {node.state.name}. Shutting down.")
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
            node.cleanup_hardware()
            if node.debug_windows: cv2.destroyAllWindows()
            node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
        print("Shutdown complete.")

if __name__ == '__main__':
    main()