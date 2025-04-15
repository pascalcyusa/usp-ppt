#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

import os
from dotenv import load_dotenv
import time
from enum import Enum, auto
import math
import cv2  # OpenCV for image processing
import numpy as np
from picamera2 import Picamera2  # Pi Camera library
from libcamera import controls  # For camera controls like autofocus
import requests  # For Airtable API calls
import json     # For Airtable API calls
import RPi.GPIO as GPIO  # For IR Sensors
from geometry_msgs.msg import Twist  # For direct velocity control

# iRobot Create 3 specific messages (Keep Actions for potential fine-tuning/docking later)
from irobot_create_msgs.action import DriveDistance, RotateAngle
# from irobot_create_msgs.msg import InterfaceButtons, IrIntensityVector # Example sensor msgs
from builtin_interfaces.msg import Duration
from irobot_create_msgs.msg import AudioNoteVector, AudioNote

# --- Configuration Constants ---

# Load environment variables from .env file
load_dotenv()

# --- Airtable Configuration ---
AIRTABLE_API_TOKEN = os.getenv('AIRTABLE_API_TOKEN')
AIRTABLE_BASE_ID = os.getenv('AIRTABLE_BASE_ID')
AIRTABLE_TABLE_NAME = os.getenv('AIRTABLE_TABLE_NAME')

if not all([AIRTABLE_API_TOKEN, AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME]):
    raise EnvironmentError("Missing required Airtable environment variables.")

# --- Construct Airtable URL and Headers ---
AIRTABLE_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
AIRTABLE_HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_API_TOKEN}",
    "Content-Type": "application/json",
}

# --- Field names in your Airtable base ---
AIRTABLE_ORDER_NAME_COLUMN = "Order Name"
AIRTABLE_CREATED_TIME_FIELD = "Created"
AIRTABLE_COOKING_1_STATUS_FIELD = "Cooking 1 Status"
AIRTABLE_COOKING_2_STATUS_FIELD = "Cooking 2 Status" # Renamed for clarity
AIRTABLE_WHIPPED_CREAM_STATUS_FIELD = "Whipped Cream Status"
AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD = "Choco Chips Status"
AIRTABLE_SPRINKLES_STATUS_FIELD = "Sprinkles Status"
AIRTABLE_PICKUP_STATUS_FIELD = "Pickup Status"

# --- Airtable Status Codes (Numeric) ---
STATUS_WAITING = 0
STATUS_ARRIVED = 1
STATUS_DONE = 99

# --- Map Airtable Fields to Logical Station Indices ---
STATION_FIELD_TO_INDEX = {
    AIRTABLE_COOKING_1_STATUS_FIELD: 1,
    AIRTABLE_COOKING_2_STATUS_FIELD: 2,  # Logical index for second cook
    AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD: 3,
    AIRTABLE_WHIPPED_CREAM_STATUS_FIELD: 4,
    AIRTABLE_SPRINKLES_STATUS_FIELD: 5,
    AIRTABLE_PICKUP_STATUS_FIELD: 0
}
STATION_INDEX_TO_FIELD = {v: k for k, v in STATION_FIELD_TO_INDEX.items()}

# --- Physical Station Index for Cooking ---
PHYSICAL_COOKING_STATION_INDEX = 1

# --- GPIO Configuration ---
LEFT_IR_PIN = 16
RIGHT_IR_PIN = 18

# --- Camera Configuration ---
CAMERA_RESOLUTION = (640, 480)
CAMERA_ROTATION = cv2.ROTATE_180

# --- Color Detection Configuration ---
STATION_COLORS_HSV = {
    0: {"name": "Pickup Station", "hsv_lower": (35, 100, 100), "hsv_upper": (85, 255, 255), "color_bgr": (255, 0, 0)},
    PHYSICAL_COOKING_STATION_INDEX: {"name": "Cooking Station", "hsv_lower": (35, 100, 100), "hsv_upper": (85, 255, 255), "color_bgr": (0, 255, 0)},
    3: {"name": "Chocolate Chips", "hsv_lower": (35, 100, 100), "hsv_upper": (85, 255, 255), "color_bgr": (0, 0, 255)},
    4: {"name": "Whipped Cream", "hsv_lower": (35, 100, 100), "hsv_upper": (85, 255, 255), "color_bgr": (255, 255, 0)},
    5: {"name": "Sprinkles", "hsv_lower": (35, 100, 100), "hsv_upper": (85, 255, 255), "color_bgr": (255, 0, 255)},
}
NUM_STATIONS_PHYSICAL = len(STATION_COLORS_HSV)

# --- Navigation & Control Parameters ---
IR_POLL_RATE = 0.001
COLOR_POLL_RATE = 0.1
AIRTABLE_POLL_RATE = 2.0
BASE_DRIVE_SPEED = 0.01
BASE_ROTATE_SPEED = 0.2
TURN_FACTOR = 0.7
LOST_LINE_ROTATE_SPEED = 0.1
COLOR_DETECTION_THRESHOLD = 2000
COLOR_COOLDOWN_SEC = 5.0
STATION_WAIT_TIMEOUT_SEC = 120.0

# --- State Machine Definition ---
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

# --- Main Robot Control Class ---
class PancakeRobotNode(Node):
    def __init__(self):
        super().__init__('pancake_robot_node')
        self.get_logger().info("Pancake Robot Node Initializing...")

        # Basic Airtable Config Check (Optional but recommended)
        if "YOUR_" in AIRTABLE_API_TOKEN or "YOUR_" in AIRTABLE_BASE_ID:
            self.get_logger().warn("!!! POSSIBLE AIRTABLE CONFIG ISSUE: Default token/base ID detected. !!!")

        # Robot State Initialization
        self.state = RobotState.IDLE
        self.current_order = None # {'record_id': ..., 'order_name': ..., 'station_status': {...}}
        self.station_sequence = [] # Logical indices
        self.current_sequence_index = 0
        self.target_station_index = -1 # Logical index
        self.pancakes_made_count = 0
        self.last_color_detection_times = { idx: 0.0 for idx in STATION_COLORS_HSV.keys() }
        self.wait_start_time = 0.0

        # --- Hardware Setup (Identical to test4, error handling sets state) ---
        try:
            GPIO.setmode(GPIO.BOARD); GPIO.setup(LEFT_IR_PIN, GPIO.IN); GPIO.setup(RIGHT_IR_PIN, GPIO.IN)
            self.get_logger().info(f"GPIO initialized.")
        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to initialize GPIO: {e}"); self.state = RobotState.GPIO_ERROR; return
        try:
            self.picam2 = Picamera2(); config = self.picam2.create_preview_configuration(main={"size": CAMERA_RESOLUTION})
            self.picam2.configure(config); self.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous, "LensPosition": 0.0})
            self.picam2.start(); time.sleep(2); self.get_logger().info("Pi Camera initialized."); self.debug_windows = True
        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to initialize Pi Camera: {e}"); self.cleanup_gpio(); self.state = RobotState.CAMERA_ERROR; return

        # --- ROS2 Setup (Identical to test4) ---
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.audio_publisher = self.create_publisher(AudioNoteVector, '/cmd_audio', 10)
        self.drive_client = ActionClient(self, DriveDistance, '/drive_distance')
        self.rotate_client = ActionClient(self, RotateAngle, '/rotate_angle')
        self.control_timer_period = 0.05
        self.control_timer = self.create_timer(self.control_timer_period, self.control_loop)
        self.airtable_poll_timer = None

        self.get_logger().info("Pancake Robot Node Initialized and Ready.")
        self.play_sound([(440, 200), (550, 300)])

    # --- Movement Control (Identical to test4) ---
    def move_robot(self, linear_x, angular_z):
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.GPIO_ERROR, RobotState.AIRTABLE_ERROR]:
            twist_msg = Twist(); twist_msg.linear.x = 0.0; twist_msg.angular.z = 0.0
            self.cmd_vel_pub.publish(twist_msg); return
        twist_msg = Twist(); twist_msg.linear.x = float(linear_x); twist_msg.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(twist_msg)

    def stop_moving(self):
        self.get_logger().info("Sending stop command (Twist zero)."); self.move_robot(0.0, 0.0); time.sleep(0.1)

    # --- Sensor Reading (Identical to test4) ---
    def read_ir_sensors(self):
        try: return GPIO.input(LEFT_IR_PIN), GPIO.input(RIGHT_IR_PIN)
        except Exception as e: self.get_logger().error(f"Error reading GPIO: {e}"); return GPIO.HIGH, GPIO.HIGH

    # --- Color Detection (Identical to test4 - checks physical_target_idx) ---
    def check_for_station_color(self, frame, physical_target_idx):
        if physical_target_idx not in STATION_COLORS_HSV:
            self.get_logger().warn(f"Invalid *physical* target index {physical_target_idx} for color detection.")
            return False, None
        color_info = STATION_COLORS_HSV[physical_target_idx]
        lower = np.array(color_info["hsv_lower"]); upper = np.array(color_info["hsv_upper"])
        name = color_info["name"]; bgr = color_info.get("color_bgr", (255, 255, 255))
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
            color_mask = cv2.inRange(hsv, lower, upper)
            color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(white_mask))
            pixels = cv2.countNonZero(color_mask)
            debug_frame = None
            if self.debug_windows:
                debug_frame = frame.copy()
                detected_area = cv2.bitwise_and(frame, frame, mask=color_mask)
                debug_frame = cv2.addWeighted(debug_frame, 1, detected_area, 0.5, 0)
                text = f"Detect: {name} ({physical_target_idx}) {pixels}px"; cv2.putText(debug_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)
                logic_text = f"Logical Target: {self.target_station_index}"; cv2.putText(debug_frame, logic_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            now = time.time()
            if pixels > COLOR_DETECTION_THRESHOLD and (now - self.last_color_detection_times.get(physical_target_idx, 0.0) > COLOR_COOLDOWN_SEC):
                self.get_logger().info(f"Detected: {name} (Phys {physical_target_idx}, Logical {self.target_station_index})!")
                self.last_color_detection_times[physical_target_idx] = now
                return True, debug_frame
            return False, debug_frame
        except cv2.error as e: self.get_logger().error(f"OpenCV err: {e}"); return False, None
        except Exception as e: self.get_logger().error(f"Color detect err: {e}"); return False, None


    # --- Airtable Communication (Direct Requests) ---
    def fetch_order_from_airtable(self):
        self.get_logger().info("Attempting to fetch next order from Airtable...")
        params = {
            "maxRecords": 1,
            "filterByFormula": f"AND(OR({{{AIRTABLE_COOKING_1_STATUS_FIELD}}}=0, {{{AIRTABLE_COOKING_2_STATUS_FIELD}}}=0), {{{AIRTABLE_PICKUP_STATUS_FIELD}}}=0)",
            "sort[0][field]": AIRTABLE_CREATED_TIME_FIELD, "sort[0][direction]": "asc",
            "fields[]": [AIRTABLE_ORDER_NAME_COLUMN, AIRTABLE_CREATED_TIME_FIELD] + list(STATION_FIELD_TO_INDEX.keys())
        }
        self.get_logger().debug(f"Airtable fetch params: {params}")
        try:
            response = requests.get(AIRTABLE_URL, headers=AIRTABLE_HEADERS, params=params, timeout=15)
            response.raise_for_status(); data = response.json()
            self.get_logger().debug(f"Airtable fetch response: {json.dumps(data, indent=2)}")
            records = data.get("records", [])
            if records:
                record = records[0]; record_id = record.get("id"); fields = record.get("fields", {})
                order_name = fields.get(AIRTABLE_ORDER_NAME_COLUMN); created = fields.get(AIRTABLE_CREATED_TIME_FIELD)
                if not record_id or not order_name:
                    self.get_logger().error(f"Fetched record missing ID/Name: {record}"); return None
                fetched_order = {
                    "record_id": record_id, "order_name": order_name, "created_time": created,
                    "station_status": { f: fields.get(f, 0) for f in STATION_FIELD_TO_INDEX.keys() }
                }
                self.get_logger().info(f"Fetched: '{order_name}' (ID: {record_id})")
                self.get_logger().debug(f"Details: {fetched_order}")
                return fetched_order
            else:
                self.get_logger().info("No suitable pending orders found."); return None
        except requests.exceptions.RequestException as e: self.log_airtable_error("fetch", e); return None
        except Exception as e: self.get_logger().error(f"Airtable fetch processing err: {e}"); return None

    def update_station_status_in_airtable(self, record_id, station_field_name, new_status_code):
        if not record_id or not station_field_name:
            self.get_logger().error("Airtable update err: missing record_id/field_name."); return False
        self.get_logger().info(f"Updating Airtable: Rec {record_id}, Field '{station_field_name}' to {new_status_code}")
        update_url = f"{AIRTABLE_URL}/{record_id}"
        payload = json.dumps({"fields": {station_field_name: new_status_code}})
        try:
            response = requests.patch(update_url, headers=AIRTABLE_HEADERS, data=payload, timeout=10)
            response.raise_for_status()
            self.get_logger().info(f"Airtable update success for {station_field_name}.")
            # Update local cache if needed
            if self.current_order and self.current_order["record_id"] == record_id:
                if station_field_name in self.current_order["station_status"]:
                    self.current_order["station_status"][station_field_name] = new_status_code
                    self.get_logger().debug(f"Local cache updated for {station_field_name}.")
            return True
        except requests.exceptions.RequestException as e: self.log_airtable_error(f"update field {station_field_name}", e); return False
        except Exception as e: self.get_logger().error(f"Airtable update unexpected err: {e}"); return False

    def check_station_status_in_airtable(self, record_id, station_field_name):
        if not record_id or not station_field_name:
            self.get_logger().error("Airtable check err: missing record_id/field_name."); return None
        # self.get_logger().debug(f"Checking Airtable: Rec {record_id}, Field {station_field_name}") # Too verbose for polling
        check_url = f"{AIRTABLE_URL}/{record_id}"
        params = {"fields[]": station_field_name} # Request only the specific field
        try:
            response = requests.get(check_url, headers=AIRTABLE_HEADERS, params=params, timeout=10)
            response.raise_for_status(); data = response.json()
            status = data.get("fields", {}).get(station_field_name)
            # self.get_logger().debug(f"Status for {station_field_name}: {status}") # Too verbose
            return status # Returns None if field not found or other issues
        except requests.exceptions.RequestException as e: self.log_airtable_error(f"check field {station_field_name}", e); return None
        except Exception as e: self.get_logger().error(f"Airtable check unexpected err: {e}"); return None

    def log_airtable_error(self, action, req_exception):
        self.get_logger().error(f"Airtable {action} error: {req_exception}")
        if hasattr(req_exception, 'response') and req_exception.response is not None:
            self.get_logger().error(f"Status Code: {req_exception.response.status_code}")
            try: self.get_logger().error(f"Response: {json.dumps(req_exception.response.json())}")
            except json.JSONDecodeError: self.get_logger().error(f"Response Text: {req_exception.response.text}")
        self.state = RobotState.AIRTABLE_ERROR; self.stop_moving()

    # --- Sound Utility (Identical to test4) ---
    def play_sound(self, notes):
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.GPIO_ERROR, RobotState.AIRTABLE_ERROR]: return
        audio_msg = AudioNoteVector(); audio_msg.append = False
        for freq, ms in notes:
            note = AudioNote(); note.frequency = int(freq); note.max_runtime = Duration(sec=0, nanosec=int(ms * 1e6))
            audio_msg.notes.append(note)
        self.get_logger().debug(f"Playing sound: {notes}"); self.audio_publisher.publish(audio_msg)

    # --- Airtable Polling (Identical logic to test4, uses direct request wrappers) ---
    def start_airtable_polling(self):
        if self.state == RobotState.AIRTABLE_ERROR: self.get_logger().error("Cannot poll, in AIRTABLE_ERROR state."); return
        target_field = STATION_INDEX_TO_FIELD.get(self.target_station_index)
        if not target_field: self.get_logger().error(f"Polling Err: No field for logical index {self.target_station_index}."); self.state = RobotState.ERROR; return
        self.get_logger().info(f"Starting Airtable polling for logical station {self.target_station_index} (Field: {target_field})")
        self.stop_airtable_polling()
        self.airtable_poll_timer = self.create_timer(AIRTABLE_POLL_RATE, self.airtable_poll_callback)
        self.wait_start_time = time.time()

    def stop_airtable_polling(self):
        if self.airtable_poll_timer:
            if not self.airtable_poll_timer.is_canceled(): self.airtable_poll_timer.cancel()
            self.airtable_poll_timer = None; self.get_logger().info("Airtable polling stopped.")

    def airtable_poll_callback(self):
        if self.state != RobotState.WAITING_FOR_STATION_COMPLETION:
            self.get_logger().warn("Airtable poll callback unexpected. Stopping."); self.stop_airtable_polling(); return
        if not self.current_order or "record_id" not in self.current_order:
             self.get_logger().error("Polling Err: No current order/record_id."); self.state = RobotState.ERROR; self.stop_airtable_polling(); self.stop_moving(); return
        record_id = self.current_order["record_id"]
        target_field = STATION_INDEX_TO_FIELD.get(self.target_station_index)
        if not target_field:
            self.get_logger().error(f"Polling Err: Invalid target idx {self.target_station_index}."); self.state = RobotState.ERROR; self.stop_airtable_polling(); self.stop_moving(); return

        self.get_logger().debug(f"Polling Airtable: Rec {record_id}, Logical St {self.target_station_index} ({target_field})")
        current_status = self.check_station_status_in_airtable(record_id, target_field) # Use wrapper

        if current_status is None:
            # Error logged by wrapper, state should be set to AIRTABLE_ERROR
             self.get_logger().error(f"Airtable check failed during poll for {target_field}. Stopping poll."); self.stop_airtable_polling(); self.stop_moving(); return

        self.get_logger().debug(f"Current status for {target_field}: {current_status}")
        if current_status == STATUS_DONE: # 99
            self.get_logger().info(f"Logical Station {self.target_station_index} ({target_field}) reported DONE (99)!")
            self.play_sound([(600, 100), (800, 150)]); self.stop_airtable_polling()
            # Let main loop transition
        elif (time.time() - self.wait_start_time) > STATION_WAIT_TIMEOUT_SEC:
             self.get_logger().error(f"TIMEOUT waiting for logical station {self.target_station_index} ({target_field}) to be DONE.")
             self.state = RobotState.STATION_TIMED_OUT; self.stop_airtable_polling(); self.stop_moving()


    # --- Main Control Loop (State Machine Logic - Identical to test4) ---
    def control_loop(self):
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.GPIO_ERROR, RobotState.AIRTABLE_ERROR, RobotState.ALL_ORDERS_COMPLETE, RobotState.STATION_TIMED_OUT]:
            if self.state != RobotState.ALL_ORDERS_COMPLETE: self.get_logger().error(f"Robot in terminal/error state: {self.state.name}. Halting.", throttle_duration_sec=5)
            self.stop_moving(); self.stop_airtable_polling(); return

        left_ir, right_ir = self.read_ir_sensors()
        frame = self.picam2.capture_array()
        if CAMERA_ROTATION is not None: frame = cv2.rotate(frame, CAMERA_ROTATION)

        current_state = self.state; next_state = current_state

        # --- State Implementations (Functionally same as test4, using direct request wrappers) ---
        if current_state == RobotState.IDLE:
            self.get_logger().info("State: IDLE. Checking orders."); self.current_order = None; self.station_sequence = []; self.current_sequence_index = 0; self.target_station_index = -1
            next_state = RobotState.FETCHING_ORDER

        elif current_state == RobotState.FETCHING_ORDER:
            fetched_order = self.fetch_order_from_airtable() # Uses direct requests
            if fetched_order: self.current_order = fetched_order; self.get_logger().info(f"Processing Order: {self.current_order['order_name']}"); next_state = RobotState.PLANNING_ROUTE
            elif self.state == RobotState.AIRTABLE_ERROR: pass # Error handled by fetch
            else: self.get_logger().info("No more pending orders."); self.play_sound([(600,100),(700,100),(800,300)] if self.pancakes_made_count>0 else [(400,500)]); next_state = RobotState.ALL_ORDERS_COMPLETE

        elif current_state == RobotState.PLANNING_ROUTE:
            if not self.current_order: self.get_logger().error("Planning err: No order data."); next_state = RobotState.ERROR
            else:
                self.get_logger().info("Planning route..."); self.station_sequence = []
                order_preference = [1, 2, 3, 4, 5, 0] # Logical indices
                for idx in order_preference:
                    field = STATION_INDEX_TO_FIELD.get(idx)
                    if field and self.current_order["station_status"].get(field, -1) == STATUS_WAITING:
                        phys_idx = PHYSICAL_COOKING_STATION_INDEX if idx == 2 else idx
                        if phys_idx in STATION_COLORS_HSV or idx == 2:
                            self.station_sequence.append(idx)
                            name = STATION_COLORS_HSV.get(phys_idx, {}).get('name', f'Logical {idx}')
                            self.get_logger().info(f" - Add Step {idx} ({name}, Field: {field})")
                        else: self.get_logger().warn(f"Skip {field}: Phys Idx {phys_idx} no color")
                self.get_logger().info(f"Planned route (logical): {self.station_sequence}")
                self.current_sequence_index = 0
                if not self.station_sequence: self.get_logger().error("Planning err: No stations."); next_state = RobotState.ORDER_COMPLETE # Or ERROR
                else: self.target_station_index = self.station_sequence[0]; next_state = RobotState.RETURNING_TO_PICKUP if self.target_station_index == 0 else RobotState.MOVING_TO_STATION

        elif current_state == RobotState.MOVING_TO_STATION or current_state == RobotState.RETURNING_TO_PICKUP:
            phys_idx = -1
            if self.target_station_index == 0: phys_idx = 0
            elif self.target_station_index == 1: phys_idx = PHYSICAL_COOKING_STATION_INDEX
            elif self.target_station_index == 2: phys_idx = PHYSICAL_COOKING_STATION_INDEX
            elif self.target_station_index in STATION_COLORS_HSV: phys_idx = self.target_station_index
            else: self.get_logger().error(f"Move err: No physical station for logical {self.target_station_index}"); next_state = RobotState.ERROR; self.stop_moving()
            if next_state != RobotState.ERROR:
                detected, dbg_frame = self.check_for_station_color(frame, phys_idx)
                if detected:
                    name = STATION_COLORS_HSV[phys_idx]['name']; self.get_logger().info(f"Marker {name} detected (Logical {self.target_station_index})"); self.play_sound([(500, 150)]); self.stop_moving()
                    next_state = RobotState.ARRIVED_AT_STATION if current_state == RobotState.MOVING_TO_STATION else RobotState.ARRIVED_AT_PICKUP
                else: # Line following
                    lin = BASE_DRIVE_SPEED; ang = 0.0
                    if left_ir == GPIO.HIGH and right_ir == GPIO.HIGH: ang = 0.0
                    elif left_ir == GPIO.LOW and right_ir == GPIO.HIGH: ang = BASE_ROTATE_SPEED * TURN_FACTOR; lin *= 0.8
                    elif left_ir == GPIO.HIGH and right_ir == GPIO.LOW: ang = -BASE_ROTATE_SPEED * TURN_FACTOR; lin *= 0.8
                    elif left_ir == GPIO.LOW and right_ir == GPIO.LOW: self.get_logger().warn("IR: Both black.", throttle_duration_sec=2); lin = 0.0; ang = LOST_LINE_ROTATE_SPEED
                    self.move_robot(lin, ang)
                if self.debug_windows and dbg_frame is not None:
                    ir = f"IR L:{left_ir} R:{right_ir}"; cv2.putText(dbg_frame, ir, (frame.shape[1]-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2); cv2.imshow("Robot View", dbg_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): self.get_logger().info("Quit via window."); self.state = RobotState.ERROR; rclpy.shutdown()

        elif current_state == RobotState.ARRIVED_AT_STATION:
            if not self.current_order: self.get_logger().error("Arrival err: No order."); next_state = RobotState.ERROR
            else:
                field = STATION_INDEX_TO_FIELD.get(self.target_station_index)
                if not field: self.get_logger().error(f"Arrival err: No field for logical {self.target_station_index}"); next_state = RobotState.ERROR
                else:
                    phys_idx = PHYSICAL_COOKING_STATION_INDEX if self.target_station_index == 2 else self.target_station_index
                    name = STATION_COLORS_HSV.get(phys_idx, {}).get('name', f'Logical {self.target_station_index}')
                    self.get_logger().info(f"Arrived logical {self.target_station_index} ({name}, Field {field}). Update status {STATUS_ARRIVED}.")
                    success = self.update_station_status_in_airtable(self.current_order["record_id"], field, STATUS_ARRIVED) # Use wrapper
                    if success: self.start_airtable_polling(); next_state = RobotState.WAITING_FOR_STATION_COMPLETION
                    else: self.get_logger().error(f"Failed Airtable update on arrival for {field}."); next_state = RobotState.AIRTABLE_ERROR; self.stop_moving() # State set by wrapper?

        elif current_state == RobotState.WAITING_FOR_STATION_COMPLETION:
             if self.airtable_poll_timer is None: # Timer stopped
                 if self.state == RobotState.STATION_TIMED_OUT: pass # Error handled
                 elif self.state == RobotState.AIRTABLE_ERROR: pass # Error handled
                 elif self.state == RobotState.ERROR: pass # Error handled
                 else: # Assume DONE detected
                      field = STATION_INDEX_TO_FIELD.get(self.target_station_index, "??")
                      self.get_logger().info(f"Logical Station {self.target_station_index} ({field}) complete.")
                      self.current_sequence_index += 1
                      if self.current_sequence_index < len(self.station_sequence):
                          self.target_station_index = self.station_sequence[self.current_sequence_index]
                          next_field = STATION_INDEX_TO_FIELD.get(self.target_station_index, "??")
                          phys_idx = PHYSICAL_COOKING_STATION_INDEX if self.target_station_index == 2 else self.target_station_index
                          next_name = STATION_COLORS_HSV.get(phys_idx, {}).get('name', f'Logical {self.target_station_index}')
                          self.get_logger().info(f"Next logical step: {self.target_station_index} ({next_name}, Field: {next_field})")
                          next_state = RobotState.RETURNING_TO_PICKUP if self.target_station_index == 0 else RobotState.MOVING_TO_STATION
                      else: # Finished sequence
                          if self.target_station_index == 0: next_state = RobotState.ORDER_COMPLETE # Should have gone via ARRIVED_AT_PICKUP
                          else: self.get_logger().error("Sequence end err: Last step not Pickup."); next_state = RobotState.ERROR

        elif current_state == RobotState.ARRIVED_AT_PICKUP:
            if not self.current_order: self.get_logger().error("Pickup err: No order."); next_state = RobotState.ERROR
            else:
                self.get_logger().info(f"Arrived Pickup. Order '{self.current_order['order_name']}' sequence complete.")
                self.pancakes_made_count += 1; self.play_sound([(800, 100), (700, 100), (600, 200)])
                field = STATION_INDEX_TO_FIELD.get(0)
                if field:
                    success = self.update_station_status_in_airtable(self.current_order["record_id"], field, STATUS_DONE) # Use wrapper
                    if not success: self.get_logger().error(f"Failed final Pickup ({field}) update. Continuing...")
                else: self.get_logger().warn("No field for Pickup (0). Cannot update.")
                next_state = RobotState.ORDER_COMPLETE

        elif current_state == RobotState.ORDER_COMPLETE:
            self.get_logger().info(f"Order '{self.current_order.get('order_name', 'N/A')}' finished. -> IDLE.")
            self.current_order = None; self.station_sequence = []; self.current_sequence_index = 0; self.target_station_index = -1
            next_state = RobotState.IDLE

        # --- State Transition ---
        if next_state != current_state:
            self.get_logger().info(f"State transition: {current_state.name} -> {next_state.name}"); self.state = next_state


    # --- Cleanup Methods (Identical to test4) ---
    def cleanup_gpio(self):
        self.get_logger().info("Cleaning up GPIO...")
        try:
            if GPIO.getmode() is not None: GPIO.cleanup(); self.get_logger().info("GPIO cleanup successful.")
            else: self.get_logger().info("GPIO not set up.")
        except Exception as e: self.get_logger().error(f"GPIO cleanup err: {e}")

    def shutdown_camera(self):
        if hasattr(self, 'picam2') and self.picam2:
            try:
                if self.picam2.started: self.picam2.stop(); self.get_logger().info("Pi Camera stopped.")
            except Exception as e: self.get_logger().error(f"Camera stop err: {e}")
        if self.debug_windows:
            try: cv2.destroyAllWindows(); self.get_logger().info("OpenCV windows closed.")
            except Exception as e: self.get_logger().error(f"OpenCV window close err: {e}")

    def shutdown_robot(self):
        self.get_logger().info("Initiating shutdown..."); self.stop_moving()
        if hasattr(self, 'control_timer') and self.control_timer and not self.control_timer.is_canceled(): self.control_timer.cancel()
        self.stop_airtable_polling(); self.shutdown_camera(); self.cleanup_gpio(); self.get_logger().info("Shutdown complete.")


# --- Main Execution Function (Identical to test4) ---
def main(args=None):
    rclpy.init(args=args); pancake_robot_node = None; exit_code = 0
    try:
        pancake_robot_node = PancakeRobotNode()
        if pancake_robot_node.state not in [RobotState.GPIO_ERROR, RobotState.CAMERA_ERROR, RobotState.AIRTABLE_ERROR]:
            rclpy.spin(pancake_robot_node)
        else: pancake_robot_node.get_logger().fatal(f"Init failed: {pancake_robot_node.state.name}. Abort."); exit_code = 1
    except KeyboardInterrupt: print("\nKeyboard interrupt.")
    except Exception as e: print(f"\nUnexpected ROS err: {e}"); import traceback; traceback.print_exc(); exit_code = 1
    finally:
        if pancake_robot_node:
            pancake_robot_node.get_logger().info("ROS shutdown. Node cleanup..."); pancake_robot_node.shutdown_robot()
            pancake_robot_node.destroy_node(); pancake_robot_node.get_logger().info("Node destroyed.")
        else:
             print("Node DNE, attempt final GPIO cleanup...");
             try:
                 if GPIO.getmode() is not None: GPIO.cleanup(); print("GPIO cleanup attempted.")
             except Exception as e: print(f"Final GPIO cleanup err: {e}")
        if rclpy.ok(): rclpy.shutdown(); print("ROS2 shutdown complete.")
    return exit_code

if __name__ == '__main__':
    import sys; sys.exit(main())