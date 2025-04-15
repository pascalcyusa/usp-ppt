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
import requests  # Still needed for fetch_order_from_airtable
import json     # For Airtable API calls (fetch_order) and error logging
import RPi.GPIO as GPIO  # For IR Sensors
from geometry_msgs.msg import Twist  # For direct velocity control

# iRobot Create 3 specific messages (Keep Actions for potential fine-tuning/docking later)
from irobot_create_msgs.action import DriveDistance, RotateAngle
# from irobot_create_msgs.msg import InterfaceButtons, IrIntensityVector # Example sensor msgs
from builtin_interfaces.msg import Duration
from irobot_create_msgs.msg import AudioNoteVector, AudioNote

# --- Import the Airtable Handler Class ---
from AirtablePancake import at as AirtableHandler # Use 'as' for clarity


# --- Configuration Constants ---

# Load environment variables from .env file
load_dotenv()

# --- Airtable Configuration (Loaded from .env) ---
AIRTABLE_API_TOKEN = os.getenv('AIRTABLE_API_TOKEN')
AIRTABLE_BASE_ID = os.getenv('AIRTABLE_BASE_ID')
AIRTABLE_TABLE_NAME = os.getenv('AIRTABLE_TABLE_NAME')

if not all([AIRTABLE_API_TOKEN, AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME]):
    raise EnvironmentError(
        "Missing required Airtable environment variables (AIRTABLE_API_TOKEN, AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME). Please check your .env file.")

# --- Construct Airtable URL and Headers (ONLY for fetch_order) ---
# The AirtableHandler class manages its own URL and headers internally
AIRTABLE_FETCH_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
AIRTABLE_FETCH_HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_API_TOKEN}",
    "Content-Type": "application/json",
}

# --- Field names in your Airtable base (MUST match exactly, case-sensitive) ---
AIRTABLE_ORDER_NAME_COLUMN = "Order Name"       # Column for the order identifier
AIRTABLE_CREATED_TIME_FIELD = "Created"   # Column for order creation timestamp

# Station Status Fields (Numeric)
AIRTABLE_COOKING_1_STATUS_FIELD = "Cooking 1 Status"
AIRTABLE_COOKING_2_STATUS_FIELD = "Cooking 2 Status" # Renamed for clarity, but refers to the original field name in .env/Airtable
AIRTABLE_WHIPPED_CREAM_STATUS_FIELD = "Whipped Cream Status"
AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD = "Choco Chips Status"
AIRTABLE_SPRINKLES_STATUS_FIELD = "Sprinkles Status"
AIRTABLE_PICKUP_STATUS_FIELD = "Pickup Status"

# --- Airtable Status Codes (Numeric) ---
STATUS_WAITING = 0
STATUS_ARRIVED = 1
STATUS_DONE = 99

# --- Map Airtable Fields and Order Requirements to Station Indices ---
# Maps the Airtable *Status* Field Name to the *Logical* Station Index it represents
# Index 2 now represents the *second cooking step*, not a physical location.
STATION_FIELD_TO_INDEX = {
    AIRTABLE_COOKING_1_STATUS_FIELD: 1,
    AIRTABLE_COOKING_2_STATUS_FIELD: 2,  # Logical index for the second cooking step
    AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD: 3,
    AIRTABLE_WHIPPED_CREAM_STATUS_FIELD: 4,
    AIRTABLE_SPRINKLES_STATUS_FIELD: 5,
    AIRTABLE_PICKUP_STATUS_FIELD: 0
}
# Inverse map for convenience
STATION_INDEX_TO_FIELD = {v: k for k, v in STATION_FIELD_TO_INDEX.items()}

# --- Physical Station Index for the Cooking Area ---
PHYSICAL_COOKING_STATION_INDEX = 1

# --- GPIO Configuration ---
LEFT_IR_PIN = 16   # BOARD pin number for Left IR sensor
RIGHT_IR_PIN = 18  # BOARD pin number for Right IR sensor

# --- Camera Configuration ---
CAMERA_RESOLUTION = (640, 480)  # Width, Height
CAMERA_ROTATION = cv2.ROTATE_180  # Adjust if camera is mounted upside down

# --- Color Detection Configuration ---
# Define HSV Lower and Upper bounds for each *physical* station's target color marker
STATION_COLORS_HSV = {
    # Index 0: Color marker to detect when returning to the start/pickup station
    0: {"name": "Pickup Station", "hsv_lower": (35, 100, 100), "hsv_upper": (85, 255, 255), "color_bgr": (255, 0, 0)},

    # Index 1: Cooking Station (Used for both Cooking 1 and Cooking 2 logical steps)
    PHYSICAL_COOKING_STATION_INDEX: {"name": "Cooking Station", "hsv_lower": (35, 100, 100), "hsv_upper": (85, 255, 255), "color_bgr": (0, 255, 0)}, # Changed color for demo

    # Index 2: No physical marker - This logical step uses marker 1
    # 2: {"name": "UNUSED - Second Cook (Logical)", ... },

    # Index 3: Chocolate Chips Station
    3: {"name": "Chocolate Chips", "hsv_lower": (35, 100, 100), "hsv_upper": (85, 255, 255), "color_bgr": (0, 0, 255)}, # Changed color for demo

    # Index 4: Whipped Cream Station
    4: {"name": "Whipped Cream", "hsv_lower": (35, 100, 100), "hsv_upper": (85, 255, 255), "color_bgr": (255, 255, 0)}, # Changed color for demo

    # Index 5: Sprinkles Station
    5: {"name": "Sprinkles", "hsv_lower": (35, 100, 100), "hsv_upper": (85, 255, 255), "color_bgr": (255, 0, 255)}, # Changed color for demo
}
# Total physical stations with markers
NUM_STATIONS_PHYSICAL = len(STATION_COLORS_HSV) # Now reflects only stations with physical markers

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
    RETURNING_TO_PICKUP = auto() # Specifically for moving to physical station 0
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

        # --- Instantiate Airtable Handler ---
        try:
            self.airtable_handler = AirtableHandler(
                api_key=AIRTABLE_API_TOKEN,
                base_id=AIRTABLE_BASE_ID,
                table_name=AIRTABLE_TABLE_NAME
            )
            self.get_logger().info("Airtable Handler initialized.")
        except ValueError as e:
             self.get_logger().error(f"FATAL: Failed to initialize Airtable Handler: {e}")
             self.state = RobotState.AIRTABLE_ERROR
             return # Stop initialization
        except Exception as e:
             self.get_logger().error(f"FATAL: Unexpected error initializing Airtable Handler: {e}")
             self.state = RobotState.AIRTABLE_ERROR
             return # Stop initialization

        # Robot State Initialization
        self.state = RobotState.IDLE
        self.current_order = None # Stores fetched order details
        self.station_sequence = [] # Ordered list of *logical* station indices to visit [1, 2, 3, 0]
        self.current_sequence_index = 0
        self.target_station_index = -1 # The current *logical* target station index from the sequence
        self.pancakes_made_count = 0
        self.last_color_detection_times = { idx: 0.0 for idx in STATION_COLORS_HSV.keys() } # Cooldown per *physical* marker
        self.wait_start_time = 0.0

        # --- Hardware Setup ---
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

        # ROS2 Publishers & Clients
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.audio_publisher = self.create_publisher(AudioNoteVector, '/cmd_audio', 10)
        self.drive_client = ActionClient(self, DriveDistance, '/drive_distance')
        self.rotate_client = ActionClient(self, RotateAngle, '/rotate_angle')

        # Timers
        self.control_timer_period = 0.05 # 20 Hz
        self.control_timer = self.create_timer(self.control_timer_period, self.control_loop)
        self.airtable_poll_timer = None

        self.get_logger().info("Pancake Robot Node Initialized and Ready.")
        self.play_sound([(440, 200), (550, 300)])

    # --- Movement Control ---
    def move_robot(self, linear_x, angular_z):
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.GPIO_ERROR, RobotState.AIRTABLE_ERROR]:
            twist_msg = Twist()
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            self.cmd_vel_pub.publish(twist_msg)
            return
        twist_msg = Twist()
        twist_msg.linear.x = float(linear_x)
        twist_msg.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(twist_msg)

    def stop_moving(self):
        self.get_logger().info("Sending stop command (Twist zero).")
        self.move_robot(0.0, 0.0)
        time.sleep(0.1)

    # --- Sensor Reading ---
    def read_ir_sensors(self):
        try:
            left_val = GPIO.input(LEFT_IR_PIN)
            right_val = GPIO.input(RIGHT_IR_PIN)
            return left_val, right_val
        except Exception as e:
            self.get_logger().error(f"Error reading GPIO IR sensors: {e}")
            return GPIO.HIGH, GPIO.HIGH

    # --- Color Detection ---
    def check_for_station_color(self, frame, physical_target_idx):
        """Analyzes a frame for the target *physical* station's color marker."""
        if physical_target_idx not in STATION_COLORS_HSV:
            # This might happen temporarily if target is 2, but physical should be 1. Log warning if physical is invalid.
            self.get_logger().warn(f"Invalid *physical* target index {physical_target_idx} for color detection.")
            return False, None

        color_info = STATION_COLORS_HSV[physical_target_idx]
        lower_bound = np.array(color_info["hsv_lower"])
        upper_bound = np.array(color_info["hsv_upper"])
        target_color_name = color_info["name"]
        color_bgr = color_info.get("color_bgr", (255, 255, 255))

        try:
            hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            white_lower = np.array([0, 0, 200])
            white_upper = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv_image, white_lower, white_upper)
            color_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
            color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(white_mask))
            detected_pixels = cv2.countNonZero(color_mask)

            debug_frame = None
            if self.debug_windows:
                debug_frame = frame.copy()
                detected_area = cv2.bitwise_and(frame, frame, mask=color_mask)
                debug_frame = cv2.addWeighted(debug_frame, 1, detected_area, 0.5, 0)
                text = f"Detecting: {target_color_name} ({physical_target_idx}) - {detected_pixels} px"
                cv2.putText(debug_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
                # Also show the logical target
                logic_text = f"Logical Target: {self.target_station_index}"
                cv2.putText(debug_frame, logic_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)


            current_time = time.time()
            if detected_pixels > COLOR_DETECTION_THRESHOLD and \
               (current_time - self.last_color_detection_times.get(physical_target_idx, 0.0) > COLOR_COOLDOWN_SEC):
                self.get_logger().info(f"Detected physical marker for {target_color_name} (Phys Idx {physical_target_idx}, Logical Target {self.target_station_index})!")
                self.last_color_detection_times[physical_target_idx] = current_time
                return True, debug_frame

            return False, debug_frame

        except cv2.error as cv_err:
            self.get_logger().error(f"OpenCV error during color detection: {cv_err}")
            return False, None
        except Exception as e:
            self.get_logger().error(f"Unexpected error during color detection: {e}")
            return False, None

    # --- Airtable Communication (Fetch Only - Update/Check use Handler) ---
    def fetch_order_from_airtable(self):
        """Fetches the oldest order that needs processing (using direct requests)."""
        self.get_logger().info("Attempting to fetch next order from Airtable...")
        # Fetch orders where Cook 1 OR Cook 2 is 0, AND Pickup is 0
        # Order by creation time
        params = {
            "maxRecords": 1,
             "filterByFormula": f"AND(OR({{{AIRTABLE_COOKING_1_STATUS_FIELD}}}=0, {{{AIRTABLE_COOKING_2_STATUS_FIELD}}}=0), {{{AIRTABLE_PICKUP_STATUS_FIELD}}}=0)",
            "sort[0][field]": AIRTABLE_CREATED_TIME_FIELD,
            "sort[0][direction]": "asc",
            "fields[]": [
                AIRTABLE_ORDER_NAME_COLUMN,
                AIRTABLE_CREATED_TIME_FIELD,
                AIRTABLE_COOKING_1_STATUS_FIELD,
                AIRTABLE_COOKING_2_STATUS_FIELD, # Ensure this is fetched
                AIRTABLE_WHIPPED_CREAM_STATUS_FIELD,
                AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD,
                AIRTABLE_SPRINKLES_STATUS_FIELD,
                AIRTABLE_PICKUP_STATUS_FIELD
            ]
        }
        self.get_logger().debug(f"Airtable fetch query params: {params}")

        try:
            response = requests.get(url=AIRTABLE_FETCH_URL, headers=AIRTABLE_FETCH_HEADERS, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            self.get_logger().debug(f"Airtable fetch response: {json.dumps(data, indent=2)}")

            records = data.get("records", [])
            if records:
                record = records[0]
                record_id = record.get("id")
                fields = record.get("fields", {})
                order_name = fields.get(AIRTABLE_ORDER_NAME_COLUMN)
                created_time = fields.get(AIRTABLE_CREATED_TIME_FIELD)

                if not record_id or not order_name:
                    self.get_logger().error(f"Fetched record missing ID or Name: {record}")
                    return None

                fetched_order = {
                    "record_id": record_id,
                    "order_name": order_name,
                    "created_time": created_time,
                    "station_status": {
                        f: fields.get(f, 0) for f in STATION_FIELD_TO_INDEX.keys() if f # Populate with actual statuses
                    }
                }
                self.get_logger().info(f"Fetched order: '{order_name}' (Created: {created_time}, Record ID: {record_id})")
                self.get_logger().debug(f"Order details: {fetched_order}")
                return fetched_order
            else:
                self.get_logger().info("No suitable pending orders found in Airtable.")
                return None

        except requests.exceptions.RequestException as req_err:
            self.log_airtable_error("fetch", req_err)
            return None
        except Exception as e:
            self.get_logger().error(f"Unexpected error processing Airtable fetch response: {e}")
            self.state = RobotState.AIRTABLE_ERROR
            self.stop_moving()
            return None

    # --- Error Logging Helper ---
    def log_airtable_error(self, action_description, request_exception):
        self.get_logger().error(f"Airtable {action_description} error: {request_exception}")
        if hasattr(request_exception, 'response') and request_exception.response is not None:
            self.get_logger().error(f"Response Status Code: {request_exception.response.status_code}")
            try:
                error_details = request_exception.response.json()
                self.get_logger().error(f"Response Body: {json.dumps(error_details)}")
            except json.JSONDecodeError:
                self.get_logger().error(f"Response Text: {request_exception.response.text}")
        self.state = RobotState.AIRTABLE_ERROR
        self.stop_moving()

    # --- Sound Utility ---
    def play_sound(self, notes):
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.GPIO_ERROR, RobotState.AIRTABLE_ERROR]:
            return
        audio_msg = AudioNoteVector()
        audio_msg.append = False
        for freq, duration_ms in notes:
            note = AudioNote()
            note.frequency = int(freq)
            note.max_runtime = Duration(sec=0, nanosec=int(duration_ms * 1_000_000))
            audio_msg.notes.append(note)
        self.get_logger().debug(f"Playing sound: {notes}")
        self.audio_publisher.publish(audio_msg)

    # --- Airtable Polling for Completion ---
    def start_airtable_polling(self):
        if self.state == RobotState.AIRTABLE_ERROR:
             self.get_logger().error("Cannot start Airtable polling, already in AIRTABLE_ERROR state.")
             return

        target_field = STATION_INDEX_TO_FIELD.get(self.target_station_index)
        if not target_field:
            self.get_logger().error(f"Polling Error: Cannot find field for logical index {self.target_station_index}")
            self.state = RobotState.ERROR
            return

        self.get_logger().info(f"Starting Airtable polling (every {AIRTABLE_POLL_RATE}s) for logical station {self.target_station_index} (Field: {target_field}) completion.")
        self.stop_airtable_polling() # Ensure no existing timer
        self.airtable_poll_timer = self.create_timer(AIRTABLE_POLL_RATE, self.airtable_poll_callback)
        self.wait_start_time = time.time()

    def stop_airtable_polling(self):
        if self.airtable_poll_timer is not None:
            if not self.airtable_poll_timer.is_canceled():
                self.airtable_poll_timer.cancel()
            self.airtable_poll_timer = None
            self.get_logger().info("Airtable polling timer stopped.")

    def airtable_poll_callback(self):
        if self.state != RobotState.WAITING_FOR_STATION_COMPLETION:
            self.get_logger().warn("Airtable poll callback executed unexpectedly. Stopping polling.")
            self.stop_airtable_polling()
            return

        if not self.current_order or "record_id" not in self.current_order:
             self.get_logger().error("Polling Error: No current order or record_id found.")
             self.state = RobotState.ERROR
             self.stop_airtable_polling()
             self.stop_moving()
             return

        record_id = self.current_order["record_id"]
        # Poll the field corresponding to the *logical* target index
        target_field = STATION_INDEX_TO_FIELD.get(self.target_station_index)

        if not target_field:
            self.get_logger().error(f"Polling Error: Invalid target station index {self.target_station_index}.")
            self.state = RobotState.ERROR
            self.stop_airtable_polling()
            self.stop_moving()
            return

        self.get_logger().debug(f"Polling Airtable: Record {record_id}, Logical Station {self.target_station_index} ({target_field})")
        current_status = self.airtable_handler.checkValue(record_id, target_field)

        if current_status is None:
            self.get_logger().error(f"Airtable check failed for {target_field}. Stopping polling and setting AIRTABLE_ERROR state.")
            self.state = RobotState.AIRTABLE_ERROR # Handler should set this, but double-check
            self.stop_airtable_polling()
            self.stop_moving()
            return

        self.get_logger().debug(f"Current status from Airtable for {target_field}: {current_status}")

        if current_status == STATUS_DONE: # 99
            self.get_logger().info(f"Logical Station {self.target_station_index} ({target_field}) reported DONE (99)!")
            self.play_sound([(600, 100), (800, 150)])
            self.stop_airtable_polling()
            # Let the main loop handle the transition after timer becomes None

        elif (time.time() - self.wait_start_time) > STATION_WAIT_TIMEOUT_SEC:
             self.get_logger().error(f"TIMEOUT waiting for logical station {self.target_station_index} ({target_field}) to reach status {STATUS_DONE}.")
             self.state = RobotState.STATION_TIMED_OUT
             self.stop_airtable_polling()
             self.stop_moving()

    # --- Main Control Loop (State Machine Logic) ---
    def control_loop(self):
        if self.state in [
                RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.GPIO_ERROR,
                RobotState.AIRTABLE_ERROR, RobotState.ALL_ORDERS_COMPLETE, RobotState.STATION_TIMED_OUT]:
            if self.state != RobotState.ALL_ORDERS_COMPLETE:
                self.get_logger().error(f"Robot in terminal/error state: {self.state.name}. Halting operations.", throttle_duration_sec=5)
            self.stop_moving()
            self.stop_airtable_polling()
            return

        left_ir, right_ir = self.read_ir_sensors()
        frame = self.picam2.capture_array()
        if CAMERA_ROTATION is not None:
            frame = cv2.rotate(frame, CAMERA_ROTATION)

        current_state = self.state
        next_state = current_state

        if current_state == RobotState.IDLE:
            self.get_logger().info("State: IDLE. Checking for new orders.")
            self.current_order = None
            self.station_sequence = []
            self.current_sequence_index = 0
            self.target_station_index = -1
            next_state = RobotState.FETCHING_ORDER

        elif current_state == RobotState.FETCHING_ORDER:
            fetched_order = self.fetch_order_from_airtable()
            if fetched_order:
                self.current_order = fetched_order
                self.get_logger().info(f"Processing Order: {self.current_order['order_name']}")
                next_state = RobotState.PLANNING_ROUTE
            elif self.state == RobotState.AIRTABLE_ERROR:
                pass # Error handled by fetch_order_from_airtable or handler init
            else: # No orders
                self.get_logger().info("No more pending orders found.")
                if self.pancakes_made_count > 0:
                    self.play_sound([(600, 100), (700, 100), (800, 300)])
                    self.get_logger().info(f"Completed {self.pancakes_made_count} order(s) this run.")
                else:
                    self.play_sound([(400, 500)])
                next_state = RobotState.ALL_ORDERS_COMPLETE

        elif current_state == RobotState.PLANNING_ROUTE:
            if not self.current_order:
                self.get_logger().error("Planning route error: No current order data.")
                next_state = RobotState.ERROR
            else:
                self.get_logger().info("Planning station route for current order...")
                self.station_sequence = []
                # Define a rough order preference (Cooks first, then toppings, then pickup)
                order_preference = [1, 2, 3, 4, 5, 0] # Logical indices

                for station_idx in order_preference:
                    station_field = STATION_INDEX_TO_FIELD.get(station_idx)
                    if station_field and self.current_order["station_status"].get(station_field, 0) == STATUS_WAITING:
                        # Check if physical station exists (skip logical station 2 here)
                        physical_idx = PHYSICAL_COOKING_STATION_INDEX if station_idx == 2 else station_idx
                        if physical_idx in STATION_COLORS_HSV or station_idx == 2: # Allow logical 2 even without physical marker
                            self.station_sequence.append(station_idx)
                            station_name = STATION_COLORS_HSV.get(physical_idx, {}).get('name', f'Logical Step {station_idx}')
                            self.get_logger().info(f" - Adding Logical Step {station_idx} ({station_name}, Field: {station_field})")
                        else:
                             self.get_logger().warn(f"Skipping station with field {station_field} - Physical Index {physical_idx} not defined in STATION_COLORS_HSV")

                self.get_logger().info(f"Planned route (logical indices): {self.station_sequence}")
                self.current_sequence_index = 0
                if not self.station_sequence:
                    self.get_logger().error("Planning route error: No valid stations in sequence for this order.")
                    next_state = RobotState.ORDER_COMPLETE # If nothing needed, maybe order is done? Or error? Let's try order complete.
                else:
                    self.target_station_index = self.station_sequence[self.current_sequence_index]
                    # Determine initial movement state based on first logical target
                    if self.target_station_index == 0:
                         next_state = RobotState.RETURNING_TO_PICKUP
                    else:
                         next_state = RobotState.MOVING_TO_STATION


        elif current_state == RobotState.MOVING_TO_STATION or current_state == RobotState.RETURNING_TO_PICKUP:
            # Determine the PHYSICAL station index to look for
            physical_target_idx = -1
            if self.target_station_index == 0: # Pickup
                physical_target_idx = 0
            elif self.target_station_index == 1: # First Cook
                physical_target_idx = PHYSICAL_COOKING_STATION_INDEX
            elif self.target_station_index == 2: # Second Cook -> Go to Physical Cooking Station
                physical_target_idx = PHYSICAL_COOKING_STATION_INDEX
            elif self.target_station_index in STATION_COLORS_HSV: # Other topping stations
                physical_target_idx = self.target_station_index
            else:
                self.get_logger().error(f"Movement error: Cannot determine physical station for logical target {self.target_station_index}")
                next_state = RobotState.ERROR
                self.stop_moving()

            if next_state != RobotState.ERROR:
                # Check for the PHYSICAL color marker
                detected, debug_frame = self.check_for_station_color(frame, physical_target_idx)

                if detected:
                    station_name = STATION_COLORS_HSV[physical_target_idx]['name']
                    self.get_logger().info(f"Physical marker for {station_name} detected (Logical Target: {self.target_station_index})")
                    self.play_sound([(500, 150)])
                    self.stop_moving()

                    if current_state == RobotState.MOVING_TO_STATION:
                        next_state = RobotState.ARRIVED_AT_STATION
                    else: # Was RETURNING_TO_PICKUP (target_station_index was 0)
                        next_state = RobotState.ARRIVED_AT_PICKUP
                else:
                    # --- IR Line Following Logic ---
                    linear_speed = BASE_DRIVE_SPEED
                    angular_speed = 0.0
                    if left_ir == GPIO.HIGH and right_ir == GPIO.HIGH:
                        angular_speed = 0.0 # Go straight
                    elif left_ir == GPIO.LOW and right_ir == GPIO.HIGH:
                        angular_speed = BASE_ROTATE_SPEED * TURN_FACTOR # Turn Left
                        linear_speed *= 0.8
                    elif left_ir == GPIO.HIGH and right_ir == GPIO.LOW:
                        angular_speed = -BASE_ROTATE_SPEED * TURN_FACTOR # Turn Right
                        linear_speed *= 0.8
                    elif left_ir == GPIO.LOW and right_ir == GPIO.LOW:
                        self.get_logger().warn("IR: Both sensors on black. Rotating slightly.", throttle_duration_sec=2)
                        linear_speed = 0.0
                        angular_speed = LOST_LINE_ROTATE_SPEED

                    self.move_robot(linear_speed, angular_speed)

                # --- Update Debug Window ---
                if self.debug_windows and debug_frame is not None:
                    ir_text = f"IR L: {left_ir} R: {right_ir}"
                    cv2.putText(debug_frame, ir_text, (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    cv2.imshow("Robot View", debug_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.get_logger().info("Quit requested via OpenCV window.")
                        self.state = RobotState.ERROR
                        rclpy.shutdown()

        elif current_state == RobotState.ARRIVED_AT_STATION:
            if not self.current_order or "record_id" not in self.current_order:
                self.get_logger().error("Arrival error: No current order/record_id.")
                next_state = RobotState.ERROR
            else:
                # Update Airtable for the *logical* station index
                station_field = STATION_INDEX_TO_FIELD.get(self.target_station_index)
                if not station_field:
                    self.get_logger().error(f"Arrival error: No Airtable field for logical station index {self.target_station_index}.")
                    next_state = RobotState.ERROR
                else:
                    physical_idx = PHYSICAL_COOKING_STATION_INDEX if self.target_station_index == 2 else self.target_station_index
                    station_name = STATION_COLORS_HSV.get(physical_idx, {}).get('name', f'Logical Step {self.target_station_index}')

                    self.get_logger().info(f"Arrived for logical step {self.target_station_index} ({station_name}, Field: {station_field}). Updating status to {STATUS_ARRIVED}.")
                    success = self.airtable_handler.changeValue(
                        self.current_order["record_id"], station_field, STATUS_ARRIVED
                    )

                    if success:
                        self.start_airtable_polling()
                        next_state = RobotState.WAITING_FOR_STATION_COMPLETION
                    else:
                        self.get_logger().error(f"Failed to update Airtable status upon arrival for {station_field}. Setting AIRTABLE_ERROR state.")
                        next_state = RobotState.AIRTABLE_ERROR # Handler should set state
                        self.stop_moving()

        elif current_state == RobotState.WAITING_FOR_STATION_COMPLETION:
            # Logic: Stay in this state while the poll timer is active.
            # The poll_callback handles checking Airtable and stopping the timer.
            if self.airtable_poll_timer is None:
                 # Timer stopped. Check why.
                 if self.state == RobotState.STATION_TIMED_OUT:
                     self.get_logger().error("Station completion timed out. Halting order processing.")
                     pass # Remain in STATION_TIMED_OUT
                 elif self.state == RobotState.AIRTABLE_ERROR:
                      self.get_logger().error("Airtable error occurred while waiting. Halting.")
                      pass # Remain in AIRTABLE_ERROR
                 elif self.state == RobotState.ERROR:
                      self.get_logger().error("Error occurred while waiting (e.g., invalid index). Halting.")
                      pass # Remain in ERROR
                 else:
                      # Assume timer stopped because STATUS_DONE (99) was detected
                      station_field = STATION_INDEX_TO_FIELD.get(self.target_station_index, "UNKNOWN_FIELD")
                      self.get_logger().info(f"Logical Station {self.target_station_index} ({station_field}) processing complete.")
                      self.current_sequence_index += 1

                      if self.current_sequence_index < len(self.station_sequence):
                          # More steps to perform
                          self.target_station_index = self.station_sequence[self.current_sequence_index]
                          next_station_field = STATION_INDEX_TO_FIELD.get(self.target_station_index, "UNKNOWN")
                          physical_idx = PHYSICAL_COOKING_STATION_INDEX if self.target_station_index == 2 else self.target_station_index
                          next_station_name = STATION_COLORS_HSV.get(physical_idx, {}).get('name', f'Logical Step {self.target_station_index}')

                          self.get_logger().info(f"Proceeding to next logical step: {self.target_station_index} ({next_station_name}, Field: {next_station_field})")

                          if self.target_station_index == 0: # Next step is Pickup
                              next_state = RobotState.RETURNING_TO_PICKUP
                          else: # Next step is another processing step
                              next_state = RobotState.MOVING_TO_STATION
                      else:
                          # All steps in the sequence are done (last one must have been Pickup)
                          self.get_logger().info("Finished all steps in the sequence.")
                          # This path shouldn't be taken if Pickup is handled by ARRIVED_AT_PICKUP state
                          # If the last step WASN'T pickup, something is wrong.
                          if self.target_station_index == 0:
                                next_state = RobotState.ORDER_COMPLETE # Should have gone to ARRIVED_AT_PICKUP
                          else:
                                self.get_logger().error("Finished sequence but last step wasn't Pickup (Index 0).")
                                next_state = RobotState.ERROR # Error if last step wasn't pickup

            # else: Timer still running, stay in WAITING_FOR_STATION_COMPLETION.

        elif current_state == RobotState.ARRIVED_AT_PICKUP:
             if not self.current_order or "record_id" not in self.current_order:
                self.get_logger().error("Pickup arrival error: No current order/record_id.")
                next_state = RobotState.ERROR
             else:
                self.get_logger().info(f"Arrived back at Pickup Station (Index 0). Order '{self.current_order['order_name']}' sequence complete.")
                self.pancakes_made_count += 1
                self.play_sound([(800, 100), (700, 100), (600, 200)])

                pickup_field = STATION_INDEX_TO_FIELD.get(0)
                if pickup_field:
                    success = self.airtable_handler.changeValue(self.current_order["record_id"], pickup_field, STATUS_DONE)
                    if not success:
                        self.get_logger().error(f"Failed to update final Pickup status ({pickup_field}) to DONE for {self.current_order['record_id']}. Continuing...")
                else:
                    self.get_logger().warn("No Airtable field mapped for Pickup Station (Index 0). Cannot update final status.")

                next_state = RobotState.ORDER_COMPLETE

        elif current_state == RobotState.ORDER_COMPLETE:
            self.get_logger().info(f"Order '{self.current_order.get('order_name', 'N/A')}' cycle finished. Returning to IDLE.")
            self.current_order = None
            self.station_sequence = []
            self.current_sequence_index = 0
            self.target_station_index = -1
            next_state = RobotState.IDLE

        # --- State Transition ---
        if next_state != current_state:
            self.get_logger().info(f"State transition: {current_state.name} -> {next_state.name}")
            self.state = next_state

    # --- Cleanup Methods ---
    def cleanup_gpio(self):
        self.get_logger().info("Cleaning up GPIO...")
        try:
            if GPIO.getmode() is not None:
                GPIO.cleanup()
                self.get_logger().info("GPIO cleanup successful.")
            else:
                self.get_logger().info("GPIO was not set up, skipping cleanup.")
        except Exception as e:
            self.get_logger().error(f"Error during GPIO cleanup: {e}")

    def shutdown_camera(self):
        if hasattr(self, 'picam2') and self.picam2:
            try:
                if self.picam2.started:
                    self.get_logger().info("Stopping Pi Camera...")
                    self.picam2.stop()
                    self.get_logger().info("Pi Camera stopped.")
            except Exception as e:
                self.get_logger().error(f"Error stopping camera: {e}")
        if self.debug_windows:
            try:
                cv2.destroyAllWindows()
                self.get_logger().info("OpenCV windows closed.")
            except Exception as e:
                 self.get_logger().error(f"Error closing OpenCV windows: {e}")

    def shutdown_robot(self):
        self.get_logger().info("Initiating robot shutdown sequence...")
        try:
            self.stop_moving()
        except Exception as e:
            self.get_logger().error(f"Error stopping robot movement: {e}")
        if hasattr(self, 'control_timer') and self.control_timer and not self.control_timer.is_canceled():
            self.control_timer.cancel()
        self.stop_airtable_polling()
        self.shutdown_camera()
        self.cleanup_gpio()
        self.get_logger().info("Robot shutdown sequence complete.")

# --- Main Execution Function ---
def main(args=None):
    rclpy.init(args=args)
    pancake_robot_node = None
    exit_code = 0
    try:
        pancake_robot_node = PancakeRobotNode()
        if pancake_robot_node.state not in [RobotState.GPIO_ERROR, RobotState.CAMERA_ERROR, RobotState.AIRTABLE_ERROR]:
            rclpy.spin(pancake_robot_node)
        else:
            pancake_robot_node.get_logger().fatal(f"Node initialization failed with state: {pancake_robot_node.state.name}. Aborting spin.")
            exit_code = 1
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected, shutting down.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during ROS execution: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        if pancake_robot_node:
            pancake_robot_node.get_logger().info("ROS shutdown requested. Initiating node cleanup...")
            pancake_robot_node.shutdown_robot()
            pancake_robot_node.destroy_node()
            pancake_robot_node.get_logger().info("Pancake Robot Node destroyed.")
        else:
             print("Node object might not exist, attempting final GPIO cleanup...")
             try:
                 if GPIO.getmode() is not None:
                     GPIO.cleanup()
                     print("GPIO cleanup attempted.")
             except Exception as e:
                 print(f"Error during final GPIO cleanup: {e}")
        if rclpy.ok():
            rclpy.shutdown()
        print("ROS2 shutdown complete.")
    return exit_code

if __name__ == '__main__':
    import sys
    sys.exit(main())