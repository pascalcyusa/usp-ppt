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
# Status for Robot 2 waiting position
AIRTABLE_ROBOT2_WAIT_STATUS_FIELD = "Cooking 2 Status"
AIRTABLE_WHIPPED_CREAM_STATUS_FIELD = "Whipped Cream Status"
AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD = "Choco Chips Status"
AIRTABLE_SPRINKLES_STATUS_FIELD = "Sprinkles Status"
AIRTABLE_PICKUP_STATUS_FIELD = "Pickup Status"

# --- Airtable Status Codes (Numeric) ---
STATUS_WAITING = 0
STATUS_ARRIVED = 1
STATUS_DONE = 99

# --- Map Airtable Fields and Order Requirements to Station Indices ---
# Maps the Airtable *Status* Field Name to the Station Index it represents
STATION_FIELD_TO_INDEX = {
    AIRTABLE_COOKING_1_STATUS_FIELD: 1,
    AIRTABLE_ROBOT2_WAIT_STATUS_FIELD: 2,  # Station 2 is Robot 2's waiting position
    AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD: 3,
    AIRTABLE_WHIPPED_CREAM_STATUS_FIELD: 4,
    AIRTABLE_SPRINKLES_STATUS_FIELD: 5,
    AIRTABLE_PICKUP_STATUS_FIELD: 0
}
# Inverse map for convenience
STATION_INDEX_TO_FIELD = {v: k for k, v in STATION_FIELD_TO_INDEX.items()}

# --- GPIO Configuration ---
LEFT_IR_PIN = 16   # BOARD pin number for Left IR sensor
RIGHT_IR_PIN = 18  # BOARD pin number for Right IR sensor

# --- Camera Configuration ---
CAMERA_RESOLUTION = (640, 480)  # Width, Height
CAMERA_ROTATION = cv2.ROTATE_180  # Adjust if camera is mounted upside down

# --- Color Detection Configuration ---
# Define HSV Lower and Upper bounds for each station's target color marker
STATION_COLORS_HSV = {
    # Index 0: Color marker to detect when returning to the start/pickup station
    0: {"name": "Pickup Station", "hsv_lower": (35, 100, 100), "hsv_upper": (85, 255, 255), "color_bgr": (255, 0, 0)},

    # Index 1: Cooking Station
    1: {"name": "Cooking Station", "hsv_lower": (35, 100, 100), "hsv_upper": (85, 255, 255), "color_bgr": (255, 0, 0)},

    # Index 2: Robot 2 Wait
    2: {"name": "Robot 2 Wait", "hsv_lower": (35, 100, 100), "hsv_upper": (85, 255, 255), "color_bgr": (255, 0, 0)},

    # Index 3: Chocolate Chips Station
    3: {"name": "Chocolate Chips", "hsv_lower": (35, 100, 100), "hsv_upper": (85, 255, 255), "color_bgr": (255, 0, 0)},

    # Index 4: Whipped Cream Station
    4: {"name": "Whipped Cream", "hsv_lower": (35, 100, 100), "hsv_upper": (85, 255, 255), "color_bgr": (255, 0, 0)},

    # Index 5: Sprinkles Station
    5: {"name": "Sprinkles", "hsv_lower": (35, 100, 100), "hsv_upper": (85, 255, 255), "color_bgr": (255, 0, 0)},
}
# Total physical stations with markers
NUM_STATIONS_PHYSICAL = len(STATION_COLORS_HSV)

# --- Navigation & Control Parameters ---
# Seconds (50 Hz) - How often to check IR sensors
IR_POLL_RATE = 0.001
# Seconds (10 Hz) - How often to check camera for colors
COLOR_POLL_RATE = 0.1
# Seconds - How often to check Airtable for '99' status
AIRTABLE_POLL_RATE = 2.0

BASE_DRIVE_SPEED = 0.01         # m/s - Forward speed during line following
BASE_ROTATE_SPEED = 0.2         # rad/s - Turning speed during line following
TURN_FACTOR = 0.7               # Aggressiveness of turns based on IR
LOST_LINE_ROTATE_SPEED = 0.1    # rad/s - Speed for rotation when line is lost

# Min pixels of target color to trigger detection (tune this!)
COLOR_DETECTION_THRESHOLD = 2000
# Min seconds before detecting the *same* station color again
COLOR_COOLDOWN_SEC = 5.0
# Max seconds to wait for Airtable status 99 before failing
STATION_WAIT_TIMEOUT_SEC = 120.0

# --- State Machine Definition ---


class RobotState(Enum):
    IDLE = auto()
    FETCHING_ORDER = auto()
    PLANNING_ROUTE = auto()         # New state to determine station sequence for the order
    MOVING_TO_STATION = auto()
    # LINE_FOLLOWING - Merged into MOVING_TO_STATION
    # STOPPING_BEFORE_PROCESS - Renamed
    ARRIVED_AT_STATION = auto()     # State after detecting color and stopping
    # PROCESSING_AT_STATION - Renamed
    WAITING_FOR_STATION_COMPLETION = auto()  # Waiting for Airtable status 99
    STATION_TIMED_OUT = auto()      # Station took too long to complete
    RETURNING_TO_PICKUP = auto()    # Moving back towards station 0
    # STOPPING_BEFORE_IDLE - Renamed
    ARRIVED_AT_PICKUP = auto()      # Final stop state before checking next order
    # Intermediate state after finishing all steps for one order
    ORDER_COMPLETE = auto()
    ALL_ORDERS_COMPLETE = auto()    # Finished all available orders in Airtable
    ERROR = auto()                  # General runtime error
    CAMERA_ERROR = auto()           # Camera initialization failed
    AIRTABLE_ERROR = auto()         # Airtable communication error or config issue
    GPIO_ERROR = auto()             # GPIO setup failed
    # LINE_LOST - Handled within MOVING_TO_STATION for now

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
        # Stores details: {'record_id': ..., 'order_name': ..., 'required_stations': [...], 'station_status': {...}}
        self.current_order = None
        # Ordered list of station indices to visit for the current order [1, 3, 4, 0]
        self.station_sequence = []
        self.current_sequence_index = 0  # Index into self.station_sequence
        # Physical station index we are currently moving towards/at
        self.target_station_index = -1
        self.pancakes_made_count = 0
        self.last_color_detection_times = {
            idx: 0.0 for idx in STATION_COLORS_HSV.keys()}
        self.wait_start_time = 0.0  # Timestamp when waiting for station completion started

        # --- Hardware Setup ---
        # GPIO (IR Sensors)
        try:
            GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering
            GPIO.setup(LEFT_IR_PIN, GPIO.IN)
            GPIO.setup(RIGHT_IR_PIN, GPIO.IN)
            self.get_logger().info(
                f"GPIO initialized (Pins: L={LEFT_IR_PIN}, R={RIGHT_IR_PIN}).")
        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to initialize GPIO: {e}")
            self.state = RobotState.GPIO_ERROR
            return

        # Camera (Picamera2)
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": CAMERA_RESOLUTION})
            self.picam2.configure(config)
            self.picam2.set_controls(
                {"AfMode": controls.AfModeEnum.Continuous, "LensPosition": 0.0})
            self.picam2.start()
            time.sleep(2)  # Allow camera to initialize and focus
            self.get_logger().info("Pi Camera initialized successfully.")
            self.debug_windows = True  # Enable/disable OpenCV display windows
        except Exception as e:
            self.get_logger().error(
                f"FATAL: Failed to initialize Pi Camera: {e}")
            # Attempt GPIO cleanup before setting error state
            self.cleanup_gpio()
            self.state = RobotState.CAMERA_ERROR
            return

        # ROS2 Publishers & Clients
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.audio_publisher = self.create_publisher(
            AudioNoteVector, '/cmd_audio', 10)
        # Keep Action Clients for potential future use (e.g., precise docking, recovery)
        self.drive_client = ActionClient(
            self, DriveDistance, '/drive_distance')
        self.rotate_client = ActionClient(self, RotateAngle, '/rotate_angle')

        # Timers
        self.control_timer_period = 0.05 # Seconds (20 Hz)
        self.control_timer = self.create_timer(
            self.control_timer_period, self.control_loop)

        # Separate timer for slower Airtable polling when waiting
        self.airtable_poll_timer = None  # Will be created/destroyed as needed

        self.get_logger().info("Pancake Robot Node Initialized and Ready.")
        self.play_sound([(440, 200), (550, 300)])  # Startup sound

    # --- Movement Control ---
    def move_robot(self, linear_x, angular_z):
        """Publishes Twist messages to control robot velocity."""
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.GPIO_ERROR, RobotState.AIRTABLE_ERROR]:
            # Ensure robot stops if entering an error state
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
        """Sends a zero velocity Twist message to stop the robot."""
        self.get_logger().info("Sending stop command (Twist zero).")
        self.move_robot(0.0, 0.0)
        time.sleep(0.1) # Ensure message is sent/processed

    # --- Sensor Reading ---
    def read_ir_sensors(self):
        """Reads the state of the left and right IR sensors."""
        try:
            left_val = GPIO.input(LEFT_IR_PIN)
            right_val = GPIO.input(RIGHT_IR_PIN)
            return left_val, right_val
        except Exception as e:
            self.get_logger().error(f"Error reading GPIO IR sensors: {e}")
            # Consider transitioning to ERROR or GPIO_ERROR state if persistent
            return GPIO.HIGH, GPIO.HIGH # Default to off-line state on error

    # --- Color Detection ---
    def check_for_station_color(self, frame, target_idx):
        """Analyzes a frame for the target station's color marker."""
        if target_idx not in STATION_COLORS_HSV:
            self.get_logger().warn(
                f"Invalid target index {target_idx} for color detection.")
            return False, None

        color_info = STATION_COLORS_HSV[target_idx]
        lower_bound = np.array(color_info["hsv_lower"])
        upper_bound = np.array(color_info["hsv_upper"])
        target_color_name = color_info["name"]
        color_bgr = color_info.get("color_bgr", (255, 255, 255))

        try:
            hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Create white background mask first
            white_lower = np.array([0, 0, 200])
            white_upper = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv_image, white_lower, white_upper)

            # Create color mask excluding white background
            color_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
            color_mask = cv2.bitwise_and(
                color_mask, cv2.bitwise_not(white_mask))

            detected_pixels = cv2.countNonZero(color_mask)

            # Create debug visualization
            debug_frame = None
            if self.debug_windows:
                debug_frame = frame.copy()
                detected_area = cv2.bitwise_and(frame, frame, mask=color_mask)
                debug_frame = cv2.addWeighted(
                    debug_frame, 1, detected_area, 0.5, 0)
                text = f"{target_color_name}: {detected_pixels} px"
                cv2.putText(debug_frame, text, (10, 30 + 30 * target_idx),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)

            # Check detection threshold and cooldown
            current_time = time.time()
            if detected_pixels > COLOR_DETECTION_THRESHOLD and \
               (current_time - self.last_color_detection_times.get(target_idx, 0.0) > COLOR_COOLDOWN_SEC):
                self.get_logger().info(
                    f"Detected {target_color_name} (Index {target_idx})!")
                self.last_color_detection_times[target_idx] = current_time
                return True, debug_frame

            return False, debug_frame

        except cv2.error as cv_err:
            self.get_logger().error(
                f"OpenCV error during color detection: {cv_err}")
            return False, None
        except Exception as e:
            self.get_logger().error(
                f"Unexpected error during color detection: {e}")
            return False, None

    # --- Airtable Communication (Fetch Only - Update/Check use Handler) ---
    def fetch_order_from_airtable(self):
        """Fetches the oldest order that needs processing (using direct requests)."""
        self.get_logger().info("Attempting to fetch next order from Airtable...")
        params = {
            "maxRecords": 1,
            "filterByFormula": f"AND({{{AIRTABLE_COOKING_1_STATUS_FIELD}}}=0, {{{AIRTABLE_PICKUP_STATUS_FIELD}}}=0)",
            "sort[0][field]": AIRTABLE_CREATED_TIME_FIELD,
            "sort[0][direction]": "asc",
            "fields[]": [
                AIRTABLE_ORDER_NAME_COLUMN,
                AIRTABLE_CREATED_TIME_FIELD,
                AIRTABLE_COOKING_1_STATUS_FIELD,
                AIRTABLE_ROBOT2_WAIT_STATUS_FIELD,
                AIRTABLE_WHIPPED_CREAM_STATUS_FIELD,
                AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD,
                AIRTABLE_SPRINKLES_STATUS_FIELD,
                AIRTABLE_PICKUP_STATUS_FIELD
            ]
        }
        self.get_logger().debug(f"Airtable fetch query params: {params}")

        try:
            # Use the specific URL and Headers defined for fetching
            response = requests.get(
                url=AIRTABLE_FETCH_URL, headers=AIRTABLE_FETCH_HEADERS, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            self.get_logger().debug(
                f"Airtable fetch response: {json.dumps(data, indent=2)}")

            records = data.get("records", [])
            if records:
                record = records[0]
                record_id = record.get("id")
                fields = record.get("fields", {})
                order_name = fields.get(AIRTABLE_ORDER_NAME_COLUMN)
                created_time = fields.get(AIRTABLE_CREATED_TIME_FIELD)

                if not record_id or not order_name:
                    self.get_logger().error(
                        f"Fetched record missing ID or Name: {record}")
                    return None

                fetched_order = {
                    "record_id": record_id,
                    "order_name": order_name,
                    "created_time": created_time,
                    "station_status": {
                        # Populate with actual fetched statuses or default to 0
                        f: fields.get(f, 0) for f in STATION_FIELD_TO_INDEX.keys() if f # Filter out None keys if any
                    }
                }
                self.get_logger().info(
                    f"Fetched order: '{order_name}' (Created: {created_time}, Record ID: {record_id})")
                self.get_logger().debug(f"Order details: {fetched_order}")
                return fetched_order
            else:
                self.get_logger().info("No suitable pending orders found in Airtable.")
                return None

        except requests.exceptions.RequestException as req_err:
            self.log_airtable_error("fetch", req_err) # Use helper for logging fetch errors
            return None
        except Exception as e:
            self.get_logger().error(
                f"Unexpected error processing Airtable fetch response: {e}")
            # Potentially set state to AIRTABLE_ERROR here too
            self.state = RobotState.AIRTABLE_ERROR
            self.stop_moving()
            return None

    # --- Removed update_station_status_in_airtable ---
    # --- Removed check_station_status_in_airtable ---

    # --- Error Logging Helper (kept for fetch_order) ---
    def log_airtable_error(self, action_description, request_exception):
        """Helper to log detailed Airtable request errors (primarily for fetch)."""
        self.get_logger().error(
            f"Airtable {action_description} error: {request_exception}")
        if hasattr(request_exception, 'response') and request_exception.response is not None:
            self.get_logger().error(
                f"Response Status Code: {request_exception.response.status_code}")
            try:
                error_details = request_exception.response.json()
                self.get_logger().error(
                    f"Response Body: {json.dumps(error_details)}")
            except json.JSONDecodeError:
                self.get_logger().error(
                    f"Response Text: {request_exception.response.text}")
        # Set state to AIRTABLE_ERROR for fetch failures
        self.state = RobotState.AIRTABLE_ERROR
        self.stop_moving()


    # --- Sound Utility ---
    def play_sound(self, notes):
        """Publishes a sequence of AudioNotes."""
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.GPIO_ERROR, RobotState.AIRTABLE_ERROR]:
            return

        audio_msg = AudioNoteVector()
        audio_msg.append = False # Play immediately
        for freq, duration_ms in notes:
            note = AudioNote()
            note.frequency = int(freq)
            note.max_runtime = Duration(
                sec=0, nanosec=int(duration_ms * 1_000_000))
            audio_msg.notes.append(note)
        self.get_logger().debug(f"Playing sound: {notes}")
        self.audio_publisher.publish(audio_msg)

    # --- Airtable Polling for Completion ---
    def start_airtable_polling(self):
        if self.state == RobotState.AIRTABLE_ERROR:
             self.get_logger().error("Cannot start Airtable polling, already in AIRTABLE_ERROR state.")
             return

        self.get_logger().info(
            f"Starting Airtable polling timer (every {AIRTABLE_POLL_RATE}s) for station {self.target_station_index} completion.")
        # Ensure no existing timer is running
        self.stop_airtable_polling()
        self.airtable_poll_timer = self.create_timer(
            AIRTABLE_POLL_RATE, self.airtable_poll_callback)
        self.wait_start_time = time.time() # Reset timeout timer

    def stop_airtable_polling(self):
        """Cancels and destroys the Airtable polling timer."""
        if self.airtable_poll_timer is not None:
            if not self.airtable_poll_timer.is_canceled():
                self.airtable_poll_timer.cancel()
            # self.destroy_timer(self.airtable_poll_timer) # Use if on Humble+
            self.airtable_poll_timer = None
            self.get_logger().info("Airtable polling timer stopped.")

    def airtable_poll_callback(self):
        """Callback function for the Airtable polling timer (uses AirtableHandler)."""
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
        target_field = STATION_INDEX_TO_FIELD.get(self.target_station_index)

        if not target_field:
            self.get_logger().error(f"Polling Error: Invalid target station index {self.target_station_index}.")
            self.state = RobotState.ERROR
            self.stop_airtable_polling()
            self.stop_moving()
            return

        self.get_logger().debug( # Make polling check less verbose
            f"Polling Airtable: Record {record_id}, Station {self.target_station_index} ({target_field})")

        # --- Use AirtableHandler to check value ---
        current_status = self.airtable_handler.checkValue(record_id, target_field)

        if current_status is None:
            # checkValue prints its own errors, but we need to handle the failure here
            self.get_logger().error(f"Airtable check failed for {target_field}. Stopping polling and setting ERROR state.")
            self.state = RobotState.AIRTABLE_ERROR
            self.stop_airtable_polling()
            self.stop_moving()
            return

        self.get_logger().debug(f"Current status from Airtable: {current_status}")

        # Check if station is done
        if current_status == STATUS_DONE: # This is 99
            self.get_logger().info(
                f"Station {self.target_station_index} ({target_field}) reported DONE (99)!")
            self.play_sound([(600, 100), (800, 150)])
            self.stop_airtable_polling()

            # --- Transition logic moved back to main loop after polling stops ---
            # Set a flag or let the main loop detect timer stopped?
            # For simplicity, let the main loop handle the transition when self.airtable_poll_timer becomes None

        # Check for timeout
        elif (time.time() - self.wait_start_time) > STATION_WAIT_TIMEOUT_SEC:
             self.get_logger().error(f"TIMEOUT waiting for station {self.target_station_index} ({target_field}) to reach status {STATUS_DONE}.")
             self.state = RobotState.STATION_TIMED_OUT
             self.stop_airtable_polling()
             self.stop_moving()
             # State machine will handle STATION_TIMED_OUT state in main loop


    # --- Main Control Loop (State Machine Logic) ---
    def control_loop(self):
        """The core state machine logic, incorporating sensor checks and Airtable updates."""

        # --- Terminal/Error State Handling ---
        if self.state in [
                RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.GPIO_ERROR,
                RobotState.AIRTABLE_ERROR, RobotState.ALL_ORDERS_COMPLETE, RobotState.STATION_TIMED_OUT]:
            if self.state != RobotState.ALL_ORDERS_COMPLETE: # Avoid logging repeatedly for normal completion
                self.get_logger().error(
                    f"Robot in terminal/error state: {self.state.name}. Halting operations.", throttle_duration_sec=5)
            self.stop_moving()
            self.stop_airtable_polling() # Ensure timer is stopped in any final state
            # Don't cancel control_timer, let spin manage shutdown
            return # Exit loop iteration

        # --- Sensor Reading and Processing (executed every loop) ---
        left_ir, right_ir = self.read_ir_sensors()
        frame = self.picam2.capture_array()
        if CAMERA_ROTATION is not None:
            frame = cv2.rotate(frame, CAMERA_ROTATION)

        # --- State Machine Logic ---
        current_state = self.state
        next_state = current_state # Default to no state change

        # --- State Implementations ---
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
                self.get_logger().info(
                    f"Processing Order: {self.current_order['order_name']}")
                next_state = RobotState.PLANNING_ROUTE
            elif self.state == RobotState.AIRTABLE_ERROR:
                # Error handled by fetch_order_from_airtable or handler init
                pass # Stay in AIRTABLE_ERROR
            else: # No orders found and no error
                self.get_logger().info("No more pending orders found.")
                if self.pancakes_made_count > 0:
                    self.play_sound([(600, 100), (700, 100), (800, 300)])
                    self.get_logger().info(
                        f"Completed {self.pancakes_made_count} order(s) this run.")
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
                # Add stations based on their status fields being WAITING (0)
                # Ensure Pickup (0) is always the last station if it's needed
                pickup_needed = False
                for station_field, station_idx in STATION_FIELD_TO_INDEX.items():
                    if station_idx == 0: # Handle pickup separately
                         if self.current_order["station_status"].get(station_field, 0) == STATUS_WAITING:
                              pickup_needed = True
                         continue # Don't add pickup to sequence yet

                    # Add other stations if waiting
                    if self.current_order["station_status"].get(station_field, 0) == STATUS_WAITING:
                         # Ensure the index exists in our color map
                         if station_idx in STATION_COLORS_HSV:
                             self.station_sequence.append(station_idx)
                             self.get_logger().info(
                                 f" - Adding Station {station_idx} ({STATION_COLORS_HSV[station_idx]['name']})")
                         else:
                              self.get_logger().warn(f"Skipping station with field {station_field} - Index {station_idx} not defined in STATION_COLORS_HSV")

                # Add Pickup station (index 0) at the end if it was marked as waiting
                if pickup_needed:
                     if 0 in STATION_COLORS_HSV:
                         self.station_sequence.append(0)
                         self.get_logger().info(f" - Adding Final Station 0 ({STATION_COLORS_HSV[0]['name']})")
                     else:
                          self.get_logger().warn("Pickup station needed (Status=0) but Index 0 not defined in STATION_COLORS_HSV")


                self.get_logger().info(
                    f"Planned route (indices): {self.station_sequence}")
                self.current_sequence_index = 0
                if not self.station_sequence:
                    # This might happen if only Pickup was 0 and it wasn't in STATION_COLORS_HSV
                    self.get_logger().error("Planning route error: No valid stations in sequence.")
                    next_state = RobotState.ERROR # Or perhaps ORDER_COMPLETE? Let's error for now.
                else:
                    self.target_station_index = self.station_sequence[self.current_sequence_index]
                    if self.target_station_index in STATION_COLORS_HSV:
                        self.get_logger().info(
                            f"First target station: {self.target_station_index} ({STATION_COLORS_HSV[self.target_station_index]['name']})")
                        # Determine initial movement state based on first target
                        if self.target_station_index == 0:
                             next_state = RobotState.RETURNING_TO_PICKUP # Should be rare if Pickup is last
                        else:
                             next_state = RobotState.MOVING_TO_STATION
                    else:
                         self.get_logger().error(f"First target station index {self.target_station_index} not in STATION_COLORS_HSV.")
                         next_state = RobotState.ERROR


        elif current_state == RobotState.MOVING_TO_STATION or current_state == RobotState.RETURNING_TO_PICKUP:
            if self.target_station_index < 0 or self.target_station_index not in STATION_COLORS_HSV:
                self.get_logger().error(f"Movement error: Invalid target station index {self.target_station_index}")
                next_state = RobotState.ERROR
                self.stop_moving()
            else:
                # --- Color Check First ---
                detected, debug_frame = self.check_for_station_color(
                    frame, self.target_station_index)
                if detected:
                    station_name = STATION_COLORS_HSV[self.target_station_index]['name']
                    self.get_logger().info(
                        f"Target color marker for {station_name} (Index {self.target_station_index}) DETECTED!")
                    self.play_sound([(500, 150)]) # Arrival beep
                    self.stop_moving()

                    if current_state == RobotState.MOVING_TO_STATION:
                        next_state = RobotState.ARRIVED_AT_STATION
                    else: # Was RETURNING_TO_PICKUP
                        next_state = RobotState.ARRIVED_AT_PICKUP

                else:
                    # --- IR Line Following Logic ---
                    linear_speed = BASE_DRIVE_SPEED
                    angular_speed = 0.0
                    # LOW = On Line (Black), HIGH = Off Line (White/Floor)
                    if left_ir == GPIO.HIGH and right_ir == GPIO.HIGH: # On white (between lines)
                        angular_speed = 0.0 # Go straight
                    elif left_ir == GPIO.LOW and right_ir == GPIO.HIGH: # Left sensor on black
                        angular_speed = BASE_ROTATE_SPEED * TURN_FACTOR # Turn Left
                        linear_speed *= 0.8
                    elif left_ir == GPIO.HIGH and right_ir == GPIO.LOW: # Right sensor on black
                        angular_speed = -BASE_ROTATE_SPEED * TURN_FACTOR # Turn Right
                        linear_speed *= 0.8
                    elif left_ir == GPIO.LOW and right_ir == GPIO.LOW: # Both sensors on black (line lost?)
                        self.get_logger().warn("IR: Both sensors on black. Rotating slightly.", throttle_duration_sec=2)
                        linear_speed = 0.0
                        angular_speed = LOST_LINE_ROTATE_SPEED # Rotate to find line

                    self.move_robot(linear_speed, angular_speed)

                # --- Update Debug Window ---
                if self.debug_windows and debug_frame is not None:
                    ir_text = f"IR L: {left_ir} R: {right_ir}"
                    cv2.putText(debug_frame, ir_text, (frame.shape[1] - 150, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    # Combine original, debug, and HSV views if desired
                    # cv2.imshow("Camera View", frame)
                    cv2.imshow("Robot View (with detection)", debug_frame)
                    # cv2.imshow("HSV View", cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
                    # cv2.moveWindow("Robot View (with detection)", ...)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.get_logger().info("Quit requested via OpenCV window.")
                        self.state = RobotState.ERROR # Trigger shutdown sequence
                        rclpy.shutdown()

        elif current_state == RobotState.ARRIVED_AT_STATION:
            if not self.current_order or "record_id" not in self.current_order:
                self.get_logger().error("Arrival error: No current order/record_id.")
                next_state = RobotState.ERROR
            else:
                station_field = STATION_INDEX_TO_FIELD.get(self.target_station_index)
                if not station_field:
                    self.get_logger().error(
                        f"Arrival error: No Airtable field for station index {self.target_station_index}.")
                    next_state = RobotState.ERROR
                else:
                    station_name = STATION_COLORS_HSV.get(self.target_station_index,{}).get('name', f'Index {self.target_station_index}')
                    self.get_logger().info(
                        f"Arrived at {station_name} ({station_field}). Updating status to {STATUS_ARRIVED}.")

                    # --- Use AirtableHandler to change value ---
                    success = self.airtable_handler.changeValue(
                        self.current_order["record_id"], station_field, STATUS_ARRIVED
                    )

                    if success:
                        # Start polling timer only if update was successful
                        self.start_airtable_polling() # Starts timer and resets wait_start_time
                        next_state = RobotState.WAITING_FOR_STATION_COMPLETION
                    else:
                        # changeValue prints errors, set state here
                        self.get_logger().error(
                            f"Failed to update Airtable status upon arrival at {station_field}. Setting AIRTABLE_ERROR state.")
                        next_state = RobotState.AIRTABLE_ERROR
                        self.stop_moving()


        elif current_state == RobotState.WAITING_FOR_STATION_COMPLETION:
            # Logic is now: Stay in this state *while* the poll timer is active.
            # The poll_callback handles checking Airtable and stopping the timer (on success or timeout).
            # If the timer is stopped (becomes None), *then* transition.

            if self.airtable_poll_timer is None:
                 # Timer stopped. Check *why* it stopped based on current state.
                 if self.state == RobotState.STATION_TIMED_OUT:
                     # Timeout occurred in callback, error handling already done. Stay in this state.
                     self.get_logger().error("Station completion timed out. Halting order processing.")
                     pass # Remain in STATION_TIMED_OUT
                 elif self.state == RobotState.AIRTABLE_ERROR:
                      # Airtable error occurred during polling.
                      self.get_logger().error("Airtable error occurred while waiting. Halting.")
                      pass # Remain in AIRTABLE_ERROR
                 else:
                      # Assume timer stopped because STATUS_DONE (99) was detected by callback
                      self.get_logger().info(
                          f"Station {self.target_station_index} processing complete (detected by poll timer).")
                      self.current_sequence_index += 1 # Move to next station in sequence

                      if self.current_sequence_index < len(self.station_sequence):
                          # More stations to visit
                          self.target_station_index = self.station_sequence[self.current_sequence_index]
                          if self.target_station_index in STATION_COLORS_HSV:
                              next_station_name = STATION_COLORS_HSV[self.target_station_index]['name']
                              self.get_logger().info(
                                  f"Proceeding to next station: {self.target_station_index} ({next_station_name})")
                              if self.target_station_index == 0: # Next station is Pickup
                                  next_state = RobotState.RETURNING_TO_PICKUP
                              else: # Next station is another processing station
                                  next_state = RobotState.MOVING_TO_STATION
                          else:
                               self.get_logger().error(f"Next target station index {self.target_station_index} not in STATION_COLORS_HSV.")
                               next_state = RobotState.ERROR
                               self.stop_moving()
                      else:
                          # Should not happen if Pickup (0) is always last
                          self.get_logger().error("Reached end of sequence unexpectedly after waiting. This shouldn't happen if Pickup is last.")
                          # Perhaps order is complete? Or error? Let's assume error for now.
                          next_state = RobotState.ERROR
                          self.stop_moving()
            # else: Timer still running, stay in WAITING_FOR_STATION_COMPLETION state.


        elif current_state == RobotState.ARRIVED_AT_PICKUP:
             if not self.current_order or "record_id" not in self.current_order:
                self.get_logger().error("Pickup arrival error: No current order/record_id.")
                next_state = RobotState.ERROR
             else:
                self.get_logger().info(
                    f"Arrived back at Pickup Station (Index 0). Order '{self.current_order['order_name']}' complete.")
                self.pancakes_made_count += 1
                self.play_sound([(800, 100), (700, 100), (600, 200)]) # Order complete sound

                # Update Pickup status to DONE (99)
                pickup_field = STATION_INDEX_TO_FIELD.get(0)
                if pickup_field:
                     # --- Use AirtableHandler to change value ---
                    success = self.airtable_handler.changeValue(
                         self.current_order["record_id"], pickup_field, STATUS_DONE
                    )
                    if not success:
                        # Log error but continue to next order
                        self.get_logger().error(f"Failed to update final Pickup status ({pickup_field}) to DONE for {self.current_order['record_id']}. Continuing...")
                        # Potential: Set a flag or log persistently if this fails often
                else:
                    self.get_logger().warn(
                        "No Airtable field mapped for Pickup Station (Index 0). Cannot update final status.")

                # Transition to check for next order regardless of final update success? Yes.
                next_state = RobotState.ORDER_COMPLETE

        elif current_state == RobotState.ORDER_COMPLETE:
            # Reset for next potential order
            self.get_logger().info(f"Order '{self.current_order.get('order_name', 'N/A')}' cycle finished. Returning to IDLE.")
            self.current_order = None
            self.station_sequence = []
            self.current_sequence_index = 0
            self.target_station_index = -1
            next_state = RobotState.IDLE

        # --- State Transition ---
        if next_state != current_state:
            self.get_logger().info(
                f"State transition: {current_state.name} -> {next_state.name}")
            self.state = next_state

    # --- Cleanup Methods ---
    def cleanup_gpio(self):
        """Cleans up GPIO resources."""
        self.get_logger().info("Cleaning up GPIO...")
        try:
            # Check if GPIO has been set up before cleaning
            if GPIO.getmode() is not None:
                GPIO.cleanup()
                self.get_logger().info("GPIO cleanup successful.")
            else:
                self.get_logger().info("GPIO was not set up, skipping cleanup.")
        except Exception as e:
            self.get_logger().error(f"Error during GPIO cleanup: {e}")

    def shutdown_camera(self):
        """Safely stops the Pi Camera and closes OpenCV windows."""
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
        """Commands the robot to stop moving and cleans up resources."""
        self.get_logger().info("Initiating robot shutdown sequence...")
        # 1. Stop movement
        try:
            self.stop_moving()
        except Exception as e:
            self.get_logger().error(f"Error stopping robot movement: {e}")

        # 2. Cancel timers
        if hasattr(self, 'control_timer') and self.control_timer and not self.control_timer.is_canceled():
            self.control_timer.cancel()
        self.stop_airtable_polling()  # Ensure polling timer is stopped

        # 3. Stop camera and close windows
        self.shutdown_camera()

        # 4. Clean up GPIO
        self.cleanup_gpio()
        self.get_logger().info("Robot shutdown sequence complete.")

# --- Main Execution Function ---
def main(args=None):
    rclpy.init(args=args)
    pancake_robot_node = None
    exit_code = 0
    try:
        pancake_robot_node = PancakeRobotNode()

        # Check for fatal initialization errors before spinning
        if pancake_robot_node.state not in [RobotState.GPIO_ERROR, RobotState.CAMERA_ERROR, RobotState.AIRTABLE_ERROR]:
            rclpy.spin(pancake_robot_node)
        else:
            pancake_robot_node.get_logger().fatal(
                f"Node initialization failed with state: {pancake_robot_node.state.name}. Aborting spin.")
            exit_code = 1 # Indicate error exit

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected, shutting down.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during ROS execution: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        # Cleanup sequence
        if pancake_robot_node:
            pancake_robot_node.get_logger().info(
                "ROS shutdown requested. Initiating node cleanup...")
            pancake_robot_node.shutdown_robot() # Calls stop, cancels timers, cleans GPIO/Camera
            pancake_robot_node.destroy_node()
            pancake_robot_node.get_logger().info("Pancake Robot Node destroyed.")
        else:
            # Attempt cleanup even if node init failed partially
             print("Node object might not exist, attempting final GPIO cleanup...")
             try:
                 if GPIO.getmode() is not None:
                     GPIO.cleanup()
                     print("GPIO cleanup attempted.")
             except Exception as e:
                 print(f"Error during final GPIO cleanup: {e}")


        # Shutdown ROS client library
        if rclpy.ok():
            rclpy.shutdown()
        print("ROS2 shutdown complete.")


    return exit_code


if __name__ == '__main__':
    import sys
    sys.exit(main())