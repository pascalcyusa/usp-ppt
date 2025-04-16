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
    # Log this error using rclpy logger if possible, otherwise raise
    # Can't use self.get_logger() here as the node isn't initialized yet.
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
AIRTABLE_COOKING_2_STATUS_FIELD = "Cooking 2 Status" # NOTE: Index 2 currently has no color defined
AIRTABLE_WHIPPED_CREAM_STATUS_FIELD = "Whipped Cream Status"
AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD = "Choco Chips Status"
AIRTABLE_SPRINKLES_STATUS_FIELD = "Sprinkles Status"
AIRTABLE_PICKUP_STATUS_FIELD = "Pickup Status"

# --- Airtable Status Codes (Numeric) ---
STATUS_WAITING = 0
STATUS_ARRIVED = 1
STATUS_DONE = 99

# --- Map Airtable Fields and Order Requirements to Station Indices ---
# Ensure these indices have corresponding entries in STATION_COLORS_HSV if visual detection is needed
STATION_FIELD_TO_INDEX = {
    AIRTABLE_COOKING_1_STATUS_FIELD: 1,  # First cooking station
    AIRTABLE_COOKING_2_STATUS_FIELD: 2,  # Second cooking station (NEEDS COLOR MAPPING if visual stop needed)
    AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD: 3,
    AIRTABLE_WHIPPED_CREAM_STATUS_FIELD: 4,
    AIRTABLE_SPRINKLES_STATUS_FIELD: 5,
    AIRTABLE_PICKUP_STATUS_FIELD: 0      # Pickup station (usually start/end)
}
STATION_INDEX_TO_FIELD = {v: k for k, v in STATION_FIELD_TO_INDEX.items()}

# --- Hardware Configuration ---
LEFT_IR_PIN = 16
RIGHT_IR_PIN = 18
# Assume LOW signal means the sensor is OVER the line
IR_LINE_DETECT_SIGNAL = GPIO.LOW # Or GPIO.HIGH if HIGH means on line
IR_OFF_LINE_SIGNAL = GPIO.HIGH   # Or GPIO.LOW if HIGH means on line

CAMERA_RESOLUTION = (640, 480)
# Use picamera2's Transform for rotation instead of cv2.rotate
CAMERA_TRANSFORM = Transform(hflip=True, vflip=True) # Equivalent to cv2.ROTATE_180

# --- Color Detection Configuration ---
# BGR format for color_bgr used by OpenCV text/drawing
STATION_COLORS_HSV = {
    # Index 0: Pickup Station (e.g., Green)
    0: {"name": "Pickup Station", "hsv_lower": (35, 80, 80), "hsv_upper": (85, 255, 255), "color_bgr": (0, 255, 0)},
    # Index 1: Cooking Station 1 (e.g., Blue)
    1: {"name": "Cooking Station 1", "hsv_lower": (100, 100, 100), "hsv_upper": (130, 255, 255), "color_bgr": (255, 0, 0)},
    # Index 2: Cooking Station 2 - NO COLOR DEFINED - Robot won't stop visually for this!
    # 2: {"name": "Cooking Station 2", ...},
    # Index 3: Chocolate Chips (e.g., Red-Orange)
    3: {"name": "Chocolate Chips", "hsv_lower": (0, 120, 120), "hsv_upper": (15, 255, 255), "color_bgr": (0, 100, 255)},
    # Index 4: Whipped Cream (e.g., Yellow)
    4: {"name": "Whipped Cream", "hsv_lower": (20, 100, 100), "hsv_upper": (35, 255, 255), "color_bgr": (0, 255, 255)},
    # Index 5: Sprinkles (e.g., Pink/Magenta)
    5: {"name": "Sprinkles", "hsv_lower": (140, 80, 80), "hsv_upper": (170, 255, 255), "color_bgr": (255, 0, 255)},
}

# --- Navigation & Control Parameters ---
BASE_DRIVE_SPEED = 0.04 # Increased slightly
BASE_ROTATE_SPEED = 0.3 # Increased slightly
TURN_FACTOR = 0.8       # Reduce speed slightly more when turning
LOST_LINE_ROTATE_SPEED = 0.2 # Speed for rotating when line is lost
COLOR_DETECTION_THRESHOLD = 1500 # Pixels needed to trigger detection
COLOR_COOLDOWN_SEC = 4.0 # Min seconds between detecting the same color marker
STATION_WAIT_TIMEOUT_SEC = 180.0 # Max time to wait for station completion

class RobotState(Enum):
    IDLE = auto()
    FETCHING_ORDER = auto() # Could potentially merge with IDLE
    PLANNING_ROUTE = auto()
    MOVING_TO_STATION = auto()
    ARRIVED_AT_STATION = auto()
    WAITING_FOR_STATION_COMPLETION = auto()
    STATION_TIMED_OUT = auto()
    # REMOVED RETURNING_TO_PICKUP (Pickup is now the last station in sequence)
    # REMOVED ARRIVED_AT_PICKUP (Handled by generic ARRIVED_AT_STATION)
    ORDER_COMPLETE = auto() # Order finished, ready for next
    ALL_ORDERS_COMPLETE = auto() # No orders found, waiting
    ERROR = auto() # General unrecoverable error
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
        self.current_sequence_index = -1 # Start at -1, will be 0 when planning starts
        self.target_station_index = -1 # Which station index we are currently looking for
        self.last_color_detection_times = {idx: 0.0 for idx in STATION_COLORS_HSV.keys()}
        self.wait_start_time = 0.0
        self.picam2 = None # Initialize camera object to None
        self.debug_windows = True # Control whether OpenCV windows are shown

        # Initialize hardware
        self._init_hardware()
        # Exit if hardware init failed
        if self.state in [RobotState.GPIO_ERROR, RobotState.CAMERA_ERROR]:
            self.get_logger().fatal("Hardware initialization failed. Node cannot operate.")
            # No point in initializing ROS components if hardware failed
            return

        # Initialize ROS2 publishers and clients
        self._init_ros2()

        # Initialize timers
        # Use a slightly longer timer interval to reduce CPU load, 0.05 = 20Hz
        self.control_timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info("Pancake Robot Node Initialized and Ready.")
        self.play_sound([(440, 150), (550, 200)]) # Initialization sound

    def _init_hardware(self):
        """Initialize GPIO and Camera"""
        # GPIO Setup
        try:
            GPIO.setmode(GPIO.BOARD) # Use Raspberry Pi board pin numbers
            GPIO.setup(LEFT_IR_PIN, GPIO.IN)
            GPIO.setup(RIGHT_IR_PIN, GPIO.IN)
            self.get_logger().info(f"GPIO initialized (Pins: L={LEFT_IR_PIN}, R={RIGHT_IR_PIN}). Expecting LOW on line.")
        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to initialize GPIO: {e}")
            self.state = RobotState.GPIO_ERROR
            return # Stop initialization

        # Camera Setup
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": CAMERA_RESOLUTION},
                transform=CAMERA_TRANSFORM # Apply rotation/flip here
            )
            self.picam2.configure(config)
            # Set autofocus mode
            self.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous, "LensPosition": 0.0})
            self.picam2.start()
            time.sleep(2) # Allow camera to warm up and focus
            self.get_logger().info("Pi Camera initialized successfully.")
            if self.debug_windows:
                cv2.namedWindow("Camera Feed")
                cv2.namedWindow("Color Detection Mask") # Separate window for mask
        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to initialize Pi Camera: {e}", exc_info=True)
            self.cleanup_gpio() # Clean up GPIO if camera fails
            self.state = RobotState.CAMERA_ERROR
            return # Stop initialization

    def _init_ros2(self):
        """Initialize ROS2 publishers and action clients"""
        try:
            self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
            self.audio_publisher = self.create_publisher(AudioNoteVector, '/cmd_audio', 10)
            # TODO: Add checks to ensure action servers are available?
            self.drive_client = ActionClient(self, DriveDistance, '/drive_distance')
            self.rotate_client = ActionClient(self, RotateAngle, '/rotate_angle')
            self.get_logger().info("ROS2 publishers and action clients initialized.")
        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to initialize ROS2 components: {e}")
            # Consider setting an error state here too
            self.state = RobotState.ERROR # General ROS error


    def fetch_order_from_airtable(self):
        """Fetches the oldest order where Cooking 1 is WAITING and Pickup is WAITING."""
        self.get_logger().info("Attempting to fetch order from Airtable...")
        try:
            # Filter for orders that haven't started (Cooking 1 = 0) and haven't finished (Pickup = 0)
            # This assumes Cooking 1 is always the *first* mandatory step for a new order.
            params = {
                "maxRecords": 1,
                # Formula checks if the required first step and the final step are both waiting
                "filterByFormula": f"AND({{{AIRTABLE_COOKING_1_STATUS_FIELD}}}=0, {{{AIRTABLE_PICKUP_STATUS_FIELD}}}=0)",
                "sort[0][field]": AIRTABLE_ORDER_NAME_COLUMN, # Or use 'Created' field if available
                "sort[0][direction]": "asc"
            }

            response = requests.get(url=AIRTABLE_URL, headers=AIRTABLE_HEADERS, params=params, timeout=15) # Added timeout
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
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
            # Return a structured dictionary
            return {
                "record_id": record_id,
                "order_name": order_name,
                "station_status": {
                    field: fields.get(field, 0) # Default to 0 if field is missing in Airtable
                    for field in STATION_FIELD_TO_INDEX.keys() # Iterate through all known station fields
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
            response = requests.patch(
                url=url,
                headers=AIRTABLE_HEADERS,
                json=update_data,
                timeout=10 # Add timeout
            )
            response.raise_for_status() # Check for HTTP errors
            self.get_logger().info(f"Airtable update successful for {station_field_name}.")
            return True
        except requests.exceptions.Timeout:
            self.get_logger().error(f"Airtable update request timed out for {station_field_name}.")
            self.state = RobotState.AIRTABLE_ERROR
            return False
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Failed to update Airtable status for {station_field_name}: {e}")
            # Check response content if available
            try:
                self.get_logger().error(f"Airtable response content: {response.text}")
            except: pass # Ignore if response object doesn't exist or has no text
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
            response = requests.get(
                url=url,
                headers=AIRTABLE_HEADERS,
                timeout=10 # Add timeout
            )
            response.raise_for_status()
            data = response.json()
            current_status = data.get('fields', {}).get(station_field_name)
            # self.get_logger().debug(f"Checked Airtable status for {station_field_name}: {current_status}") # Verbose logging
            return current_status == STATUS_DONE

        except requests.exceptions.Timeout:
            self.get_logger().error(f"Airtable status check request timed out for {station_field_name}.")
            # Don't change state here, just return False, let timeout handle state change
            return False
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Error checking Airtable station status ({station_field_name}): {e}")
            return False # Assume not done if check fails
        except Exception as e:
            self.get_logger().error(f"Unexpected error during Airtable status check: {e}", exc_info=True)
            return False

    def move_robot(self, linear_x, angular_z):
        """Publishes Twist messages to control robot velocity."""
        # Prevent movement if in an error state or hardware not ready
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.GPIO_ERROR, RobotState.AIRTABLE_ERROR] or not self.cmd_vel_pub:
            # Ensure robot is stopped if attempting to move in error state
            # self.stop_moving() # Be careful not to create infinite loops if stop_moving calls move_robot(0,0)
            twist_msg = Twist()
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            if self.cmd_vel_pub:
                 self.cmd_vel_pub.publish(twist_msg)
            return

        twist_msg = Twist()
        twist_msg.linear.x = float(linear_x)
        twist_msg.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(twist_msg)

    def stop_moving(self):
        """Stops the robot movement reliably."""
        self.get_logger().info("Stopping robot...")
        # Send stop command multiple times for reliability
        for _ in range(3):
            self.move_robot(0.0, 0.0)
            time.sleep(0.02) # Short delay between stop commands

    def read_ir_sensors(self):
        """Reads the state of the IR line sensors. Returns (left_on_line, right_on_line)."""
        try:
            left_val = GPIO.input(LEFT_IR_PIN)
            right_val = GPIO.input(RIGHT_IR_PIN)
            # Return True if the sensor value matches the "on line" signal level
            left_on_line = (left_val == IR_LINE_DETECT_SIGNAL)
            right_on_line = (right_val == IR_LINE_DETECT_SIGNAL)
            # self.get_logger().debug(f"IR Sensors: L={left_val}({left_on_line}), R={right_val}({right_on_line})")
            return left_on_line, right_on_line
        except Exception as e:
            self.get_logger().error(f"IR sensor read error: {e}")
            # Return False (off line) for both in case of error to be safe
            return False, False

    def check_for_station_color(self, frame, target_idx):
        """
        Detects station color markers in camera frame.
        Returns: (detected_flag, display_frame, mask_frame)
        detected_flag: True if color threshold met and cooldown passed.
        display_frame: Original frame with overlays (text, highlights).
        mask_frame: Binary mask of the detected color.
        """
        detected_flag = False
        display_frame = frame.copy() # Work on a copy for drawing overlays
        mask_frame = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8) # Empty mask initially

        # Add current state info to the display frame always
        cv2.putText(display_frame, f"State: {self.state.name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Check if the target index is valid and defined in our color map
        if target_idx not in STATION_COLORS_HSV:
            cv2.putText(display_frame, f"Target: Invalid ({target_idx})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return False, display_frame, mask_frame # Return immediately if target invalid

        color_info = STATION_COLORS_HSV[target_idx]
        target_name = color_info['name']
        target_bgr = color_info['color_bgr']

        try:
            # Convert to HSV color space
            hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Create mask for the target color range
            lower_bound = np.array(color_info["hsv_lower"])
            upper_bound = np.array(color_info["hsv_upper"])
            color_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
            mask_frame = color_mask # Assign the calculated mask

            # Optional: Apply morphological operations to reduce noise
            # kernel = np.ones((5,5),np.uint8)
            # color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
            # color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

            # Count non-zero pixels in the mask
            detected_pixels = cv2.countNonZero(color_mask)

            # --- Visualization ---
            # Add text showing target color and detected pixel count
            text = f"Target: {target_name} ({detected_pixels} px)"
            cv2.putText(display_frame, text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, target_bgr, 2)

            # Highlight detected areas on the display frame (optional)
            # display_frame[mask_frame > 0] = target_bgr # Simple overlay, might obscure details

            # --- Detection Logic ---
            current_time = time.time()
            # Check if pixel count exceeds threshold and cooldown period has passed
            if detected_pixels > COLOR_DETECTION_THRESHOLD and \
               (current_time - self.last_color_detection_times.get(target_idx, 0.0) > COLOR_COOLDOWN_SEC):
                self.last_color_detection_times[target_idx] = current_time
                detected_flag = True
                # Add a visual confirmation on the frame when detected
                cv2.putText(display_frame, "DETECTED!", (frame.shape[1] // 2 - 50, frame.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                self.get_logger().info(f"Color detected for station {target_idx} ({target_name})") # Log detection

            return detected_flag, display_frame, mask_frame

        except cv2.error as cv2_e:
             self.get_logger().error(f"OpenCV error during color detection: {cv2_e}")
             # Return the frame with basic state info, and empty mask
             return False, display_frame, np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        except Exception as e:
            self.get_logger().error(f"Unexpected error in color detection: {e}", exc_info=True)
            # Return the frame with basic state info, and empty mask
            return False, display_frame, np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)


    def play_sound(self, notes):
        """Plays a sequence of notes through the robot's speaker."""
        if not self.audio_publisher:
            self.get_logger().warning("Audio publisher not available, cannot play sound.")
            return
        note_msg = AudioNoteVector()
        # note_msg.append = False # Play immediately
        for frequency, duration_ms in notes:
            note = AudioNote()
            note.frequency = frequency
            # Duration expects seconds and nanoseconds
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

    def control_loop(self):
        """Main state machine and control logic for the robot."""
        # --- Check for fatal errors first ---
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.GPIO_ERROR]:
            # Already in a fatal error state, do nothing but keep ROS node alive
            # Log periodically that we are in an error state
            # if int(time.time()) % 10 == 0: # Log every 10 seconds
            #    self.get_logger().error(f"Robot halted in fatal state: {self.state.name}. Manual intervention required.")
            self.stop_moving() # Ensure stopped
            return

        display_frame = None
        mask_frame = None
        color_detected = False

        # --- Camera Capture and Processing (Always run if camera is available) ---
        if self.picam2 and self.state != RobotState.CAMERA_ERROR:
            try:
                # Capture frame (already rotated by picamera2 transform)
                raw_frame = self.picam2.capture_array()

                # Check for color if we have a valid target station
                if self.target_station_index != -1:
                    color_detected, display_frame, mask_frame = self.check_for_station_color(raw_frame, self.target_station_index)
                else:
                    # If no target, just create a basic display frame with state
                    display_frame = raw_frame.copy()
                    cv2.putText(display_frame, f"State: {self.state.name}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_frame, "Target: None", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


            except Exception as e:
                self.get_logger().error(f"Camera error in control loop: {e}", exc_info=True)
                self.state = RobotState.CAMERA_ERROR
                self.stop_moving()
                # Don't process states if camera failed this cycle
                return
        else:
            # If camera is not available, log it, can't do visual detection
             if self.state != RobotState.CAMERA_ERROR: # Avoid spamming log if already in error
                 self.get_logger().warning("Camera not available, cannot perform visual checks.")
            # Allow state machine to continue if possible without camera? Risky.
            # For now, let's assume camera is essential for station detection.
            # If camera init failed, state should already be CAMERA_ERROR.


        # --- State Machine Logic ---
        try:
            # State Transition based on Color Detection (check *before* state logic)
            # Only transition if we are actively moving towards a station
            if color_detected and self.state == RobotState.MOVING_TO_STATION:
                self.get_logger().info(f"Color marker detected for station {self.target_station_index}. Arriving.")
                self.play_sound([(523, 100), (659, 150)]) # Arrival sound
                self.stop_moving()
                self.state = RobotState.ARRIVED_AT_STATION
                # Skip remaining state logic for this iteration as we just transitioned
                # The display frame from this cycle will be shown in finally block

            # --- Handle States ---
            elif self.state == RobotState.IDLE:
                self.get_logger().info("State: IDLE - Ready for new order.")
                # Stop moving if somehow active
                self.stop_moving()
                # Reset sequence/target info
                self.current_order = None
                self.station_sequence = []
                self.current_sequence_index = -1
                self.target_station_index = -1
                # Try to fetch an order
                self.current_order = self.fetch_order_from_airtable()
                if self.current_order:
                    # Check if fetch caused an Airtable error
                    if self.state == RobotState.AIRTABLE_ERROR:
                         self.get_logger().error("Airtable error occurred during fetch. Halting.")
                         # Stay in AIRTABLE_ERROR state
                    else:
                        self.get_logger().info(f"Order '{self.current_order['order_name']}' received. Planning route...")
                        self.state = RobotState.PLANNING_ROUTE
                else:
                    # Check if fetch failed due to Airtable error or just no orders
                    if self.state != RobotState.AIRTABLE_ERROR:
                        self.get_logger().info("No pending orders found. Entering ALL_ORDERS_COMPLETE state.")
                        self.state = RobotState.ALL_ORDERS_COMPLETE # Go to wait state

            elif self.state == RobotState.PLANNING_ROUTE:
                self.get_logger().info("State: PLANNING_ROUTE")
                if not self.current_order:
                    self.get_logger().error("PLANNING_ROUTE: No current order available! Returning to IDLE.")
                    self.state = RobotState.IDLE
                    return # Skip rest of loop

                self.station_sequence = [] # Reset sequence for planning
                order_status = self.current_order["station_status"]

                # --- Determine Sequence ---
                # 1. Add Cooking Station 1 if status is WAITING
                if order_status.get(AIRTABLE_COOKING_1_STATUS_FIELD) == STATUS_WAITING:
                    self.station_sequence.append(STATION_FIELD_TO_INDEX[AIRTABLE_COOKING_1_STATUS_FIELD])
                else:
                    # If the first mandatory step isn't waiting, the order shouldn't have been fetched. Log error.
                    self.get_logger().error(f"Order '{self.current_order['order_name']}' fetched, but '{AIRTABLE_COOKING_1_STATUS_FIELD}' is not WAITING ({order_status.get(AIRTABLE_COOKING_1_STATUS_FIELD)}). Aborting order.")
                    self.state = RobotState.IDLE # Go back to fetch a valid order
                    self.current_order = None
                    return

                # 2. Add Cooking Station 2 if status is WAITING
                if order_status.get(AIRTABLE_COOKING_2_STATUS_FIELD) == STATUS_WAITING:
                    idx2 = STATION_FIELD_TO_INDEX[AIRTABLE_COOKING_2_STATUS_FIELD]
                    self.station_sequence.append(idx2)
                    if idx2 not in STATION_COLORS_HSV:
                         self.get_logger().warning(f"Station {idx2} ({AIRTABLE_COOKING_2_STATUS_FIELD}) added to route, but has no color defined in STATION_COLORS_HSV. Visual stop will not work.")

                # 3. Add Toppings if status is WAITING
                topping_fields = [
                    AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD,
                    AIRTABLE_WHIPPED_CREAM_STATUS_FIELD,
                    AIRTABLE_SPRINKLES_STATUS_FIELD
                ]
                for field in topping_fields:
                    if order_status.get(field) == STATUS_WAITING:
                        if field in STATION_FIELD_TO_INDEX:
                            idx = STATION_FIELD_TO_INDEX[field]
                            if idx in STATION_COLORS_HSV:
                                self.station_sequence.append(idx)
                            else:
                                self.get_logger().warning(f"Station {idx} ({field}) required but has no color defined in STATION_COLORS_HSV. Adding to route, but visual stop will not work.")
                        else:
                             self.get_logger().warning(f"Topping field '{field}' required but not found in STATION_FIELD_TO_INDEX mapping. Skipping.")

                # 4. Always add Pickup Station (Index 0) at the end
                pickup_idx = STATION_FIELD_TO_INDEX[AIRTABLE_PICKUP_STATUS_FIELD]
                if pickup_idx in STATION_COLORS_HSV:
                    self.station_sequence.append(pickup_idx)
                else:
                     self.get_logger().error(f"Pickup station {pickup_idx} ({AIRTABLE_PICKUP_STATUS_FIELD}) has no color defined! Cannot visually detect final stop.")
                     # Decide if this is fatal or if we proceed without visual stop for pickup
                     self.station_sequence.append(pickup_idx) # Add it anyway, rely on external stop?

                # --- Finalize Planning ---
                if not self.station_sequence:
                    self.get_logger().error("Route planning resulted in an empty sequence! Aborting order.")
                    self.state = RobotState.IDLE
                    self.current_order = None
                else:
                    self.current_sequence_index = 0 # Start at the first station in the list
                    self.target_station_index = self.station_sequence[self.current_sequence_index]
                    self.get_logger().info(f"Route planned: {self.station_sequence}. Next target: Station {self.target_station_index}")
                    self.state = RobotState.MOVING_TO_STATION
                    self.play_sound([(440,100), (550,100), (660, 100)]) # Planning complete sound


            elif self.state == RobotState.MOVING_TO_STATION:
                # self.get_logger().debug(f"State: MOVING_TO_STATION (Target: {self.target_station_index})") # Very verbose
                # Color detection check happens *before* this state logic now.
                # This state's primary job is line following.

                # Ensure target index is valid (should be set by PLANNING_ROUTE or WAITING state)
                if self.target_station_index == -1 or self.current_sequence_index >= len(self.station_sequence):
                     self.get_logger().error(f"MOVING_TO_STATION: Invalid target ({self.target_station_index}) or sequence index ({self.current_sequence_index}). Stopping.")
                     self.stop_moving()
                     self.state = RobotState.ERROR
                     return

                # Line following logic using IR sensors
                left_on, right_on = self.read_ir_sensors()

                if left_on and right_on:  # Both sensors on line (ideal) - Go straight
                    self.move_robot(BASE_DRIVE_SPEED, 0.0)
                elif left_on and not right_on: # Only left is on - Turn Left (Negative angular Z)
                    self.move_robot(BASE_DRIVE_SPEED * TURN_FACTOR, -BASE_ROTATE_SPEED)
                elif not left_on and right_on: # Only right is on - Turn Right (Positive angular Z)
                    self.move_robot(BASE_DRIVE_SPEED * TURN_FACTOR, BASE_ROTATE_SPEED)
                else:  # Both sensors off line - Lost Line
                    # Simple search: Rotate in place. Alternate direction periodically.
                    # self.get_logger().warning("Line lost! Searching...")
                    current_time = time.time()
                    # Rotate right for ~1 sec, then left for ~1 sec, repeat
                    if int(current_time) % 2 == 0:
                        self.move_robot(0.0, LOST_LINE_ROTATE_SPEED) # Turn right
                    else:
                        self.move_robot(0.0, -LOST_LINE_ROTATE_SPEED) # Turn left

            elif self.state == RobotState.ARRIVED_AT_STATION:
                self.get_logger().info(f"State: ARRIVED_AT_STATION ({self.target_station_index})")
                # Robot should already be stopped by the color detection transition logic
                self.stop_moving() # Ensure stopped

                # Verify sequence index is valid
                if self.current_sequence_index < 0 or self.current_sequence_index >= len(self.station_sequence):
                    self.get_logger().error(f"ARRIVED_AT_STATION: Invalid sequence index {self.current_sequence_index}. Stopping.")
                    self.state = RobotState.ERROR
                    return

                current_station_idx = self.station_sequence[self.current_sequence_index]

                # Get the corresponding Airtable field name
                if current_station_idx not in STATION_INDEX_TO_FIELD:
                    self.get_logger().error(f"ARRIVED_AT_STATION: No Airtable field mapping found for station index {current_station_idx}. Stopping.")
                    self.state = RobotState.ERROR
                    return

                station_field = STATION_INDEX_TO_FIELD[current_station_idx]
                self.get_logger().info(f"At station {current_station_idx} ({station_field}). Updating Airtable status to ARRIVED ({STATUS_ARRIVED}).")

                # Update Airtable status to ARRIVED
                if self.update_station_status(self.current_order["record_id"], station_field, STATUS_ARRIVED):
                    self.wait_start_time = time.time() # Record time when waiting starts
                    self.state = RobotState.WAITING_FOR_STATION_COMPLETION
                    self.get_logger().info(f"Status updated. Now WAITING_FOR_STATION_COMPLETION for {station_field}.")
                else:
                    # update_station_status already set state to AIRTABLE_ERROR
                    self.get_logger().error(f"Failed to update Airtable status for {station_field}. Entering AIRTABLE_ERROR state.")
                    # Stay in AIRTABLE_ERROR state

            elif self.state == RobotState.WAITING_FOR_STATION_COMPLETION:
                # Check for timeout first
                elapsed_wait_time = time.time() - self.wait_start_time
                if elapsed_wait_time > STATION_WAIT_TIMEOUT_SEC:
                    self.get_logger().warning(f"State: WAITING_FOR_STATION_COMPLETION - Timed out after {elapsed_wait_time:.1f}s waiting for station {self.target_station_index}. Moving to STATION_TIMED_OUT state.")
                    self.play_sound([(330, 500), (220, 500)]) # Timeout sound
                    self.state = RobotState.STATION_TIMED_OUT
                    return # Skip further checks this cycle

                # Check Airtable status periodically (e.g., every ~2 seconds)
                # Avoid spamming the API by checking less frequently than the control loop rate
                if int(time.time()) % 2 == 0: # Check roughly every 2 seconds
                    # Verify sequence index is valid
                    if self.current_sequence_index < 0 or self.current_sequence_index >= len(self.station_sequence):
                        self.get_logger().error(f"WAITING_FOR_STATION: Invalid sequence index {self.current_sequence_index}. Stopping.")
                        self.state = RobotState.ERROR
                        return

                    current_station_idx = self.station_sequence[self.current_sequence_index]

                    if current_station_idx not in STATION_INDEX_TO_FIELD:
                        self.get_logger().error(f"WAITING_FOR_STATION: No Airtable field mapping for index {current_station_idx}. Stopping.")
                        self.state = RobotState.ERROR
                        return

                    station_field = STATION_INDEX_TO_FIELD[current_station_idx]
                    # self.get_logger().debug(f"Checking Airtable if {station_field} is DONE...") # Verbose

                    if self.wait_for_station_completion(self.current_order["record_id"], station_field):
                        self.get_logger().info(f"Station {current_station_idx} ({station_field}) reported DONE ({STATUS_DONE}).")
                        self.play_sound([(659, 150), (784, 200)]) # Station complete sound

                        # Move to the next station in the sequence
                        self.current_sequence_index += 1

                        if self.current_sequence_index >= len(self.station_sequence):
                            # We have completed the last station in the sequence
                            self.get_logger().info("Completed all stations in the sequence.")
                            # Check if the last station was indeed the pickup station
                            if current_station_idx == STATION_FIELD_TO_INDEX.get(AIRTABLE_PICKUP_STATUS_FIELD):
                                self.state = RobotState.ORDER_COMPLETE
                            else:
                                # This might indicate a planning logic error if pickup wasn't last
                                self.get_logger().warning(f"Sequence finished, but last station was {current_station_idx}, not the expected pickup station ({STATION_FIELD_TO_INDEX.get(AIRTABLE_PICKUP_STATUS_FIELD)}). Treating as ORDER_COMPLETE.")
                                self.state = RobotState.ORDER_COMPLETE
                        else:
                            # More stations to visit
                            self.target_station_index = self.station_sequence[self.current_sequence_index] # Update target for next move
                            self.get_logger().info(f"Moving to next station: {self.target_station_index} ({STATION_INDEX_TO_FIELD.get(self.target_station_index, 'Unknown')})")
                            self.state = RobotState.MOVING_TO_STATION
                    # else: # Status is not DONE yet, or check failed
                        # Stay in WAITING_FOR_STATION_COMPLETION state (implicit)
                        # self.get_logger().debug(f"Still waiting for {station_field}...") # Verbose
                else:
                    # Not time to check Airtable yet, just wait.
                    pass


            elif self.state == RobotState.ORDER_COMPLETE:
                self.get_logger().info(f"State: ORDER_COMPLETE for '{self.current_order['order_name']}'")
                self.play_sound([(784, 150), (880, 150), (1047, 250)])  # Order success sound
                self.stop_moving()
                # Reset for the next order
                self.current_order = None
                self.station_sequence = []
                self.current_sequence_index = -1
                self.target_station_index = -1
                # Go back to IDLE to fetch the next order immediately
                self.state = RobotState.IDLE
                self.get_logger().info("Order finished. Returning to IDLE state.")

            elif self.state == RobotState.ALL_ORDERS_COMPLETE:
                # This state is reached when fetch_order finds no pending orders
                self.get_logger().info("State: ALL_ORDERS_COMPLETE - No orders found. Waiting...")
                self.play_sound([(440, 200), (440, 200)])  # Idle / Waiting sound
                self.stop_moving()
                # Wait for a longer duration before checking Airtable again
                # This check happens implicitly when returning to IDLE state
                time.sleep(5.0) # Pause execution briefly
                self.state = RobotState.IDLE # Go back to IDLE to check for orders again

            elif self.state == RobotState.STATION_TIMED_OUT:
                self.get_logger().error("State: STATION_TIMED_OUT - Station did not complete in time.")
                self.stop_moving()
                # --- Decision needed: What to do on timeout? ---
                # Option 1: Abort the current order and go back to IDLE
                # Option 2: Skip this station and try the next? (Complex)
                # Option 3: Enter a permanent ERROR state requiring intervention
                # Implementing Option 1 for now:
                self.get_logger().error(f"Aborting order '{self.current_order['order_name']}' due to station timeout.")
                self.current_order = None
                self.station_sequence = []
                self.current_sequence_index = -1
                self.target_station_index = -1
                self.state = RobotState.IDLE # Go back to fetch next order

            elif self.state == RobotState.AIRTABLE_ERROR:
                 self.get_logger().error("State: AIRTABLE_ERROR - Communication issue with Airtable. Halting.")
                 self.play_sound([(330, 300), (330, 300), (330, 300)]) # Airtable error sound
                 self.stop_moving()
                 # Remain in this state until resolved or restarted

            # Other error states (CAMERA_ERROR, GPIO_ERROR, ERROR) are handled at the top

        except Exception as e:
            self.get_logger().error(f"Unhandled exception in state machine logic: {e}", exc_info=True)
            self.state = RobotState.ERROR
            self.stop_moving()


        # --- Display Update (Always happens if debug_windows enabled) ---
        finally:
            if self.debug_windows:
                try:
                    if display_frame is not None:
                        cv2.imshow("Camera Feed", display_frame)
                    else:
                         # If frame is None, maybe show a black screen?
                         # black_screen = np.zeros((CAMERA_RESOLUTION[1], CAMERA_RESOLUTION[0], 3), dtype=np.uint8)
                         # cv2.putText(black_screen, f"State: {self.state.name}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                         # cv2.imshow("Camera Feed", black_screen)
                         pass # Or just don't update if no frame

                    if mask_frame is not None:
                        cv2.imshow("Color Detection Mask", mask_frame)

                    # Crucial: Call waitKey *once* per loop iteration for OpenCV windows
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'): # Allow quitting by pressing 'q' in the OpenCV window
                        self.get_logger().info("Quit key 'q' pressed. Initiating shutdown.")
                        self.state = RobotState.ERROR # Force into error state to stop activity
                        self.stop_moving()
                        # This doesn't directly stop the ROS node, requires Ctrl+C or main loop termination
                        # Consider using rclpy.shutdown() here, but it might be cleaner to let main handle it
                        # rclpy.request_shutdown() # Request shutdown

                except Exception as display_e:
                    self.get_logger().error(f"Error updating OpenCV windows: {display_e}")
                    # Disable windows if they cause persistent errors?
                    # self.debug_windows = False
                    # cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = None # Initialize node to None
    executor = None
    try:
        node = PancakeRobotNode()
        # Check if node initialization failed critically
        if node.state in [RobotState.GPIO_ERROR, RobotState.CAMERA_ERROR]:
            node.get_logger().fatal(f"Node initialization failed in state {node.state.name}. Shutting down.")
            # No need to spin if init failed
        else:
            executor = SingleThreadedExecutor()
            executor.add_node(node)
            node.get_logger().info("Starting ROS2 executor spin...")
            executor.spin()

    except KeyboardInterrupt:
        if node:
            node.get_logger().info("KeyboardInterrupt received. Shutting down.")
        else:
            print("KeyboardInterrupt received during node initialization. Shutting down.")
        pass # Allow finally block to run for cleanup
    except Exception as e:
        if node:
            node.get_logger().fatal(f"Exception during executor spin or node init: {e}", exc_info=True)
        else:
            print(f"FATAL Exception during node initialization: {e}")
    finally:
        if node:
            node.get_logger().info("Initiating final cleanup...")
            # Ensure robot is stopped
            node.stop_moving()

            # Shutdown the executor if it exists
            if executor:
                executor.shutdown()

            # Cleanup GPIO
            node.cleanup_gpio()

            # Cleanup Camera
            if node.picam2:
                try:
                    node.get_logger().info("Stopping camera...")
                    node.picam2.stop()
                    # node.picam2.close() # Close might be needed depending on version
                except Exception as cam_e:
                    node.get_logger().error(f"Error stopping camera: {cam_e}")

            # Cleanup OpenCV windows if they were used
            if node.debug_windows:
                node.get_logger().info("Closing OpenCV windows...")
                cv2.destroyAllWindows()

            # Destroy the node explicitly
            node.get_logger().info("Destroying node...")
            node.destroy_node()
        else:
            print("Node object not fully created, skipping node-specific cleanup.")

        # Shutdown rclpy
        if rclpy.ok():
             print("Shutting down rclpy...")
             rclpy.shutdown()
        print("Shutdown complete.")

if __name__ == '__main__':
    main()