#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

import os
from enum import Enum, auto
import math
import cv2  # OpenCV for image processing
import numpy as np
from picamera2 import Picamera2  # Pi Camera library
from libcamera import controls  # For camera controls like autofocus
import RPi.GPIO as GPIO  # For IR Sensors
from geometry_msgs.msg import Twist  # For direct velocity control
import time
import sys

# iRobot Create 3 specific messages (Keep Actions for potential fine-tuning/docking later)
from irobot_create_msgs.action import DriveDistance, RotateAngle
from builtin_interfaces.msg import Duration
from irobot_create_msgs.msg import AudioNoteVector, AudioNote

# Import AirtablePancake for database operations
# Assumes AirtablePancake.py is in the same directory or python path
# And it handles initialization (API key, base, table) internally, likely via .env
try:
    from AirtablePancake import AirtablePancake
except ImportError:
    print("ERROR: Failed to import AirtablePancake. Make sure AirtablePancake.py is accessible.")
    sys.exit(1)


# --- Configuration Constants ---

# Station Status Fields (same as in Airtable) - Must match AirtablePancake expectations if used internally
AIRTABLE_ORDER_NAME_COLUMN = "Order Name" # Potentially unused if AirtablePancake handles naming
AIRTABLE_CREATED_TIME_FIELD = "Created"   # Potentially unused
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

# --- Map Airtable Fields and Order Requirements to Station Indices ---
# Maps the Airtable *Status* Field Name to the *Logical* Station Index it represents
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

        # Initialize AirtablePancake instance
        try:
            # AirtablePancake should handle its own setup (API key etc)
            self.at = AirtablePancake()
            self.get_logger().info("AirtablePancake handler initialized.")
            # Optional: Add a check here if AirtablePancake provides a status method
            # if not self.at.is_configured():
            #    raise ConnectionError("AirtablePancake handler not configured properly.")
        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to initialize AirtablePancake: {e}")
            self.state = RobotState.AIRTABLE_ERROR
            return # Stop initialization

        # Robot State Initialization
        self.state = RobotState.IDLE
        self.current_order = None # May store record ID or full order data if needed
        self.current_order_record_id = None # Specific variable for the ID
        self.current_order_statuses = {} # Cache statuses locally
        self.station_sequence = [] # Logical indices
        self.current_sequence_index = 0
        self.target_station_index = -1 # Logical index
        self.pancakes_made_count = 0
        self.last_color_detection_times = { idx: 0.0 for idx in STATION_COLORS_HSV.keys() }
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
        self.control_timer_period = 0.05
        self.control_timer = self.create_timer(self.control_timer_period, self.control_loop)
        self.airtable_poll_timer = None

        self.get_logger().info("Pancake Robot Node Initialized and Ready.")
        self.play_sound([(440, 200), (550, 300)])

    # --- Movement Control (Same as test4) ---
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

    # --- Sensor Reading (Same as test4) ---
    def read_ir_sensors(self):
        try:
            left_val = GPIO.input(LEFT_IR_PIN)
            right_val = GPIO.input(RIGHT_IR_PIN)
            return left_val, right_val
        except Exception as e:
            self.get_logger().error(f"Error reading GPIO IR sensors: {e}")
            return GPIO.HIGH, GPIO.HIGH

    # --- Color Detection (Same as test4) ---
    def check_for_station_color(self, frame, physical_target_idx):
        if physical_target_idx not in STATION_COLORS_HSV:
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

    # --- Airtable Communication (Using AirtablePancake instance 'self.at') ---
    def fetch_order_from_airtable(self):
        """Fetches the next order using AirtablePancake."""
        self.get_logger().info("Attempting to fetch next order via AirtablePancake...")
        try:
            # We need a way to get the *oldest* record where Cook1 or Cook2 is 0 and Pickup is 0.
            # AirtablePancake might need a method like `findOldestPendingOrder()`
            # or we query individual fields and assume a single record context.

            # --- Strategy 1: Check a primary field (e.g., Pickup) to see if *any* order is active ---
            # This assumes AirtablePancake operates on a single "current order" concept defined externally or by oldest entry.
            pickup_status = self.at.checkValue(AIRTABLE_PICKUP_STATUS_FIELD)

            if pickup_status is None:
                 self.get_logger().error("Airtable checkValue failed for Pickup Status during fetch.")
                 self.state = RobotState.AIRTABLE_ERROR
                 return None # Airtable error

            if pickup_status == STATUS_WAITING:
                # Assume an order is active (or the first one is ready)
                # Now fetch all relevant statuses for this implicit order
                order_statuses = {}
                all_fields_valid = True
                for field in STATION_FIELD_TO_INDEX.keys():
                    status = self.at.checkValue(field)
                    if status is None:
                        self.get_logger().error(f"Airtable checkValue failed for {field} during fetch.")
                        all_fields_valid = False
                        break # Stop fetching if one fails
                    order_statuses[field] = status

                if not all_fields_valid:
                    self.state = RobotState.AIRTABLE_ERROR
                    return None

                # We don't have record_id or order_name easily here unless at.checkValue returns more info or at provides it.
                # Let's assume AirtablePancake internally knows the record_id it's working on.
                # If not, this approach needs rethinking based on AirtablePancake's capabilities.
                self.current_order_statuses = order_statuses
                # Simulate getting a record ID if the class doesn't provide one easily
                self.current_order_record_id = self.at.get_current_record_id() # ASSUMING this method exists
                if not self.current_order_record_id:
                     self.get_logger().warning("Could not get record ID from AirtablePancake, proceeding without it for now.")
                     # Functionality requiring record ID might fail later.

                self.get_logger().info(f"Fetched statuses for assumed current order (Record ID: {self.current_order_record_id or 'Unknown'}).")
                self.get_logger().debug(f"Order statuses: {self.current_order_statuses}")
                # Create a compatible structure if needed by planning logic
                self.current_order = {"record_id": self.current_order_record_id, "station_status": self.current_order_statuses} # Basic structure
                return self.current_order

            elif pickup_status == STATUS_DONE:
                self.get_logger().info("Pickup status is DONE (99), assuming no pending orders.")
                return None # No orders pending
            else:
                # Pickup is ARRIVED or some other state - implies an order is in progress? Or error?
                self.get_logger().warn(f"Pickup status is {pickup_status}, uncertain if new orders exist. Assuming none for now.")
                return None

        except Exception as e:
            self.get_logger().error(f"Error fetching order/statuses via AirtablePancake: {e}")
            self.state = RobotState.AIRTABLE_ERROR
            return None

    def update_station_status_in_airtable(self, station_field_name, new_status_code):
        """Updates status using AirtablePancake."""
        try:
            self.get_logger().info(f"Updating Airtable via handler: Field '{station_field_name}' to {new_status_code}")
            # Assumes changeValue operates on the internally managed record ID
            success = self.at.changeValue(station_field_name, new_status_code)
            if success:
                 self.get_logger().info(f"Airtable update via handler successful for {station_field_name}.")
                 # Update local cache
                 if station_field_name in self.current_order_statuses:
                     self.current_order_statuses[station_field_name] = new_status_code
                 return True
            else:
                 self.get_logger().error(f"Airtable update via handler failed for {station_field_name}.")
                 self.state = RobotState.AIRTABLE_ERROR # Assume handler didn't set state
                 return False
        except Exception as e:
            self.get_logger().error(f"Error calling AirtablePancake changeValue for {station_field_name}: {e}")
            self.state = RobotState.AIRTABLE_ERROR
            return False

    def check_station_status_in_airtable(self, station_field_name):
        """Checks status using AirtablePancake."""
        try:
            # Assumes checkValue operates on the internally managed record ID
            status = self.at.checkValue(station_field_name)
            if status is None:
                self.get_logger().error(f"Airtable check via handler failed for {station_field_name}.")
                self.state = RobotState.AIRTABLE_ERROR
                return None
            else:
                #self.get_logger().debug(f"Airtable check via handler successful for {station_field_name}: Status {status}")
                return status
        except Exception as e:
            self.get_logger().error(f"Error calling AirtablePancake checkValue for {station_field_name}: {e}")
            self.state = RobotState.AIRTABLE_ERROR
            return None

    # --- Sound Utility (Same as test4) ---
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
        self.stop_airtable_polling()
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

        # Poll the field corresponding to the *logical* target index
        target_field = STATION_INDEX_TO_FIELD.get(self.target_station_index)
        if not target_field:
            self.get_logger().error(f"Polling Error: Invalid target station index {self.target_station_index}.")
            self.state = RobotState.ERROR
            self.stop_airtable_polling()
            self.stop_moving()
            return

        self.get_logger().debug(f"Polling Airtable via handler: Logical Station {self.target_station_index} ({target_field})")
        current_status = self.check_station_status_in_airtable(target_field) # Use wrapper

        if current_status is None:
            # Error logged by check_station_status_in_airtable, state should be set
            self.get_logger().error(f"Airtable check via handler failed during poll for {target_field}. Stopping polling.")
            self.stop_airtable_polling()
            self.stop_moving()
            return

        self.get_logger().debug(f"Current status via handler for {target_field}: {current_status}")

        if current_status == STATUS_DONE: # 99
            self.get_logger().info(f"Logical Station {self.target_station_index} ({target_field}) reported DONE (99) via handler!")
            self.play_sound([(600, 100), (800, 150)])
            self.stop_airtable_polling()
            # Let the main loop handle the transition

        elif (time.time() - self.wait_start_time) > STATION_WAIT_TIMEOUT_SEC:
             self.get_logger().error(f"TIMEOUT waiting for logical station {self.target_station_index} ({target_field}) to reach status {STATUS_DONE}.")
             self.state = RobotState.STATION_TIMED_OUT
             self.stop_airtable_polling()
             self.stop_moving()

    # --- Main Control Loop (State Machine Logic) ---
    def control_loop(self):
        # --- Terminal/Error State Handling (Same as test4) ---
        if self.state in [
                RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.GPIO_ERROR,
                RobotState.AIRTABLE_ERROR, RobotState.ALL_ORDERS_COMPLETE, RobotState.STATION_TIMED_OUT]:
            if self.state != RobotState.ALL_ORDERS_COMPLETE:
                self.get_logger().error(f"Robot in terminal/error state: {self.state.name}. Halting operations.", throttle_duration_sec=5)
            self.stop_moving()
            self.stop_airtable_polling()
            return

        # --- Sensor Reading and Processing (Same as test4) ---
        left_ir, right_ir = self.read_ir_sensors()
        frame = self.picam2.capture_array()
        if CAMERA_ROTATION is not None:
            frame = cv2.rotate(frame, CAMERA_ROTATION)

        # --- State Machine Logic ---
        current_state = self.state
        next_state = current_state

        if current_state == RobotState.IDLE:
            self.get_logger().info("State: IDLE. Checking for new orders.")
            self.current_order = None
            self.current_order_record_id = None
            self.current_order_statuses = {}
            self.station_sequence = []
            self.current_sequence_index = 0
            self.target_station_index = -1
            next_state = RobotState.FETCHING_ORDER

        elif current_state == RobotState.FETCHING_ORDER:
            fetched_order = self.fetch_order_from_airtable() # Uses wrapper
            if fetched_order:
                # Order data (statuses, maybe ID) stored in self.current_order... by the fetch method
                self.get_logger().info(f"Processing Order (Record ID: {self.current_order_record_id or 'Unknown'})")
                next_state = RobotState.PLANNING_ROUTE
            elif self.state == RobotState.AIRTABLE_ERROR:
                pass # Error handled by fetch method setting state
            else: # No orders found
                self.get_logger().info("No more pending orders found.")
                if self.pancakes_made_count > 0:
                    self.play_sound([(600, 100), (700, 100), (800, 300)])
                    self.get_logger().info(f"Completed {self.pancakes_made_count} order(s) this run.")
                else:
                    self.play_sound([(400, 500)])
                next_state = RobotState.ALL_ORDERS_COMPLETE

        elif current_state == RobotState.PLANNING_ROUTE:
            if not self.current_order_statuses: # Check if statuses were fetched
                self.get_logger().error("Planning route error: No current order status data fetched.")
                next_state = RobotState.ERROR
            else:
                self.get_logger().info("Planning station route for current order...")
                self.station_sequence = []
                order_preference = [1, 2, 3, 4, 5, 0] # Logical indices

                for station_idx in order_preference:
                    station_field = STATION_INDEX_TO_FIELD.get(station_idx)
                    # Use fetched statuses from self.current_order_statuses
                    if station_field and self.current_order_statuses.get(station_field, -1) == STATUS_WAITING:
                        physical_idx = PHYSICAL_COOKING_STATION_INDEX if station_idx == 2 else station_idx
                        if physical_idx in STATION_COLORS_HSV or station_idx == 2:
                            self.station_sequence.append(station_idx)
                            station_name = STATION_COLORS_HSV.get(physical_idx, {}).get('name', f'Logical Step {station_idx}')
                            self.get_logger().info(f" - Adding Logical Step {station_idx} ({station_name}, Field: {station_field})")
                        else:
                             self.get_logger().warn(f"Skipping station with field {station_field} - Physical Index {physical_idx} not defined in STATION_COLORS_HSV")

                self.get_logger().info(f"Planned route (logical indices): {self.station_sequence}")
                self.current_sequence_index = 0
                if not self.station_sequence:
                    self.get_logger().error("Planning route error: No valid stations in sequence for this order.")
                    next_state = RobotState.ORDER_COMPLETE
                else:
                    self.target_station_index = self.station_sequence[self.current_sequence_index]
                    if self.target_station_index == 0:
                         next_state = RobotState.RETURNING_TO_PICKUP
                    else:
                         next_state = RobotState.MOVING_TO_STATION

        # States MOVING_TO_STATION, RETURNING_TO_PICKUP are identical to test4

        elif current_state == RobotState.MOVING_TO_STATION or current_state == RobotState.RETURNING_TO_PICKUP:
            physical_target_idx = -1
            if self.target_station_index == 0: physical_target_idx = 0
            elif self.target_station_index == 1: physical_target_idx = PHYSICAL_COOKING_STATION_INDEX
            elif self.target_station_index == 2: physical_target_idx = PHYSICAL_COOKING_STATION_INDEX
            elif self.target_station_index in STATION_COLORS_HSV: physical_target_idx = self.target_station_index
            else:
                self.get_logger().error(f"Movement error: Cannot determine physical station for logical target {self.target_station_index}")
                next_state = RobotState.ERROR
                self.stop_moving()

            if next_state != RobotState.ERROR:
                detected, debug_frame = self.check_for_station_color(frame, physical_target_idx)
                if detected:
                    station_name = STATION_COLORS_HSV[physical_target_idx]['name']
                    self.get_logger().info(f"Physical marker for {station_name} detected (Logical Target: {self.target_station_index})")
                    self.play_sound([(500, 150)])
                    self.stop_moving()
                    next_state = RobotState.ARRIVED_AT_STATION if current_state == RobotState.MOVING_TO_STATION else RobotState.ARRIVED_AT_PICKUP
                else:
                    linear_speed = BASE_DRIVE_SPEED
                    angular_speed = 0.0
                    if left_ir == GPIO.HIGH and right_ir == GPIO.HIGH: angular_speed = 0.0
                    elif left_ir == GPIO.LOW and right_ir == GPIO.HIGH:
                        angular_speed = BASE_ROTATE_SPEED * TURN_FACTOR; linear_speed *= 0.8
                    elif left_ir == GPIO.HIGH and right_ir == GPIO.LOW:
                        angular_speed = -BASE_ROTATE_SPEED * TURN_FACTOR; linear_speed *= 0.8
                    elif left_ir == GPIO.LOW and right_ir == GPIO.LOW:
                        self.get_logger().warn("IR: Both sensors on black. Rotating slightly.", throttle_duration_sec=2)
                        linear_speed = 0.0; angular_speed = LOST_LINE_ROTATE_SPEED
                    self.move_robot(linear_speed, angular_speed)

                if self.debug_windows and debug_frame is not None:
                    ir_text = f"IR L: {left_ir} R: {right_ir}"
                    cv2.putText(debug_frame, ir_text, (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    cv2.imshow("Robot View", debug_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.get_logger().info("Quit requested via OpenCV window."); self.state = RobotState.ERROR; rclpy.shutdown()


        elif current_state == RobotState.ARRIVED_AT_STATION:
            # Update Airtable for the *logical* station index using the wrapper method
            station_field = STATION_INDEX_TO_FIELD.get(self.target_station_index)
            if not station_field:
                self.get_logger().error(f"Arrival error: No Airtable field for logical station index {self.target_station_index}.")
                next_state = RobotState.ERROR
            else:
                physical_idx = PHYSICAL_COOKING_STATION_INDEX if self.target_station_index == 2 else self.target_station_index
                station_name = STATION_COLORS_HSV.get(physical_idx, {}).get('name', f'Logical Step {self.target_station_index}')
                self.get_logger().info(f"Arrived for logical step {self.target_station_index} ({station_name}, Field: {station_field}). Updating status to {STATUS_ARRIVED}.")

                # Use the wrapper function
                success = self.update_station_status_in_airtable(station_field, STATUS_ARRIVED)

                if success:
                    self.start_airtable_polling()
                    next_state = RobotState.WAITING_FOR_STATION_COMPLETION
                else:
                    # Error/state handling done within update_station_status_in_airtable
                    self.get_logger().error(f"Failed update via handler upon arrival for {station_field}.")
                    self.stop_moving() # Ensure stop on failure


        elif current_state == RobotState.WAITING_FOR_STATION_COMPLETION:
            # Logic relies on poll timer callback setting state or timer becoming None
            if self.airtable_poll_timer is None:
                 if self.state == RobotState.STATION_TIMED_OUT:
                     self.get_logger().error("Station completion timed out. Halting order processing.")
                     pass
                 elif self.state == RobotState.AIRTABLE_ERROR:
                      self.get_logger().error("Airtable error occurred while waiting. Halting.")
                      pass
                 elif self.state == RobotState.ERROR:
                      self.get_logger().error("Error occurred while waiting. Halting.")
                      pass
                 else: # Assume STATUS_DONE detected
                      station_field = STATION_INDEX_TO_FIELD.get(self.target_station_index, "UNKNOWN_FIELD")
                      self.get_logger().info(f"Logical Station {self.target_station_index} ({station_field}) processing complete (detected via handler).")
                      self.current_sequence_index += 1

                      if self.current_sequence_index < len(self.station_sequence):
                          self.target_station_index = self.station_sequence[self.current_sequence_index]
                          next_station_field = STATION_INDEX_TO_FIELD.get(self.target_station_index, "UNKNOWN")
                          physical_idx = PHYSICAL_COOKING_STATION_INDEX if self.target_station_index == 2 else self.target_station_index
                          next_station_name = STATION_COLORS_HSV.get(physical_idx, {}).get('name', f'Logical Step {self.target_station_index}')
                          self.get_logger().info(f"Proceeding to next logical step: {self.target_station_index} ({next_station_name}, Field: {next_station_field})")
                          next_state = RobotState.RETURNING_TO_PICKUP if self.target_station_index == 0 else RobotState.MOVING_TO_STATION
                      else: # Finished sequence
                          if self.target_station_index == 0: # Last step was pickup
                                next_state = RobotState.ORDER_COMPLETE # Should have gone via ARRIVED_AT_PICKUP
                          else:
                                self.get_logger().error("Finished sequence but last step wasn't Pickup (Index 0).")
                                next_state = RobotState.ERROR


        elif current_state == RobotState.ARRIVED_AT_PICKUP:
            self.get_logger().info(f"Arrived back at Pickup Station (Index 0). Order sequence complete.")
            self.pancakes_made_count += 1
            self.play_sound([(800, 100), (700, 100), (600, 200)])

            pickup_field = STATION_INDEX_TO_FIELD.get(0)
            if pickup_field:
                 # Use wrapper function
                success = self.update_station_status_in_airtable(pickup_field, STATUS_DONE)
                if not success:
                    self.get_logger().error(f"Failed to update final Pickup status ({pickup_field}) to DONE via handler. Continuing...")
            else:
                self.get_logger().warn("No Airtable field mapped for Pickup Station (Index 0). Cannot update final status.")

            next_state = RobotState.ORDER_COMPLETE

        elif current_state == RobotState.ORDER_COMPLETE:
            self.get_logger().info("Order cycle finished. Returning to IDLE.")
            self.current_order = None
            self.current_order_record_id = None
            self.current_order_statuses = {}
            self.station_sequence = []
            self.current_sequence_index = 0
            self.target_station_index = -1
            next_state = RobotState.IDLE

        # --- State Transition ---
        if next_state != current_state:
            self.get_logger().info(f"State transition: {current_state.name} -> {next_state.name}")
            self.state = next_state

    # --- Cleanup Methods (Similar to test4) ---
    def cleanup_gpio(self):
        self.get_logger().info("Cleaning up GPIO...")
        try:
            if GPIO.getmode() is not None: GPIO.cleanup(); self.get_logger().info("GPIO cleanup successful.")
            else: self.get_logger().info("GPIO was not set up, skipping cleanup.")
        except Exception as e: self.get_logger().error(f"Error during GPIO cleanup: {e}")

    def shutdown_camera(self):
        if hasattr(self, 'picam2') and self.picam2:
            try:
                if self.picam2.started: self.picam2.stop(); self.get_logger().info("Pi Camera stopped.")
            except Exception as e: self.get_logger().error(f"Error stopping camera: {e}")
        if self.debug_windows:
            try: cv2.destroyAllWindows(); self.get_logger().info("OpenCV windows closed.")
            except Exception as e: self.get_logger().error(f"Error closing OpenCV windows: {e}")

    def shutdown_robot(self):
        self.get_logger().info("Initiating robot shutdown sequence...")
        try: self.stop_moving()
        except Exception as e: self.get_logger().error(f"Error stopping robot movement: {e}")
        if hasattr(self, 'control_timer') and self.control_timer and not self.control_timer.is_canceled():
            self.control_timer.cancel()
        self.stop_airtable_polling()
        self.shutdown_camera()
        self.cleanup_gpio()
        self.get_logger().info("Robot shutdown sequence complete.")


# --- Main Execution Function (Similar to test4) ---
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
                 if GPIO.getmode() is not None: GPIO.cleanup(); print("GPIO cleanup attempted.")
             except Exception as e: print(f"Error during final GPIO cleanup: {e}")
        if rclpy.ok(): rclpy.shutdown()
        print("ROS2 shutdown complete.")
    return exit_code

if __name__ == '__main__':
    sys.exit(main())