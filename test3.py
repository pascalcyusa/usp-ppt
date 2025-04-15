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
from AirtablePancake 

at = AirtablePancake.at()

# --- Configuration Constants ---

# Station Status Fields (same as in Airtable)
AIRTABLE_ORDER_NAME_COLUMN = "Order Name"
AIRTABLE_CREATED_TIME_FIELD = "Created"
AIRTABLE_COOKING_1_STATUS_FIELD = "Cooking 1 Status"
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

    # Index 2: Robot 2 Waiting Position
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

        # Check Action Servers (optional if not immediately used)
        # if not self.drive_client.wait_for_server(timeout_sec=2.0):
        #     self.get_logger().warn('DriveDistance action server not available.')
        # if not self.rotate_client.wait_for_server(timeout_sec=2.0):
        #     self.get_logger().warn('RotateAngle action server not available.')

        # Timers
        # Combined timer for IR + Color + State Logic? Or separate? Let's try combined first.
        # Seconds (20 Hz) - Should be fast enough for IR + Color checks
        self.control_timer_period = 0.05
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
        # self.get_logger().debug(f"Publishing cmd_vel: Lin={linear_x:.2f}, Ang={angular_z:.2f}") # Verbose

    def stop_moving(self):
        """Sends a zero velocity Twist message to stop the robot."""
        self.get_logger().info("Sending stop command (Twist zero).")
        self.move_robot(0.0, 0.0)
        # Add a small delay to ensure the message is sent and processed
        time.sleep(0.1)

    # --- Sensor Reading ---
    def read_ir_sensors(self):
        """Reads the state of the left and right IR sensors."""
        try:
            left_val = GPIO.input(LEFT_IR_PIN)
            right_val = GPIO.input(RIGHT_IR_PIN)
            # self.get_logger().debug(f"IR Sensors: L={left_val}, R={right_val}") # Verbose
            # Assuming LOW means ON LINE (detecting black), HIGH means OFF LINE (detecting white/floor)
            # Adjust this logic based on your sensor type (active low/high)
            return left_val, right_val
        except Exception as e:
            self.get_logger().error(f"Error reading GPIO IR sensors: {e}")
            # Consider transitioning to ERROR or GPIO_ERROR state if persistent
            return GPIO.HIGH, GPIO.HIGH  # Default to off-line state on error

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

            # Create white background mask first (from test1)
            white_lower = np.array([0, 0, 200])
            white_upper = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv_image, white_lower, white_upper)

            # Create color mask excluding white background (from test1)
            color_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
            color_mask = cv2.bitwise_and(
                color_mask, cv2.bitwise_not(white_mask))

            detected_pixels = cv2.countNonZero(color_mask)

            # Create debug visualization
            debug_frame = None
            if self.debug_windows:
                debug_frame = frame.copy()

                # Add visualization (matching test1)
                detected_area = cv2.bitwise_and(frame, frame, mask=color_mask)
                debug_frame = cv2.addWeighted(
                    debug_frame, 1, detected_area, 0.5, 0)

                # Add text showing detection info
                text = f"{target_color_name}: {detected_pixels} px"
                cv2.putText(debug_frame, text, (10, 30 + 30 * target_idx),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)

                # Print BGR values if pixels detected (from test1)
                if detected_pixels > 0:
                    detected_area = cv2.bitwise_and(
                        frame, frame, mask=color_mask)
                    bgr_values = cv2.mean(detected_area)
                    self.get_logger().info(
                        f'{target_color_name} - BGR values: B:{bgr_values[0]:.1f}, G:{bgr_values[1]:.1f}, R:{bgr_values[2]:.1f} - Pixels: {detected_pixels}')

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

    # --- Airtable Communication ---
    def fetch_order_from_airtable(self):
        """Fetches the oldest order that needs processing."""
        self.get_logger().info("Attempting to fetch next order from Airtable...")

        try:
            # Check the first order with Pickup Status != 99
            pickup_status = at.checkValue(AIRTABLE_PICKUP_STATUS_FIELD)
            if pickup_status != STATUS_DONE:  # If not 99
                # Initialize order data structure
                order = {
                    "order_name": f"Order {self.pancakes_made_count + 1}",
                    "station_status": {}
                }

                # Get all station statuses
                for field in [AIRTABLE_COOKING_1_STATUS_FIELD,
                              AIRTABLE_ROBOT2_WAIT_STATUS_FIELD,
                              AIRTABLE_WHIPPED_CREAM_STATUS_FIELD,
                              AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD,
                              AIRTABLE_SPRINKLES_STATUS_FIELD,
                              AIRTABLE_PICKUP_STATUS_FIELD]:
                    status = at.checkValue(field)
                    order["station_status"][field] = status

                return order
            else:
                self.get_logger().info("No pending orders found.")
                return None

        except Exception as e:
            self.get_logger().error(f"Error fetching from Airtable: {e}")
            self.state = RobotState.AIRTABLE_ERROR
            return None

    def update_station_status_in_airtable(self, record_id, station_field_name, new_status_code):
        """Updates a specific station's numeric status field for an order."""
        try:
            at.changeValue(station_field_name, new_status_code)
            self.get_logger().info(f"Airtable update successful.")
            return True
        except Exception as e:
            self.get_logger().error(f"Error updating Airtable status: {e}")
            return False

    def check_station_status_in_airtable(self, record_id, station_field_name):
        """Checks the current numeric status of a specific station field for an order."""
        try:
            return at.checkValue(station_field_name)
        except Exception as e:
            self.get_logger().error(f"Error checking Airtable status: {e}")
            return None

    # --- Sound Utility ---
    def play_sound(self, notes):
        """Publishes a sequence of AudioNotes."""
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.GPIO_ERROR, RobotState.AIRTABLE_ERROR]:
            return

        audio_msg = AudioNoteVector()
        audio_msg.append = False  # Play immediately
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
        self.get_logger().info(
            f"Starting Airtable polling timer (every {AIRTABLE_POLL_RATE}s) for station {self.target_station_index} completion.")
        self.airtable_poll_timer = self.create_timer(
            AIRTABLE_POLL_RATE, self.airtable_poll_callback)
        self.wait_start_time = time.time()

    def stop_airtable_polling(self):
        """Cancels and destroys the Airtable polling timer."""
        if self.airtable_poll_timer is not None:
            if not self.airtable_poll_timer.is_canceled():
                self.airtable_poll_timer.cancel()
            # Cannot reliably destroy timer from within its own callback in ROS2 Foxy/Galactic?
            # Let garbage collection handle it after cancel. In Humble+, use destroy_timer.
            self.airtable_poll_timer = None
            self.get_logger().info("Airtable polling timer stopped.")

    def airtable_poll_callback(self):
        """Callback function for the Airtable polling timer."""
        if self.state != RobotState.WAITING_FOR_STATION_COMPLETION:
            self.get_logger().warn("Airtable poll callback executed unexpectedly. Stopping polling.")
            self.stop_airtable_polling()
            return

        target_field = STATION_INDEX_TO_FIELD.get(self.target_station_index)

        # Add debug logging
        self.get_logger().info(
            f"Polling Airtable for station {self.target_station_index} ({target_field})")

        # Check Airtable for the status
        current_status = self.check_station_status_in_airtable(
            self.current_order["record_id"], target_field)

        # Add debug logging
        self.get_logger().info(
            f"Current status from Airtable: {current_status}")

        if current_status == STATUS_DONE:  # This is 99
            self.get_logger().info(
                f"Station {self.target_station_index} reported DONE (99)!")
            self.play_sound([(600, 100), (800, 150)])
            self.stop_airtable_polling()

            # Move to next station
            self.current_sequence_index += 1
            if self.current_sequence_index < len(self.station_sequence):
                self.target_station_index = self.station_sequence[self.current_sequence_index]
                next_station_name = STATION_COLORS_HSV[self.target_station_index]['name']
                self.get_logger().info(
                    f"Moving to next station: {self.target_station_index} ({next_station_name})")

                # Important: Set the next state explicitly
                if self.target_station_index == 0:
                    self.state = RobotState.RETURNING_TO_PICKUP
                else:
                    self.state = RobotState.MOVING_TO_STATION

    # --- Main Control Loop (State Machine Logic) ---
    def control_loop(self):
        """The core state machine logic, incorporating sensor checks and Airtable updates."""

        # --- Terminal/Error State Handling ---
        if self.state in [
                RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.GPIO_ERROR,
                RobotState.AIRTABLE_ERROR, RobotState.ALL_ORDERS_COMPLETE, RobotState.STATION_TIMED_OUT]:
            # Ensure robot is stopped in any final/error state
            self.stop_moving()
            # Log error states periodically
            if self.state != RobotState.ALL_ORDERS_COMPLETE:
                self.get_logger().error(
                    f"Robot in terminal/error state: {self.state.name}. Halting operations.", throttle_duration_sec=5)
            # Consider canceling timers if not already done
            if self.control_timer and not self.control_timer.is_canceled():
                # self.control_timer.cancel() # Careful: cancelling the timer running this loop needs thought
                pass  # Let it run but do nothing else
            self.stop_airtable_polling()
            return  # Exit loop iteration

        # --- Sensor Reading and Processing (executed every loop) ---
        left_ir, right_ir = self.read_ir_sensors()
        frame = self.picam2.capture_array()
        if CAMERA_ROTATION is not None:
            frame = cv2.rotate(frame, CAMERA_ROTATION)

        # --- State Machine Logic ---
        current_state = self.state  # Cache state for this iteration
        # self.get_logger().debug(f"State: {current_state.name}, Target St Idx: {self.target_station_index}, Seq Idx: {self.current_sequence_index}/{len(self.station_sequence)}") # Verbose

        next_state = current_state  # Default to no state change

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
                # Error handled by fetch_order_from_airtable setting state
                pass  # Stay in AIRTABLE_ERROR
            else:
                # No orders found
                self.get_logger().info("No more pending orders found.")
                if self.pancakes_made_count > 0:
                    # Batch complete sound
                    self.play_sound([(600, 100), (700, 100), (800, 300)])
                    self.get_logger().info(
                        f"Completed {self.pancakes_made_count} order(s) this run.")
                else:
                    # No orders found initially sound
                    self.play_sound([(400, 500)])
                next_state = RobotState.ALL_ORDERS_COMPLETE

        elif current_state == RobotState.PLANNING_ROUTE:
            if not self.current_order:
                self.get_logger().error("Planning route error: No current order data.")
                next_state = RobotState.ERROR
            else:
                self.get_logger().info("Planning station route for current order...")
                self.station_sequence = []
                # Add stations based on their status fields
                for station_field, station_idx in STATION_FIELD_TO_INDEX.items():
                    if self.current_order["station_status"].get(station_field, 0) == STATUS_WAITING:
                        self.station_sequence.append(station_idx)
                        self.get_logger().info(
                            f" - Adding Station {station_idx} ({STATION_COLORS_HSV[station_idx]['name']})")

                self.get_logger().info(
                    f"Planned route (indices): {self.station_sequence}")
                self.current_sequence_index = 0  # Start at the beginning of the sequence
                if not self.station_sequence:
                    self.get_logger().error("Planning route error: No stations in sequence.")
                    next_state = RobotState.ERROR
                else:
                    self.target_station_index = self.station_sequence[self.current_sequence_index]
                    self.get_logger().info(
                        f"First target station: {self.target_station_index} ({STATION_COLORS_HSV[self.target_station_index]['name']})")
                    next_state = RobotState.MOVING_TO_STATION

        elif current_state == RobotState.MOVING_TO_STATION or current_state == RobotState.RETURNING_TO_PICKUP:
            # --- Color Check First ---
            # Only check for the *specific target* station's color
            detected, debug_frame = self.check_for_station_color(
                frame, self.target_station_index)
            if detected:
                station_name = STATION_COLORS_HSV[self.target_station_index]['name']
                self.get_logger().info(
                    f"Target color marker for {station_name} (Index {self.target_station_index}) DETECTED!")
                self.play_sound([(500, 150)])  # Arrival beep
                self.stop_moving()  # Stop immediately upon detection

                if current_state == RobotState.MOVING_TO_STATION:
                    next_state = RobotState.ARRIVED_AT_STATION
                else:  # Was RETURNING_TO_PICKUP
                    next_state = RobotState.ARRIVED_AT_PICKUP

            else:
                # --- IR Line Following Logic ---
                # (Execute if color not detected)
                linear_speed = BASE_DRIVE_SPEED
                angular_speed = 0.0

                # Logic assumes LOW = On Line, HIGH = Off Line (adjust if needed)
                if left_ir == GPIO.HIGH and right_ir == GPIO.HIGH:
                    # Both Off Line (White/Floor) - On the line
                    angular_speed = 0.0
                    # self.get_logger().debug("IR: On Line") # Verbose
                elif left_ir == GPIO.LOW and right_ir == GPIO.HIGH:
                    # Left On Line (Black), Right Off Line -> Turn Left (changed from Right)
                    angular_speed = BASE_ROTATE_SPEED * TURN_FACTOR  # Removed negative sign
                    linear_speed *= 0.8  # Slow down slightly when turning
                    # self.get_logger().debug("IR: Turn Left") # Verbose
                elif left_ir == GPIO.HIGH and right_ir == GPIO.LOW:
                    # Left Off Line, Right On Line (Black) -> Turn Right (changed from Left)
                    angular_speed = -BASE_ROTATE_SPEED * TURN_FACTOR  # Added negative sign
                    linear_speed *= 0.8  # Slow down slightly when turning
                    # self.get_logger().debug("IR: Turn Right") # Verbose
                elif left_ir == GPIO.LOW and right_ir == GPIO.LOW:
                    # Both On Line (Black) - Line Lost or Junction?
                    # Simple recovery: Stop linear, rotate slightly (changed direction)
                    self.get_logger().warn(
                        "IR: Both sensors on black (Line Lost/Junction?). Rotating slightly.")
                    linear_speed = 0.0
                    # Removed negative sign for opposite recovery rotation
                    angular_speed = LOST_LINE_ROTATE_SPEED

                self.move_robot(linear_speed, angular_speed)

            # --- Update Debug Window ---
            if self.debug_windows and debug_frame is not None:
                # Add IR sensor status text
                ir_text = f"IR L: {left_ir} R: {right_ir}"
                cv2.putText(debug_frame, ir_text, (frame.shape[1] - 150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # Show both original and HSV views
                cv2.imshow("Camera View", frame)  # Original camera feed
                # Debug view with overlays
                cv2.imshow("Robot View (with detection)", debug_frame)
                cv2.imshow("HSV View", cv2.cvtColor(
                    frame, cv2.COLOR_BGR2HSV))  # HSV conversion view

                # Move windows to convenient positions
                cv2.moveWindow("Camera View", 0, 0)
                cv2.moveWindow("Robot View (with detection)",
                               frame.shape[1], 0)
                cv2.moveWindow("HSV View", 0, frame.shape[0])

                if cv2.waitKey(1) & 0xFF == ord('q'):  # Allow quitting via window
                    self.get_logger().info("Quit requested via OpenCV window.")
                    self.state = RobotState.ERROR  # Or another state to signal shutdown
                    rclpy.shutdown()  # Request ROS shutdown

        elif current_state == RobotState.ARRIVED_AT_STATION:
            # Action: Update Airtable status to ARRIVED (1)
            station_field = STATION_INDEX_TO_FIELD.get(
                self.target_station_index)
            if not station_field:
                self.get_logger().error(
                    f"Arrival error: No Airtable field for station index {self.target_station_index}.")
                next_state = RobotState.ERROR
            else:
                self.get_logger().info(
                    f"Arrived at {STATION_COLORS_HSV[self.target_station_index]['name']} ({station_field}). Updating status to {STATUS_ARRIVED}.")
                if self.update_station_status_in_airtable(self.current_order["record_id"], station_field, STATUS_ARRIVED):
                    # Start polling timer
                    self.start_airtable_polling()
                    next_state = RobotState.WAITING_FOR_STATION_COMPLETION
                else:
                    # Update failed, error state should be set by update function
                    self.get_logger().error(
                        f"Failed to update Airtable status upon arrival at {station_field}.")
                    # State should already be AIRTABLE_ERROR

        elif current_state == RobotState.WAITING_FOR_STATION_COMPLETION:
            # Primary logic is handled by the airtable_poll_callback timer.
            # This state just checks if the polling timer has finished (successfully or timed out).
            # Timer stopped itself (found 99 or timed out)
            if self.airtable_poll_timer is None:
                # Check if timeout occurred (state would be STATION_TIMED_OUT)
                if self.state == RobotState.STATION_TIMED_OUT:
                    self.get_logger().error("Station completion timed out. Halting order processing.")
                    # Stay in STATION_TIMED_OUT state, error handling takes over.
                    pass
                # Check for other errors that might stop polling
                elif self.state == RobotState.AIRTABLE_ERROR:
                    self.get_logger().error("Airtable error occurred while waiting for station completion.")
                    pass  # Stay in AIRTABLE_ERROR
                else:
                    # Polling must have found STATUS_DONE (99)
                    self.get_logger().info(
                        f"Station {self.target_station_index} processing complete.")
                    # Move to the next station in the sequence
                    self.current_sequence_index += 1
                    if self.current_sequence_index < len(self.station_sequence):
                        # More stations to visit
                        self.target_station_index = self.station_sequence[self.current_sequence_index]
                        next_station_name = STATION_COLORS_HSV[self.target_station_index]['name']
                        self.get_logger().info(
                            f"Proceeding to next station: {self.target_station_index} ({next_station_name})")
                        if self.target_station_index == 0:  # Next station is Pickup
                            next_state = RobotState.RETURNING_TO_PICKUP
                        else:  # Next station is another processing station
                            next_state = RobotState.MOVING_TO_STATION
                    else:
                        # This was the last station in the sequence (should be Pickup)
                        # Should not happen if Pickup (0) is always last? Add check.
                        self.get_logger().error("Reached end of sequence unexpectedly.")
                        next_state = RobotState.ERROR  # Or ORDER_COMPLETE?

            # else: Timer still running, do nothing here, wait for callback.

        elif current_state == RobotState.ARRIVED_AT_PICKUP:
            # Successfully returned to the start/pickup station
            self.get_logger().info(
                f"Arrived back at Pickup Station (Index 0). Order '{self.current_order['order_name']}' complete.")
            self.pancakes_made_count += 1
            # Order complete sound
            self.play_sound([(800, 100), (700, 100), (600, 200)])

            # Update Pickup status to DONE (99)
            pickup_field = STATION_INDEX_TO_FIELD.get(0)
            if pickup_field:
                if not self.update_station_status_in_airtable(self.current_order["record_id"], pickup_field, STATUS_DONE):
                    self.get_logger().error("Failed to update final Pickup status to DONE.")
                    # Continue anyway? Or AIRTABLE_ERROR? Let's log and continue for now.
            else:
                self.get_logger().warn(
                    "No Airtable field mapped for Pickup Station (Index 0). Cannot update status.")

            # Transition to check for next order
            next_state = RobotState.ORDER_COMPLETE  # Go to intermediate state before IDLE

        elif current_state == RobotState.ORDER_COMPLETE:
            # Reset for next potential order
            self.current_order = None
            self.station_sequence = []
            self.current_sequence_index = 0
            self.target_station_index = -1
            self.get_logger().info("Order cycle finished. Returning to IDLE.")
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
            GPIO.cleanup()
            self.get_logger().info("GPIO cleanup successful.")
        except Exception as e:
            self.get_logger().error(f"Error during GPIO cleanup: {e}")

    def shutdown_camera(self):
        """Safely stops the Pi Camera."""
        if hasattr(self, 'picam2') and self.picam2:
            try:
                if self.picam2.started:
                    self.get_logger().info("Stopping Pi Camera...")
                    self.picam2.stop()
                    self.get_logger().info("Pi Camera stopped.")
            except Exception as e:
                self.get_logger().error(f"Error stopping camera: {e}")
        if self.debug_windows:
            cv2.destroyAllWindows()

    def shutdown_robot(self):
        """Commands the robot to stop moving and cleans up resources."""
        self.get_logger().info("Initiating robot shutdown sequence...")
        # 1. Stop movement
        self.stop_moving()
        # 2. Cancel timers
        if hasattr(self, 'control_timer') and self.control_timer and not self.control_timer.is_canceled():
            self.control_timer.cancel()
        self.stop_airtable_polling()  # Ensure polling timer is stopped
        # 3. Stop camera
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
                f"Node initialization failed with state: {pancake_robot_node.state.name}. Shutting down.")
            exit_code = 1  # Indicate error exit

    except KeyboardInterrupt:
        print("Keyboard interrupt detected, shutting down.")
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
            # Calls stop, cancels timers, cleans GPIO/Camera
            pancake_robot_node.shutdown_robot()
            pancake_robot_node.destroy_node()
            pancake_robot_node.get_logger().info("Pancake Robot Node destroyed.")

        # Shutdown ROS client library
        if rclpy.ok():
            rclpy.shutdown()
        print("ROS2 shutdown complete.")
        # Ensure GPIO cleanup is attempted even if node init failed partially
        try:
            if GPIO.getmode() is not None:  # Check if GPIO mode was set
                GPIO.cleanup()
                print("Final GPIO cleanup check performed.")
        except Exception as e:
            print(f"Error during final GPIO cleanup check: {e}")

    return exit_code


if __name__ == '__main__':
    import sys
    sys.exit(main())
