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
from libcamera import controls, Transform  # Import Transform for rotation
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
AIRTABLE_API_TOKEN = os.getenv("AIRTABLE_API_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")

if not all([AIRTABLE_API_TOKEN, AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME]):
    print(
        "FATAL: Missing required Airtable environment variables. Please check your .env file."
    )
    raise EnvironmentError(
        "Missing required Airtable environment variables. Please check your .env file."
    )

# --- Construct Airtable URL and Headers ---
AIRTABLE_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
AIRTABLE_HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_API_TOKEN}",
    "Content-Type": "application/json",
}

# --- Field names in Airtable base ---
AIRTABLE_ORDER_NAME_COLUMN = "Order Name"
AIRTABLE_COOKING_1_STATUS_FIELD = "Cooking 1 Status"
AIRTABLE_COOKING_2_STATUS_FIELD = "Cooking 2 Status"
AIRTABLE_WHIPPED_CREAM_STATUS_FIELD = (
    "Whipped Cream Status"  # Assuming physical station 3
)
AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD = (
    "Choco Chips Status"  # Assuming physical station 4
)
AIRTABLE_SPRINKLES_STATUS_FIELD = "Sprinkles Status"  # Assuming physical station 5
AIRTABLE_PICKUP_STATUS_FIELD = "Pickup Status"  # Assuming physical station 0 (after 5)

# --- Airtable Status Codes ---
STATUS_WAITING = 0
STATUS_ARRIVED = 1
STATUS_DONE = 99

# --- Map Airtable Fields to Station Indices ---
# IMPORTANT: These indices MUST match the physical layout encountered by the robot
STATION_FIELD_TO_INDEX = {
    AIRTABLE_COOKING_1_STATUS_FIELD: 1,
    AIRTABLE_COOKING_2_STATUS_FIELD: 2,
    AIRTABLE_WHIPPED_CREAM_STATUS_FIELD: 3,
    AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD: 4,
    AIRTABLE_SPRINKLES_STATUS_FIELD: 5,
    AIRTABLE_PICKUP_STATUS_FIELD: 0,  # Pickup is treated as index 0
}
STATION_INDEX_TO_FIELD = {v: k for k, v in STATION_FIELD_TO_INDEX.items()}

# Define the physical order the robot encounters stations following the line
# Adjust this list if the physical layout is different!
PHYSICAL_STATION_LAYOUT = [1, 2, 3, 4, 5, 0]

# --- Hardware Configuration ---
LEFT_IR_PIN = 16
RIGHT_IR_PIN = 18
IR_LINE_DETECT_SIGNAL = GPIO.HIGH  # HIGH (1) = ON LINE (Dark)
IR_OFF_LINE_SIGNAL = GPIO.LOW  # LOW (0) = OFF LINE (Light)

CAMERA_RESOLUTION = (640, 480)
CAMERA_TRANSFORM = Transform(hflip=True, vflip=True)

# --- Color Detection Configuration (Common Green) ---
COMMON_HSV_LOWER = (35, 100, 100)
COMMON_HSV_UPPER = (85, 255, 255)
COMMON_COLOR_BGR = (0, 255, 0)
STATION_COLORS_HSV = {
    idx: {
        "name": STATION_INDEX_TO_FIELD.get(idx, f"Station {idx}").replace(
            " Status", ""
        ),
        "hsv_lower": COMMON_HSV_LOWER,
        "hsv_upper": COMMON_HSV_UPPER,
        "color_bgr": COMMON_COLOR_BGR,
    }
    # Ensure all stations in the layout have an entry, even if just for name lookup
    for idx in PHYSICAL_STATION_LAYOUT  # Use physical layout for completeness
}

# --- Navigation & Control Parameters ---
AIRTABLE_POLL_RATE = 2.0
BASE_DRIVE_SPEED = 0.01  # m/s - Tune as needed # <<< Adjusted Speed >>>
BASE_ROTATE_SPEED = 0.2  # rad/s - Speed for *correction* turns (tune as needed)
TURN_FACTOR = 0.7  # Speed multiplier during correction turns
LOST_LINE_ROTATE_SPEED = 0.15  # rad/s - Speed for *search* rotation (tune as needed) - USE NEGATIVE FOR RIGHT TURN
COLOR_DETECTION_THRESHOLD = 2000
COLOR_COOLDOWN_SEC = 3.0  # Cooldown *after successful arrival*
# Cooldown after *ignoring* a skipped station should be shorter or handled differently
SKIP_DETECTION_COOLDOWN_SEC = (
    0.5  # Time to ignore color detection after seeing a skipped station
)
STATION_WAIT_TIMEOUT_SEC = 120.0
LEAVING_STATION_DURATION_SEC = 2.0  # Time to drive forward after completing a station


# --- Robot States ---
class RobotState(Enum):
    IDLE = auto()
    FETCHING_ORDER = auto()
    PLANNING_ROUTE = auto()
    CALCULATING_SKIPS = auto()  # New state/step
    LEAVING_STATION = auto()
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


# --- Main Node Class ---
class PancakeRobotNode(Node):
    def __init__(self):
        super().__init__("pancake_robot_node")
        self.get_logger().info("Pancake Robot Node Initializing...")
        # Initialize attributes...
        self.state = RobotState.IDLE
        self.current_order = None
        self.station_sequence = []  # The sequence of stations to *visit*
        self.current_sequence_index = -1  # Index within self.station_sequence
        self.target_station_index = (
            -1
        )  # The *physical* index of the station we are currently heading towards
        self.last_color_detection_time = 0.0  # Single cooldown timer
        self.wait_start_time = 0.0
        self.leaving_station_start_time = 0.0
        self._last_airtable_check_time = 0.0
        self.initial_line_found = False
        self.picam2 = None
        self.debug_windows = True  # Set to False to disable CV windows
        self.cmd_vel_pub = None
        self.audio_publisher = None
        self.drive_client = None
        self.rotate_client = None

        # --- New attributes for skipping logic ---
        self.skipped_stations_to_ignore = (
            0  # How many green patches to ignore before the target
        )
        self.green_patches_seen_since_last_stop = 0  # Counter for patches seen
        self.ignore_color_until = 0.0  # Timestamp until which color detections should be ignored (after seeing a skipped station)

        # Run initializations
        self._init_hardware()
        if self.state in [RobotState.GPIO_ERROR, RobotState.CAMERA_ERROR]:
            self.get_logger().fatal("Hardware init failed.")
            return
        self._init_ros2()
        if self.state == RobotState.ERROR:
            self.get_logger().fatal("ROS2 init failed.")
            self.cleanup_hardware()
            return
        # Start timer
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20 Hz loop
        self.get_logger().info("Initialized and Ready.")
        self.play_sound([(440, 150), (550, 200)])

    def _init_hardware(self):
        try:  # GPIO
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(LEFT_IR_PIN, GPIO.IN)
            GPIO.setup(RIGHT_IR_PIN, GPIO.IN)
            self.get_logger().info(
                f"GPIO ok (L{LEFT_IR_PIN},R{RIGHT_IR_PIN}). Expect {IR_LINE_DETECT_SIGNAL}=ON_LINE."
            )
        except Exception as e:
            self.get_logger().error(f"FATAL: GPIO init: {e}", exc_info=True)
            self.state = RobotState.GPIO_ERROR
            return
        try:  # Camera
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": CAMERA_RESOLUTION}, transform=CAMERA_TRANSFORM
            )
            self.picam2.configure(config)
            # Use fixed focus; continuous can be slow/unreliable for this
            self.picam2.set_controls(
                {"AfMode": controls.AfModeEnum.Manual, "LensPosition": 0.0}
            )
            # Reduce exposure time if possible to minimize motion blur
            # self.picam2.set_controls({"ExposureTime": 10000}) # Example: 10ms, adjust as needed
            self.picam2.start()
            time.sleep(2)  # Allow camera to stabilize
            self.get_logger().info("Camera ok.")
            if self.debug_windows:
                cv2.namedWindow("Camera Feed")
                cv2.namedWindow("Color Detection Mask")
        except Exception as e:
            self.get_logger().error(f"FATAL: Camera init: {e}", exc_info=True)
            self.cleanup_gpio()
            self.state = RobotState.CAMERA_ERROR
            return

    def _init_ros2(self):
        try:
            self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
            self.audio_publisher = self.create_publisher(
                AudioNoteVector, "/cmd_audio", 10
            )
            self.drive_client = ActionClient(self, DriveDistance, "/drive_distance")
            self.rotate_client = ActionClient(self, RotateAngle, "/rotate_angle")
            if not self.audio_publisher or not self.cmd_vel_pub:
                raise RuntimeError("Publisher creation failed.")
            # Wait briefly for connections to establish
            time.sleep(1.0)
            self.get_logger().info("ROS2 ok.")
        except Exception as e:
            self.get_logger().error(f"FATAL: ROS2 init: {e}", exc_info=True)
            self.state = RobotState.ERROR

    # --- Airtable Functions (Unchanged from previous version) ---
    def fetch_order_from_airtable(self):
        try:
            params = {
                "maxRecords": 1,
                # Find orders where *at least* Cooking 1 OR Cooking 2 is WAITING,
                # AND Pickup is WAITING (to avoid reprocessing completed orders)
                "filterByFormula": f"AND(OR({{{AIRTABLE_COOKING_1_STATUS_FIELD}}}=0, {{{AIRTABLE_COOKING_2_STATUS_FIELD}}}=0), {{{AIRTABLE_PICKUP_STATUS_FIELD}}}=0)",
                "sort[0][field]": AIRTABLE_ORDER_NAME_COLUMN,
                "sort[0][direction]": "asc",
            }
            response = requests.get(
                url=AIRTABLE_URL,
                headers=AIRTABLE_HEADERS,
                params=params,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
            records = data.get("records", [])
            if not records:
                return None
            record = records[0]
            record_id = record.get("id")
            fields = record.get("fields", {})
            order_name = fields.get(AIRTABLE_ORDER_NAME_COLUMN)
            if not record_id or not order_name:
                self.get_logger().error(f"Airtable record invalid: {record}")
                return None
            self.get_logger().info(f"Fetched order '{order_name}' (ID: {record_id}).")
            return {
                "record_id": record_id,
                "order_name": order_name,
                "station_status": {
                    field: fields.get(
                        field, STATUS_DONE
                    )  # Default to DONE if field missing
                    for field in STATION_FIELD_TO_INDEX.keys()
                },
            }
        except requests.exceptions.Timeout:
            self.get_logger().error("Airtable fetch timed out.")
            self.state = RobotState.AIRTABLE_ERROR
            return None
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Airtable fetch error: {e}")
            self.state = RobotState.AIRTABLE_ERROR
            return None
        except Exception as e:
            self.get_logger().error(
                f"Airtable fetch unexpected error: {e}", exc_info=True
            )
            self.state = RobotState.AIRTABLE_ERROR
            return None

    def update_station_status(self, record_id, field, status):
        if not record_id or not field:
            self.get_logger().error("Airtable update error: Missing ID or field")
            return False
        data = {"fields": {field: status}}
        url = f"{AIRTABLE_URL}/{record_id}"
        try:
            response = requests.patch(  # Use PATCH for updates
                url=url, headers=AIRTABLE_HEADERS, json=data, timeout=10
            )
            response.raise_for_status()  # Check for HTTP errors
            self.get_logger().info(f"Airtable: Updated {field} to {status}.")
            return True
        except requests.exceptions.Timeout:
            self.get_logger().error(f"Airtable update timed out for {field}.")
            self.state = RobotState.AIRTABLE_ERROR
            return False
        except requests.exceptions.RequestException as e:
            self.get_logger().error(
                f"Airtable update error for {field}: {e} - Response: {e.response.text if e.response else 'No Response'}"
            )
            self.state = RobotState.AIRTABLE_ERROR
            return False
        except Exception as e:
            self.get_logger().error(
                f"Airtable update unexpected error: {e}", exc_info=True
            )
            self.state = RobotState.AIRTABLE_ERROR
            return False

    def wait_for_station_completion(self, record_id, field):
        if not record_id or not field:
            self.get_logger().error("Airtable check error: Missing ID or field")
            return False
        url = f"{AIRTABLE_URL}/{record_id}"
        try:
            response = requests.get(url=url, headers=AIRTABLE_HEADERS, timeout=10)
            response.raise_for_status()
            data = response.json()
            current_status = data.get("fields", {}).get(field)
            self.get_logger().debug(
                f"Airtable check {field}: Status is {current_status}"
            )
            return current_status == STATUS_DONE
        except requests.exceptions.Timeout:
            self.get_logger().warning(f"Airtable check timed out for {field}.")
            return False  # Don't set error state, just failed check
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Airtable check error ({field}): {e}")
            # Don't set error state immediately, maybe transient network issue
            return False
        except Exception as e:
            self.get_logger().error(
                f"Airtable check unexpected error: {e}", exc_info=True
            )
            return False  # Don't set error state

    # --- Robot Movement and Sensors ---
    def move_robot(self, linear_x, angular_z):
        if (
            self.state
            in [
                RobotState.ERROR,
                RobotState.CAMERA_ERROR,
                RobotState.GPIO_ERROR,
                RobotState.AIRTABLE_ERROR,
            ]
            or not self.cmd_vel_pub
        ):
            # Ensure robot stops if in error state or publisher not ready
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            if self.cmd_vel_pub:
                try:
                    self.cmd_vel_pub.publish(twist)
                except Exception as e:
                    self.get_logger().error(f"Error publishing stop Twist: {e}")
            return

        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        # self.get_logger().debug(f"Publishing Twist: Lin={linear_x:.3f}, Ang={angular_z:.3f}")
        try:
            self.cmd_vel_pub.publish(twist)
        except Exception as e:
            self.get_logger().error(f"Failed to publish Twist: {e}")
            # Consider setting an error state if publish fails repeatedly

    def stop_moving(self):
        # self.get_logger().debug("Stopping movement...")
        # Send stop command multiple times for robustness
        for _ in range(3):
            self.move_robot(0.0, 0.0)
            time.sleep(0.02)

    def read_ir_sensors(self):
        """Reads IR sensors. Returns (left_on_line, right_on_line). HIGH = ON LINE."""
        try:
            left_val = GPIO.input(LEFT_IR_PIN)
            right_val = GPIO.input(RIGHT_IR_PIN)
            # self.get_logger().debug(f"IR Raw: L={left_val}, R={right_val}")
            # Return True if the sensor value matches the 'ON LINE' signal level
            return (left_val == IR_LINE_DETECT_SIGNAL), (
                right_val == IR_LINE_DETECT_SIGNAL
            )
        except Exception as e:
            # Check if it's RuntimeError due to GPIO cleanup already happening
            if isinstance(e, RuntimeError) and "cannot determine voltage" in str(e):
                self.get_logger().warning(
                    f"Ignoring IR read error during shutdown: {e}"
                )
                return False, False  # Assume off line during shutdown
            else:
                self.get_logger().error(f"IR sensor read error: {e}", exc_info=True)
                self.state = RobotState.GPIO_ERROR  # Set error state on read failure
                return False, False

    # --- Color Detection ---
    def check_for_station_color(self, frame):
        """
        Checks for the common station color (green) in the frame.
        Returns: (detected_flag, display_frame, mask_frame)
        """
        detected_flag = False
        display_frame = frame.copy()  # Work on a copy
        mask_frame = np.zeros(
            (frame.shape[0], frame.shape[1]), dtype=np.uint8
        )  # Blank mask initially

        # --- Add State and Target Info to Display ---
        cv2.putText(
            display_frame,
            f"State: {self.state.name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        target_name = STATION_COLORS_HSV.get(self.target_station_index, {}).get(
            "name", "None"
        )
        cv2.putText(
            display_frame,
            f"Target: {target_name} ({self.target_station_index})",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            display_frame,
            f"Seen: {self.green_patches_seen_since_last_stop} / Ignore: {self.skipped_stations_to_ignore}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # --- Color Detection Logic ---
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower = np.array(COMMON_HSV_LOWER)
            upper = np.array(COMMON_HSV_UPPER)
            mask = cv2.inRange(hsv, lower, upper)

            # Optional: Apply morphological operations to reduce noise
            # kernel = np.ones((5,5),np.uint8)
            # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            mask_frame = mask  # Store the mask for display/debugging
            pixels = cv2.countNonZero(mask)

            # Check if enough pixels are detected AND cooldown period has passed
            now = time.time()

            # Add pixel count to display
            cv2.putText(
                display_frame,
                f"Green Px: {pixels}",
                (frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            # Check BOTH the general cooldown AND the short skip cooldown
            if (
                pixels > COLOR_DETECTION_THRESHOLD
                and now > self.last_color_detection_time
                and now > self.ignore_color_until
            ):
                # We detected *a* green patch and are allowed to process it
                detected_flag = True
                # Don't log detection here, log it in the state machine where we decide if it's the target or skipped

        except cv2.error as e:
            self.get_logger().error(f"OpenCV error during color detection: {e}")
            # Return current frame state without detection
            return False, display_frame, mask_frame
        except Exception as e:
            self.get_logger().error(
                f"Unexpected error during color detection: {e}", exc_info=True
            )
            # Return current frame state without detection
            return False, display_frame, mask_frame

        # Add detection status visualization
        if detected_flag:
            cv2.putText(
                display_frame,
                "GREEN DETECTED",
                (frame.shape[1] // 2 - 100, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return detected_flag, display_frame, mask_frame

    # --- Sound (Unchanged) ---
    def play_sound(self, notes):
        if not hasattr(self, "audio_publisher") or self.audio_publisher is None:
            self.get_logger().warning("Audio publisher not initialized.")
            return
        msg = AudioNoteVector()
        log_str = []
        for f, d in notes:
            if f <= 0 or d <= 0:
                self.get_logger().warning(f"Skip invalid note: {f}Hz,{d}ms")
                continue
            n = AudioNote()
            n.frequency = int(f)
            # Ensure duration is at least minimal
            dur_sec = int(d / 1000)
            dur_nsec = int((d % 1000) * 1e6)
            if dur_sec == 0 and dur_nsec == 0:
                dur_nsec = 1000  # Ensure non-zero duration if ms was < 1
            n.max_runtime = Duration(sec=dur_sec, nanosec=dur_nsec)
            msg.notes.append(n)
            log_str.append(f"({f}Hz,{d}ms)")

        if not msg.notes:
            self.get_logger().warning("No valid notes to play.")
            return

        self.get_logger().info(f"Playing Audio Sequence: {', '.join(log_str)}")
        try:
            # Append override notes flag if needed by Create3 audio system
            # msg.append = False # Or True depending on desired behavior
            self.audio_publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Audio publish failed: {e}", exc_info=True)

    # --- Cleanup ---
    def cleanup_gpio(self):
        self.get_logger().info("Cleaning up GPIO...")
        try:
            # Check if GPIO has been initialized before cleaning up
            # Getting the mode will raise an exception if not set
            current_mode = GPIO.getmode()
            if current_mode is not None:
                GPIO.cleanup()
                self.get_logger().info("GPIO cleanup successful.")
            else:
                self.get_logger().info("GPIO was not initialized, skipping cleanup.")
        except Exception as e:
            # Ignore specific errors that might occur if pins already cleaned up
            if "eboard Directed Edge Detector" not in str(e):
                self.get_logger().error(f"GPIO cleanup error: {e}")

    def cleanup_hardware(self):
        self.get_logger().info("Cleaning up hardware...")
        # Stop Camera first
        if self.picam2:
            try:
                self.get_logger().info("Stopping camera...")
                self.picam2.stop()
                self.get_logger().info("Camera stopped.")
            except Exception as e:
                self.get_logger().error(f"Camera stop error: {e}")
        # Cleanup GPIO
        self.cleanup_gpio()

        # --- Calculate Skips Function ---

    def calculate_skips_to_next_station(self):
        """
        Calculates how many physical stations need to be ignored (skipped).
        Sets self.target_station_index for the upcoming move.
        This should be called AFTER current_sequence_index is potentially incremented,
        before entering MOVING_TO_STATION.
        Returns True on success, False on error.
        """
        # --- Handle the first move explicitly (index == 0) ---
        if self.current_sequence_index == 0:
            if not self.station_sequence:  # Safety check: Ensure sequence exists
                self.get_logger().error(
                    "Cannot calculate skips for first move: station_sequence is empty."
                )
                self.state = RobotState.ERROR
                return False
            # *** FIX: Set the target for the first move ***
            self.target_station_index = self.station_sequence[0]
            self.skipped_stations_to_ignore = 0  # No skips needed from origin
            self.green_patches_seen_since_last_stop = 0
            self.get_logger().info(
                f"Calculating for first move. Target: {self.target_station_index}, Skips: 0"
            )
            return True  # Proceed to MOVING_TO_STATION

        # --- Handle subsequent moves (index > 0) ---
        # Validate index bounds *before* accessing sequence elements
        if self.current_sequence_index < 0 or self.current_sequence_index >= len(
            self.station_sequence
        ):
            # Log index and sequence length for debugging
            seq_len = (
                len(self.station_sequence)
                if self.station_sequence is not None
                else "None"
            )
            self.get_logger().error(
                f"Cannot calculate skips: sequence index {self.current_sequence_index} out of bounds for sequence length {seq_len}."
            )
            self.state = RobotState.ERROR
            return False
        # Also check if index-1 is valid
        if self.current_sequence_index - 1 < 0:
            self.get_logger().error(
                f"Cannot calculate skips: invalid index {self.current_sequence_index} for determining last visited station."
            )
            self.state = RobotState.ERROR
            return False

        # Get the physical index of the station just completed
        last_visited_station_idx = self.station_sequence[
            self.current_sequence_index - 1
        ]
        # Get the physical index of the next target station
        target_station_idx = self.station_sequence[self.current_sequence_index]
        # *** Set the target for subsequent moves ***
        self.target_station_index = target_station_idx  # Update the main target index

        try:
            last_layout_pos = PHYSICAL_STATION_LAYOUT.index(last_visited_station_idx)
            target_layout_pos = PHYSICAL_STATION_LAYOUT.index(target_station_idx)
        except ValueError:
            self.get_logger().error(
                f"Invalid station index found during skip calculation: last={last_visited_station_idx}, target={target_station_idx}. Check PHYSICAL_STATION_LAYOUT and sequence: {self.station_sequence}."
            )
            self.state = RobotState.ERROR
            return False

        skipped_count = 0
        # Create the set *after* validation ensures station_sequence is okay
        planned_stations_set = set(self.station_sequence)  # For efficient lookup

        # Iterate through layout positions *between* last and target
        current_layout_pos = (last_layout_pos + 1) % len(PHYSICAL_STATION_LAYOUT)
        iteration_guard = 0  # Prevent infinite loops

        while (
            current_layout_pos != target_layout_pos
            and iteration_guard < len(PHYSICAL_STATION_LAYOUT) * 2
        ):
            station_idx_at_pos = PHYSICAL_STATION_LAYOUT[current_layout_pos]
            # Check if this station *exists physically* but *is not in our planned route*
            if station_idx_at_pos not in planned_stations_set:
                skipped_count += 1
                self.get_logger().debug(
                    f"Counting physical station {station_idx_at_pos} at layout pos {current_layout_pos} as skipped."
                )

            current_layout_pos = (current_layout_pos + 1) % len(PHYSICAL_STATION_LAYOUT)
            iteration_guard += 1

        if iteration_guard >= len(PHYSICAL_STATION_LAYOUT) * 2:
            self.get_logger().error(
                f"Loop detected or target layout position {target_layout_pos} not found during skipped station calculation. Last pos: {last_layout_pos}. Sequence: {self.station_sequence}. Layout: {PHYSICAL_STATION_LAYOUT}. Check logic."
            )
            self.state = RobotState.ERROR
            return False

        self.skipped_stations_to_ignore = skipped_count
        self.green_patches_seen_since_last_stop = 0  # Reset counter for the new leg
        self.get_logger().info(
            f"Calculated move: From {last_visited_station_idx} to {target_station_idx}. Need to ignore {self.skipped_stations_to_ignore} green patches."
        )
        return True

    # --- Main Control Loop ---
    def control_loop(self):
        """Main state machine and control logic for the robot."""
        # --- Error State Check ---
        # This check should be at the very beginning
        if self.state in [
            RobotState.ERROR,
            RobotState.CAMERA_ERROR,
            RobotState.GPIO_ERROR,
            RobotState.AIRTABLE_ERROR,
        ]:
            # If already stopping or stopped, just return to avoid repeated stop commands
            # Add a small check for current velocity if possible, otherwise just stop once
            # For simplicity, we just call stop_moving() which includes multiple publishes
            self.stop_moving()
            # Optionally log the error state only once or periodically
            # self.get_logger().error(f"Robot in error state: {self.state.name}. Halting.", throttle_duration_sec=5.0)
            return  # Prevent further execution in error states

        # --- Camera Frame Acquisition ---
        display_frame, mask_frame, raw_frame = None, None, None
        color_detected_this_cycle = False  # Flag for this specific loop iteration

        if self.picam2 and self.state != RobotState.CAMERA_ERROR:
            try:
                raw_frame = self.picam2.capture_array()
                # Perform color check regardless of state for debugging display
                color_detected_this_cycle, display_frame, mask_frame = (
                    self.check_for_station_color(raw_frame)
                )

            except Exception as e:
                self.get_logger().error(
                    f"Camera capture or processing error: {e}", exc_info=True
                )
                self.state = RobotState.CAMERA_ERROR
                self.stop_moving()
                return  # Exit loop on camera error
        else:
            # Handle case where camera isn't available but node is running
            if self.state not in [
                RobotState.IDLE,
                RobotState.ALL_ORDERS_COMPLETE,
            ]:  # Only error if camera needed
                self.get_logger().error("Camera not available!")
                # self.state = RobotState.CAMERA_ERROR # Set error state if camera required for current operation
                # For now, allow IDLE without camera, but error otherwise?

        # --- State Machine ---
        try:
            current_state = self.state  # Store state at start of loop iteration

            # --- State: IDLE ---
            if current_state == RobotState.IDLE:
                self.stop_moving()
                # Reset most state variables
                self.current_order = None
                self.station_sequence = []
                self.current_sequence_index = -1
                self.target_station_index = -1
                self.initial_line_found = False
                self.skipped_stations_to_ignore = 0
                self.green_patches_seen_since_last_stop = 0
                self.ignore_color_until = 0.0

                # Check for new orders periodically
                if (
                    time.time() - self._last_airtable_check_time
                    > AIRTABLE_POLL_RATE * 2
                ):  # Check less often in idle
                    self.get_logger().info("Idle: Checking for new orders...")
                    self._last_airtable_check_time = time.time()
                    self.current_order = self.fetch_order_from_airtable()
                    if self.current_order:
                        if (
                            self.state != RobotState.AIRTABLE_ERROR
                        ):  # Check if fetch failed
                            self.get_logger().info(
                                f"Order '{self.current_order['order_name']}' received."
                            )
                            self.state = RobotState.PLANNING_ROUTE
                    elif self.state != RobotState.AIRTABLE_ERROR:
                        # No orders found, remain IDLE until next check or transition to ALL_ORDERS_COMPLETE after a while
                        # self.get_logger().info("Idle: No new orders found.")
                        # Consider adding a timeout to enter ALL_ORDERS_COMPLETE state?
                        pass  # Stay IDLE

            # --- State: PLANNING_ROUTE ---
            elif current_state == RobotState.PLANNING_ROUTE:
                if not self.current_order:
                    self.get_logger().error(
                        "PLANNING: No current_order found! Returning to IDLE."
                    )
                    self.state = RobotState.IDLE
                    return  # Exit loop for this cycle

                self.station_sequence = []
                status = self.current_order["station_status"]

                # Build sequence based on physical layout and required status
                for station_idx in PHYSICAL_STATION_LAYOUT:
                    field_name = STATION_INDEX_TO_FIELD.get(station_idx)
                    if field_name and status.get(field_name) == STATUS_WAITING:
                        self.station_sequence.append(station_idx)

                # Ensure Pickup (0) is always last if it's needed
                if (
                    STATION_FIELD_TO_INDEX[AIRTABLE_PICKUP_STATUS_FIELD]
                    in self.station_sequence
                ):
                    if (
                        self.station_sequence[-1]
                        != STATION_FIELD_TO_INDEX[AIRTABLE_PICKUP_STATUS_FIELD]
                    ):
                        self.station_sequence.remove(
                            STATION_FIELD_TO_INDEX[AIRTABLE_PICKUP_STATUS_FIELD]
                        )
                        self.station_sequence.append(
                            STATION_FIELD_TO_INDEX[AIRTABLE_PICKUP_STATUS_FIELD]
                        )
                else:
                    # If pickup wasn't explicitly waiting=0, but other steps are, add it.
                    # This assumes pickup is *always* required if any other step is.
                    # Check if any non-pickup station is in the sequence
                    if any(
                        s != STATION_FIELD_TO_INDEX[AIRTABLE_PICKUP_STATUS_FIELD]
                        for s in self.station_sequence
                    ):
                        self.get_logger().warning(
                            f"Pickup status was not 0, but other steps required. Adding Pickup to sequence."
                        )
                        self.station_sequence.append(
                            STATION_FIELD_TO_INDEX[AIRTABLE_PICKUP_STATUS_FIELD]
                        )

                if not self.station_sequence:
                    self.get_logger().error(
                        f"Route planning resulted in empty sequence for order '{self.current_order['order_name']}'. Order might be already complete or invalid. Returning to IDLE."
                    )
                    # Potentially update Airtable to indicate an issue?
                    self.state = RobotState.IDLE
                    self.current_order = None  # Clear invalid order
                else:
                    # Successfully planned route
                    self.current_sequence_index = 0
                    # Calculate skips for the *first* leg (from origin)
                    if not self.calculate_skips_to_next_station():
                        # Error occurred during calculation, state set within function
                        return
                    self.initial_line_found = False  # Need to find line initially
                    self.get_logger().info(
                        f"Route Planned: {self.station_sequence}. Next Target: {self.target_station_index}. Skips Needed: {self.skipped_stations_to_ignore}"
                    )
                    self.state = RobotState.MOVING_TO_STATION
                    self.play_sound(
                        [(440, 100), (550, 100), (660, 100)]
                    )  # Route planned sound

            # --- State: LEAVING_STATION ---
            # This state provides a fixed duration forward movement after completing a task
            elif current_state == RobotState.LEAVING_STATION:
                elapsed = time.time() - self.leaving_station_start_time
                if elapsed < LEAVING_STATION_DURATION_SEC:
                    # Move slowly forward to clear the station area
                    self.move_robot(BASE_DRIVE_SPEED * 0.5, 0.0)
                else:
                    self.get_logger().info(
                        f"Finished leaving station. Proceeding to calculate skips for next target: {self.target_station_index}."
                    )
                    self.stop_moving()
                    # Now calculate skips *before* moving
                    self.state = RobotState.CALCULATING_SKIPS

            # --- State: CALCULATING_SKIPS ---
            # Intermediate step to ensure skips are calculated before moving
            elif current_state == RobotState.CALCULATING_SKIPS:
                if self.calculate_skips_to_next_station():
                    self.get_logger().info(
                        f"Skips calculated. Moving to station {self.target_station_index}."
                    )
                    self.initial_line_found = (
                        False  # Reset line finding for the new leg
                    )
                    self.state = RobotState.MOVING_TO_STATION
                else:
                    # Error state was set within the function
                    self.stop_moving()
                    return

            # --- State: MOVING_TO_STATION ---
            elif current_state == RobotState.MOVING_TO_STATION:
                # --- Color Detection Check (Skip/Target Logic) ---
                if color_detected_this_cycle:
                    self.green_patches_seen_since_last_stop += 1
                    now = time.time()
                    self.get_logger().debug(
                        f"Detected green patch #{self.green_patches_seen_since_last_stop}. Target: {self.target_station_index}. Need to ignore: {self.skipped_stations_to_ignore}."
                    )

                    # --- Check if this is the TARGET or a SKIPPED station ---
                    if (
                        self.green_patches_seen_since_last_stop
                        > self.skipped_stations_to_ignore
                    ):
                        # --- TARGET STATION REACHED ---
                        self.get_logger().info(
                            f"TARGET station {self.target_station_index} detected and reached!"
                        )
                        self.play_sound([(523, 100), (659, 150)])  # Arrival sound
                        self.stop_moving()
                        # Apply the longer cooldown after successful arrival
                        self.last_color_detection_time = now + COLOR_COOLDOWN_SEC
                        self.state = RobotState.ARRIVED_AT_STATION
                        return  # Exit loop for this cycle to process arrival

                    else:
                        # --- SKIPPED STATION SEEN ---
                        self.get_logger().info(
                            f"Ignoring detected green patch (Seen {self.green_patches_seen_since_last_stop}/{self.skipped_stations_to_ignore} needed). Continuing to target {self.target_station_index}."
                        )
                        # Apply a *short* cooldown to avoid re-detecting the same patch immediately
                        self.ignore_color_until = now + SKIP_DETECTION_COOLDOWN_SEC
                        # Optional: Play a subtle sound for skipped station?
                        # self.play_sound([(220, 50)])
                        # *** Continue driving - DO NOT change state ***
                        pass  # Explicitly stay in MOVING_TO_STATION

                # --- Line Following Logic (Only executes if color wasn't the target) ---
                if (
                    self.state == RobotState.MOVING_TO_STATION
                ):  # Re-check state as it might have changed above
                    # Check target validity (should be set correctly by planning/skip calc)
                    if (
                        self.target_station_index == -1
                        or self.current_sequence_index < 0
                        or self.current_sequence_index >= len(self.station_sequence)
                    ):
                        self.get_logger().error(
                            f"MOVING: Invalid target ({self.target_station_index}) or sequence index ({self.current_sequence_index}). Halting."
                        )
                        self.stop_moving()
                        self.state = RobotState.ERROR
                        return

                    # Read IR Sensors
                    left_on, right_on = self.read_ir_sensors()
                    if self.state == RobotState.GPIO_ERROR:  # Check if read_ir failed
                        self.get_logger().error("Halting due to GPIO read error.")
                        self.stop_moving()
                        return

                    # --- Line Following & Recovery ---
                    if not left_on and not right_on:
                        # === Both OFF line ===
                        # If we haven't found the line initially for this leg, turn to find it
                        # If we had the line and lost it, also turn to find it.
                        # **Consistently turn RIGHT (negative angular_z) to search**
                        if not self.initial_line_found:
                            # self.get_logger().debug("Searching for initial line (turning RIGHT)...")
                            pass  # Avoid flooding logs
                        else:
                            self.get_logger().warning(
                                "Line lost! Turning RIGHT to search..."
                            )
                            # Optional sound for lost line
                            # self.play_sound([(330, 100)])

                        self.move_robot(0.0, -LOST_LINE_ROTATE_SPEED)  # Turn Right

                    else:
                        # === At least one sensor ON line ===
                        if not self.initial_line_found:
                            self.get_logger().info("Initial line found for this leg!")
                            # self.play_sound([(660, 100)]) # Optional sound
                            self.initial_line_found = True

                        # --- Standard Line Following ---
                        if left_on and right_on:
                            # Both ON -> Drive straight
                            # self.get_logger().debug("Line Follow: Straight")
                            self.move_robot(BASE_DRIVE_SPEED, 0.0)
                        elif left_on and not right_on:
                            # Left ON, Right OFF -> Robot drifted RIGHT -> Turn LEFT
                            # self.get_logger().debug("Line Follow: Correct Left")
                            self.move_robot(
                                BASE_DRIVE_SPEED * TURN_FACTOR,  # Slightly slower
                                BASE_ROTATE_SPEED,  # Turn Left (positive)
                            )
                        elif not left_on and right_on:
                            # Left OFF, Right ON -> Robot drifted LEFT -> Turn RIGHT
                            # self.get_logger().debug("Line Follow: Correct Right")
                            self.move_robot(
                                BASE_DRIVE_SPEED * TURN_FACTOR,  # Slightly slower
                                -BASE_ROTATE_SPEED,  # Turn Right (negative)
                            )
            # --- End of MOVING_TO_STATION ---

            # --- State: ARRIVED_AT_STATION ---
            elif current_state == RobotState.ARRIVED_AT_STATION:
                # Already stopped in MOVING state
                # self.stop_moving() # Ensure stopped

                # Validate sequence index and target
                if (
                    self.current_sequence_index < 0
                    or self.current_sequence_index >= len(self.station_sequence)
                ):
                    self.get_logger().error(
                        f"ARRIVED: Invalid sequence index {self.current_sequence_index}. Halting."
                    )
                    self.state = RobotState.ERROR
                    return
                idx = self.station_sequence[self.current_sequence_index]
                if idx != self.target_station_index:
                    self.get_logger().error(
                        f"ARRIVED: Mismatch between sequence index {self.current_sequence_index} (station {idx}) and target_station_index ({self.target_station_index}). Halting."
                    )
                    self.state = RobotState.ERROR
                    return

                field = STATION_INDEX_TO_FIELD.get(idx)
                if not field:
                    self.get_logger().error(
                        f"ARRIVED: No Airtable field name found for station index {idx}. Halting."
                    )
                    self.state = RobotState.ERROR
                    return

                self.get_logger().info(
                    f"Arrived at Station {idx} ({field}). Updating status to ARRIVED."
                )
                # Update Airtable status
                if self.update_station_status(
                    self.current_order["record_id"], field, STATUS_ARRIVED
                ):
                    # Successfully updated status, now wait for completion
                    self.wait_start_time = time.time()
                    self._last_airtable_check_time = (
                        0  # Reset check timer for immediate check
                    )
                    self.state = RobotState.WAITING_FOR_STATION_COMPLETION
                    self.get_logger().info(
                        f"Waiting for station {idx} ({field}) to be marked DONE."
                    )
                    self.play_sound([(440, 100), (440, 100)])  # Waiting sound
                else:
                    # update_station_status already set state to AIRTABLE_ERROR
                    self.get_logger().error(
                        f"Failed to update Airtable status for {field}. Halting."
                    )
                    # State should already be AIRTABLE_ERROR
                    self.stop_moving()
                    return

            # --- State: WAITING_FOR_STATION_COMPLETION ---
            elif current_state == RobotState.WAITING_FOR_STATION_COMPLETION:
                # Check for timeout
                elapsed = time.time() - self.wait_start_time
                if elapsed > STATION_WAIT_TIMEOUT_SEC:
                    self.get_logger().warning(
                        f"WAIT TIMEOUT ({elapsed:.1f}s > {STATION_WAIT_TIMEOUT_SEC}s) for station {self.target_station_index}. Aborting order."
                    )
                    self.play_sound([(330, 500), (220, 500)])  # Timeout sound
                    self.state = RobotState.STATION_TIMED_OUT
                    return  # Exit loop

                # Check Airtable periodically
                if time.time() - self._last_airtable_check_time >= AIRTABLE_POLL_RATE:
                    self._last_airtable_check_time = time.time()

                    # Validate index before checking Airtable
                    if (
                        self.current_sequence_index < 0
                        or self.current_sequence_index >= len(self.station_sequence)
                    ):
                        self.get_logger().error(
                            f"WAITING: Invalid sequence index {self.current_sequence_index}. Halting."
                        )
                        self.state = RobotState.ERROR
                        return
                    idx = self.station_sequence[self.current_sequence_index]
                    field = STATION_INDEX_TO_FIELD.get(idx)
                    if not field:
                        self.get_logger().error(
                            f"WAITING: No field name for index {idx}. Halting."
                        )
                        self.state = RobotState.ERROR
                        return

                    self.get_logger().debug(
                        f"Checking Airtable if {field} (Station {idx}) is DONE..."
                    )
                    if self.wait_for_station_completion(
                        self.current_order["record_id"], field
                    ):
                        # --- Station Completed ---
                        self.get_logger().info(
                            f"Station {idx} ({field}) reported DONE by Airtable."
                        )
                        self.play_sound([(659, 150), (784, 200)])  # Success sound

                        # Increment sequence index to move to the next station
                        self.current_sequence_index += 1

                        # --- Check if Order is Complete ---
                        if self.current_sequence_index >= len(self.station_sequence):
                            self.get_logger().info(
                                "All stations for this order complete!"
                            )
                            self.state = RobotState.ORDER_COMPLETE
                        else:
                            # --- Move to Next Station ---
                            # Next target is determined by the new current_sequence_index
                            next_target_idx = self.station_sequence[
                                self.current_sequence_index
                            ]
                            self.get_logger().info(
                                f"Station {idx} done. Preparing to leave and move to next station: {next_target_idx}."
                            )
                            # Start the leaving sequence
                            self.leaving_station_start_time = time.time()
                            self.state = RobotState.LEAVING_STATION
                            # Skip calculation will happen *after* leaving state
                    elif self.state == RobotState.AIRTABLE_ERROR:
                        # wait_for_station_completion might set this on comms error
                        self.get_logger().error(
                            "Airtable error occurred during status check. Halting."
                        )
                        self.stop_moving()
                    else:
                        # Not done yet, continue waiting
                        self.get_logger().debug(
                            f"Station {field} not yet DONE. Continuing wait."
                        )
                        pass  # Remain in WAITING_FOR_STATION_COMPLETION

            # --- State: ORDER_COMPLETE ---
            elif current_state == RobotState.ORDER_COMPLETE:
                order_name = (
                    self.current_order.get("order_name", "Unknown Order")
                    if self.current_order
                    else "Unknown Order"
                )
                self.get_logger().info(f"Order '{order_name}' successfully COMPLETED.")
                self.play_sound(
                    [(784, 150), (880, 150), (1047, 250)]
                )  # Order complete fanfare
                self.stop_moving()
                # Reset for next order search
                self.current_order = None  # Clear completed order
                self.state = RobotState.IDLE  # Go back to idle to look for more orders
                # Reset timers and counters just in case
                self._last_airtable_check_time = 0.0
                # Add a small delay before checking for new orders immediately
                time.sleep(1.0)

            # --- State: ALL_ORDERS_COMPLETE ---
            # This state might not be reached if IDLE keeps checking
            elif current_state == RobotState.ALL_ORDERS_COMPLETE:
                self.get_logger().info("No pending orders found. Entering idle sleep.")
                self.play_sound([(440, 200), (440, 200)])  # Sleepy sound
                self.stop_moving()
                time.sleep(5.0)  # Wait longer before checking again
                self.state = RobotState.IDLE  # Go back to checking

            # --- State: STATION_TIMED_OUT ---
            elif current_state == RobotState.STATION_TIMED_OUT:
                order_name = (
                    self.current_order.get("order_name", "Unknown Order")
                    if self.current_order
                    else "Unknown Order"
                )
                self.get_logger().error(
                    f"STATION TIMED OUT waiting for station {self.target_station_index}. Aborting order '{order_name}'."
                )
                self.stop_moving()
                # TODO: Optionally update Airtable to indicate the timeout/failure?
                # Reset as if order failed/aborted
                self.current_order = None
                self.state = RobotState.IDLE  # Return to Idle after timeout

            # --- State: AIRTABLE_ERROR ---
            # This state is now entered via specific function calls failing
            elif current_state == RobotState.AIRTABLE_ERROR:
                # Logged when entering the state by the function that failed
                # self.get_logger().error("Persistent AIRTABLE ERROR detected. System halted.")
                self.play_sound([(330, 300), (330, 300), (330, 300)])  # Error sound
                self.stop_moving()
                # Stay in this state until resolved externally or restarted

        except Exception as e:
            self.get_logger().error(
                f"CRITICAL: Unhandled exception in state machine ({current_state.name}): {e}",  # Log current state
                exc_info=True,
            )
            self.state = RobotState.ERROR
            self.stop_moving()

        # --- Display Update ---
        finally:
            if self.debug_windows:
                try:
                    # Use the potentially modified display_frame from color check
                    if display_frame is not None:
                        cv2.imshow("Camera Feed", display_frame)
                    elif raw_frame is not None:  # Show raw if display_frame failed
                        # Add state text to raw frame if possible
                        cv2.putText(
                            raw_frame,
                            f"State: {self.state.name}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.imshow("Camera Feed", raw_frame)

                    # Show the color mask
                    if mask_frame is not None:
                        # Convert single channel mask to BGR for display if needed
                        if len(mask_frame.shape) == 2:
                            mask_display = cv2.cvtColor(mask_frame, cv2.COLOR_GRAY2BGR)
                        else:
                            mask_display = mask_frame
                        cv2.imshow("Color Detection Mask", mask_display)
                    else:
                        # Show blank mask if not generated
                        blank_mask = np.zeros(
                            (CAMERA_RESOLUTION[1], CAMERA_RESOLUTION[0], 3),
                            dtype=np.uint8,
                        )
                        cv2.putText(
                            blank_mask,
                            "No Mask",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (128, 128, 128),
                            2,
                        )
                        cv2.imshow("Color Detection Mask", blank_mask)

                    # Key press handling (non-blocking)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        self.get_logger().info(
                            "Quit key ('q') pressed. Initiating shutdown."
                        )
                        self.state = RobotState.ERROR  # Trigger shutdown sequence
                        self.stop_moving()
                        # Request ROS shutdown
                        if rclpy.ok():
                            rclpy.try_shutdown()  # Graceful shutdown request
                except Exception as e:
                    # Prevent display errors from crashing the main loop
                    self.get_logger().error(
                        f"Display update error: {e}", exc_info=False
                    )


# --- Main Function (Minimal Changes) ---
def main(args=None):
    print("Initializing ROS2...")
    rclpy.init(args=args)
    node = None
    executor = None
    print("Creating PancakeRobotNode...")
    try:
        node = PancakeRobotNode()
        # Check node state AFTER initialization attempt
        if node.state not in [
            RobotState.GPIO_ERROR,
            RobotState.CAMERA_ERROR,
            RobotState.ERROR,
            RobotState.AIRTABLE_ERROR,
        ]:
            print("Node initialized successfully. Creating executor...")
            executor = SingleThreadedExecutor()
            executor.add_node(node)
            node.get_logger().info("Starting ROS2 executor spin...")
            print("Spinning node... Press Ctrl+C to exit.")
            executor.spin()  # Blocks until shutdown is called or Ctrl+C
            print("Executor spin finished.")  # Will print after shutdown
        else:
            # Node initialization failed
            node.get_logger().fatal(
                f"Node initialization failed with state: {node.state.name}. Aborting startup."
            )
            # Cleanup might still be needed if some parts initialized
            if node:
                print("Cleaning up hardware after failed initialization...")
                node.cleanup_hardware()
                if node.debug_windows:
                    print("Closing OpenCV windows...")
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)  # Allow windows to close

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received.")
        if node:
            node.get_logger().info("KeyboardInterrupt received, initiating shutdown...")
            node.state = RobotState.ERROR  # Ensure loops exit and cleanup runs
            node.stop_moving()  # Stop motion immediately
    except Exception as e:
        print(f"FATAL Unhandled Error in main execution: {e}")
        if node:
            node.get_logger().fatal(
                f"FATAL Unhandled Error in main: {e}", exc_info=True
            )
            node.state = RobotState.ERROR  # Ensure loops exit and cleanup runs
            node.stop_moving()
    finally:
        print("Initiating final cleanup sequence...")
        if executor:
            print("Shutting down ROS2 executor...")
            # executor.shutdown() # This might already be done if spin exited cleanly
            print("Executor shutdown.")
        if node:
            node.get_logger().info("Ensuring robot is stopped...")
            # node.stop_moving() # Called earlier, but ensure again
            node.get_logger().info("Cleaning up hardware resources...")
            node.cleanup_hardware()
            if node.debug_windows:
                node.get_logger().info("Closing OpenCV windows...")
                cv2.destroyAllWindows()
                cv2.waitKey(50)  # Short delay
            node.get_logger().info("Destroying ROS2 node...")
            if node.is_valid():  # Check if node still exists
                node.destroy_node()
            node.get_logger().info("Node destroyed.")

        # Shutdown ROS 2
        if rclpy.ok():
            print("Shutting down rclpy...")
            rclpy.shutdown()
            print("rclpy shutdown complete.")
        else:
            print("rclpy already shut down.")

        print("Shutdown sequence finished.")


if __name__ == "__main__":
    main()
