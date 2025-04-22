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
AIRTABLE_WHIPPED_CREAM_STATUS_FIELD = "Whipped Cream Status"
AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD = "Choco Chips Status"
AIRTABLE_SPRINKLES_STATUS_FIELD = "Sprinkles Status"
AIRTABLE_PICKUP_STATUS_FIELD = "Pickup Status"

# --- Airtable Status Codes ---
STATUS_WAITING = 0
STATUS_ARRIVED = 1
STATUS_DONE = 99  # Also used to indicate "SKIP" if encountered upon arrival
# Add constants for the intermediate/error statuses that mean "keep waiting"
STATUS_INTERMEDIATE_ERROR_MIN = 2
STATUS_INTERMEDIATE_ERROR_MAX = 7


# --- Map Airtable Fields to Station Indices ---
# Pickup is 0 for color detection purposes, but logically last
STATION_FIELD_TO_INDEX = {
    AIRTABLE_COOKING_1_STATUS_FIELD: 1,
    AIRTABLE_COOKING_2_STATUS_FIELD: 2,
    AIRTABLE_WHIPPED_CREAM_STATUS_FIELD: 3,
    AIRTABLE_SPRINKLES_STATUS_FIELD: 4,
    AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD: 5,
    AIRTABLE_PICKUP_STATUS_FIELD: 0,
}
STATION_INDEX_TO_FIELD = {v: k for k, v in STATION_FIELD_TO_INDEX.items()}

# Define the PHYSICAL sequence of stations the robot encounters
# This is crucial because the color is the same.
# The robot relies on detecting green N times to reach the Nth station.
PHYSICAL_STATION_SEQUENCE_INDICES = [
    1,  # Cooking 1
    2,  # Cooking 2
    3,  # Whipped Cream
    4,  # Sprinkles
    5,  # Choco Chips
    0,  # Pickup
]

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
    for idx in STATION_FIELD_TO_INDEX.values()
}

# --- Navigation & Control Parameters (Using ORIGINAL values as requested) ---
AIRTABLE_POLL_RATE = 2.0
BASE_DRIVE_SPEED = 0.01  # m/s - From original code
BASE_ROTATE_SPEED = 0.2  # rad/s - Speed for *correction* turns - From original code
TURN_FACTOR = 0.7  # Speed multiplier during correction turns - From original code
LOST_LINE_ROTATE_SPEED = 0.15  # rad/s - Speed for *search* rotation - From original code (Negative used for Right turn)
COLOR_DETECTION_THRESHOLD = 2000
COLOR_COOLDOWN_SEC = 5.0  # Cooldown *after successful arrival*
STATION_WAIT_TIMEOUT_SEC = 200.0
LEAVING_STATION_DURATION_SEC = (
    2.5  # Time to drive forward after completing/skipping a station
)


# --- Robot States ---
class RobotState(Enum):
    IDLE = auto()
    FETCHING_ORDER = auto()
    PLANNING_ROUTE = auto()
    LEAVING_STATION = auto()  # State to move forward slightly after finishing/skipping
    MOVING_TO_STATION = auto()
    CHECKING_ARRIVAL_STATUS = (
        auto()
    )  # State to check status immediately on green detect
    ARRIVED_AT_STATION = auto()  # State confirming arrival *requires* processing
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
        self.ordered_station_sequence = []  # Stations required by the *current order*
        self.current_order_station_index = (
            -1
        )  # Index within the *ordered_station_sequence*
        self.target_station_physical_index = -1  # The *next* station index from PHYSICAL_STATION_SEQUENCE_INDICES we expect to see
        self.physical_stations_visited_count = (
            0  # How many green lines have we seen *this trip*
        )
        self.last_color_detection_times = {
            idx: 0.0 for idx in STATION_COLORS_HSV.keys()
        }
        self.wait_start_time = 0.0
        self.leaving_station_start_time = 0.0
        self._last_airtable_check_time = 0.0
        self.initial_line_found = False
        self.picam2 = None
        self.debug_windows = True  # Set to False to disable CV windows
        self.cmd_vel_pub = None
        self.audio_publisher = None
        self.drive_client = None  # Currently unused, Twist is used
        self.rotate_client = None  # Currently unused, Twist is used
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
        # (Same as previous version)
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
            self.picam2.set_controls(
                {"AfMode": controls.AfModeEnum.Continuous, "LensPosition": 0.0}
            )
            self.picam2.start()
            time.sleep(2)
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
        # (Same as previous version)
        try:
            self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
            self.audio_publisher = self.create_publisher(
                AudioNoteVector, "/cmd_audio", 10
            )
            if not self.audio_publisher or not self.cmd_vel_pub:
                raise RuntimeError("Publisher creation failed.")
            self.get_logger().info("ROS2 ok.")
        except Exception as e:
            self.get_logger().error(f"FATAL: ROS2 init: {e}", exc_info=True)
            self.state = RobotState.ERROR

    # --- Airtable Functions ---
    def fetch_order_from_airtable(self):
        self.get_logger().info("Attempting to fetch new order from Airtable...")
        try:
            params = {
                "maxRecords": 1,
                # Find orders where *both* Cooking 1 and Pickup are WAITING (prevents picking up partially done orders)
                "filterByFormula": f"AND({{{AIRTABLE_COOKING_1_STATUS_FIELD}}}={STATUS_WAITING}, {{{AIRTABLE_PICKUP_STATUS_FIELD}}}={STATUS_WAITING})",
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
                self.get_logger().info("No pending orders found in Airtable.")
                return None
            record = records[0]
            record_id = record.get("id")
            fields = record.get("fields", {})
            order_name = fields.get(AIRTABLE_ORDER_NAME_COLUMN)
            if not record_id or not order_name:
                self.get_logger().error(f"Airtable record invalid: {record}")
                return None
            # Fetch status for all relevant fields for this order
            order_data = {
                "record_id": record_id,
                "order_name": order_name,
                "station_status": {
                    field: fields.get(
                        field, STATUS_WAITING
                    )  # Default to WAITING if field is missing
                    for field in STATION_FIELD_TO_INDEX.keys()
                },
            }
            self.get_logger().info(
                f"Fetched order '{order_name}' (ID: {record_id}). Status: {order_data['station_status']}"
            )
            return order_data
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
            self.get_logger().error(
                f"Airtable update error: Missing ID ('{record_id}') or field ('{field}')"
            )
            return False
        if field not in STATION_FIELD_TO_INDEX:
            self.get_logger().error(
                f"Airtable update error: Invalid field name '{field}'"
            )
            return False
        # Ensure status is an integer before sending
        try:
            status_int = int(status)
        except (ValueError, TypeError):
            self.get_logger().error(
                f"Airtable update error: Invalid status type '{status}' for field '{field}'. Must be integer."
            )
            return False

        data = {"fields": {field: status_int}}
        url = f"{AIRTABLE_URL}/{record_id}"
        try:
            response = requests.patch(
                url=url, headers=AIRTABLE_HEADERS, json=data, timeout=10
            )
            response.raise_for_status()
            self.get_logger().info(
                f"Airtable: Updated '{field}' to {status_int} for order {record_id}."
            )
            # Update local cache if successful
            if self.current_order and self.current_order["record_id"] == record_id:
                if field in self.current_order["station_status"]:
                    self.current_order["station_status"][field] = status_int
            return True
        except requests.exceptions.Timeout:
            self.get_logger().error(f"Airtable update timed out for '{field}'.")
            self.state = RobotState.AIRTABLE_ERROR
            return False
        except requests.exceptions.RequestException as e:
            status_code = e.response.status_code if e.response is not None else "N/A"
            response_text = e.response.text if e.response is not None else "N/A"
            self.get_logger().error(
                f"Airtable update error for '{field}': {e} (Status code: {status_code}, Response: {response_text})"
            )
            self.state = RobotState.AIRTABLE_ERROR
            return False
        except Exception as e:
            self.get_logger().error(
                f"Airtable update unexpected error for '{field}': {e}", exc_info=True
            )
            self.state = RobotState.AIRTABLE_ERROR
            return False

    def get_station_status(self, record_id, field):
        # First check local cache if available and matches current order
        if self.current_order and self.current_order["record_id"] == record_id:
            cached_status = self.current_order["station_status"].get(field)
            if cached_status is not None:
                # self.get_logger().debug(f"Airtable get status: Using cached status {cached_status} for '{field}'.")
                return cached_status  # Return cached value directly

        # If not cached or different order, fetch from Airtable
        self.get_logger().debug(
            f"Airtable get status: Fetching fresh status for '{field}' (ID: {record_id})."
        )
        if not record_id or not field:
            self.get_logger().error(
                f"Airtable get status error: Missing ID ('{record_id}') or field ('{field}')"
            )
            return None
        if field not in STATION_FIELD_TO_INDEX:
            self.get_logger().error(
                f"Airtable get status error: Invalid field name '{field}'"
            )
            return None

        url = f"{AIRTABLE_URL}/{record_id}"
        try:
            response = requests.get(url=url, headers=AIRTABLE_HEADERS, timeout=10)
            response.raise_for_status()
            data = response.json()
            current_status = data.get("fields", {}).get(field)

            if current_status is None:
                self.get_logger().warning(
                    f"Airtable field '{field}' not found for record {record_id}. Assuming {STATUS_WAITING}."
                )
                # Update cache with assumed value
                if self.current_order and self.current_order["record_id"] == record_id:
                    self.current_order["station_status"][field] = STATUS_WAITING
                return STATUS_WAITING

            # Ensure status is integer
            try:
                status_int = int(current_status)
                # Update cache with fetched value
                if self.current_order and self.current_order["record_id"] == record_id:
                    self.current_order["station_status"][field] = status_int
                return status_int
            except (ValueError, TypeError):
                self.get_logger().error(
                    f"Airtable get status error: Received non-integer status '{current_status}' for field '{field}'. Treating as error."
                )
                self.state = RobotState.AIRTABLE_ERROR  # Treat invalid data as an error
                return None

        except requests.exceptions.Timeout:
            self.get_logger().warning(
                f"Airtable get status timed out for '{field}'. Returning None."
            )
            return None  # Indicate failure to get status
        except requests.exceptions.RequestException as e:
            self.get_logger().error(
                f"Airtable get status error ({field}): {e}. Returning None."
            )
            # Don't set global error state here, allow retry logic in state machine
            return None  # Indicate failure to get status
        except Exception as e:
            self.get_logger().error(
                f"Airtable get status unexpected error: {e}", exc_info=True
            )
            self.state = RobotState.AIRTABLE_ERROR  # Unexpected errors are more severe
            return None

    # --- Robot Movement and Sensors ---
    def move_robot(self, linear_x, angular_z):
        # (Same as previous version)
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
            if self.cmd_vel_pub:
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)
            return
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        try:
            self.cmd_vel_pub.publish(twist)
        except Exception as e:
            self.get_logger().error(f"Failed to publish Twist: {e}")
            self.state = RobotState.ERROR

    def stop_moving(self):
        # (Same as previous version)
        for _ in range(3):
            self.move_robot(0.0, 0.0)
            time.sleep(0.02)

    def read_ir_sensors(self):
        # (Same as previous version)
        """Reads IR sensors. Returns (left_on_line, right_on_line). HIGH = ON LINE."""
        try:
            left_val = GPIO.input(LEFT_IR_PIN)
            right_val = GPIO.input(RIGHT_IR_PIN)
            return (left_val == IR_LINE_DETECT_SIGNAL), (
                right_val == IR_LINE_DETECT_SIGNAL
            )
        except Exception as e:
            self.get_logger().error(f"IR sensor read error: {e}", exc_info=True)
            self.state = RobotState.GPIO_ERROR
            return False, False

    # --- Color Detection ---
    def check_for_station_color(self, frame):
        # (Same as previous version)
        detected_flag = False
        display_frame = frame.copy()
        mask_frame = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        cv2.putText(
            display_frame,
            f"State: {self.state.name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        expected_station_name = "None"
        if self.target_station_physical_index != -1:
            expected_station_name = STATION_COLORS_HSV.get(
                self.target_station_physical_index, {}
            ).get("name", f"Idx {self.target_station_physical_index}")
        cv2.putText(
            display_frame,
            f"Visits:{self.physical_stations_visited_count} Expect:{expected_station_name}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        hsv_lower = np.array(COMMON_HSV_LOWER)
        hsv_upper = np.array(COMMON_HSV_UPPER)
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
            mask_frame = mask
            pixels = cv2.countNonZero(mask)
            now = time.time()
            last_processed_detection = (
                max(self.last_color_detection_times.values())
                if self.last_color_detection_times
                else 0.0
            )
            if pixels > COLOR_DETECTION_THRESHOLD and (
                now - last_processed_detection > COLOR_COOLDOWN_SEC
            ):
                detected_flag = True
                self.get_logger().debug(
                    f"Detected significant green area ({pixels} px). Potential arrival."
                )
                cv2.rectangle(
                    display_frame,
                    (0, frame.shape[0] - 40),
                    (frame.shape[1], frame.shape[0]),
                    (0, 0, 255),
                    -1,
                )
                cv2.putText(
                    display_frame,
                    "GREEN DETECTED!",
                    (frame.shape[1] // 2 - 100, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            return detected_flag, display_frame, mask_frame
        except cv2.error as e:
            self.get_logger().error(f"OpenCV color processing error: {e}")
            return False, display_frame, np.zeros_like(mask_frame)
        except Exception as e:
            self.get_logger().error(
                f"Color detection unexpected error: {e}", exc_info=True
            )
            return False, display_frame, np.zeros_like(mask_frame)

    # --- Sound ---
    def play_sound(self, notes):
        # (Same as previous version)
        if not hasattr(self, "audio_publisher") or self.audio_publisher is None:
            self.get_logger().warning(
                "Audio publisher not initialized, cannot play sound."
            )
            return
        msg = AudioNoteVector()
        log_str = []
        for freq, duration_ms in notes:
            if freq <= 0 or duration_ms <= 0:
                self.get_logger().warning(
                    f"Skipping invalid audio note: Freq={freq}, Dur={duration_ms}"
                )
                continue
            note = AudioNote()
            note.frequency = int(freq)
            note.max_runtime = Duration(
                sec=int(duration_ms / 1000), nanosec=int((duration_ms % 1000) * 1e6)
            )
            msg.notes.append(note)
            log_str.append(f"({freq}Hz,{duration_ms}ms)")
        if not msg.notes:
            self.get_logger().warning("No valid notes provided to play_sound.")
            return
        self.get_logger().info(f"Playing Audio: {', '.join(log_str)}")
        try:
            self.audio_publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Audio publish failed: {e}", exc_info=True)

    # --- Cleanup ---
    def cleanup_gpio(self):
        # (Same as previous version)
        self.get_logger().info("Cleaning up GPIO...")
        try:
            GPIO.cleanup()
            self.get_logger().info("GPIO cleanup successful.")
        except Exception as e:
            self.get_logger().error(f"GPIO cleanup error: {e}")

    def cleanup_hardware(self):
        # (Same as previous version)
        self.get_logger().info("Cleaning up hardware...")
        if self.picam2:
            try:
                self.get_logger().info("Stopping camera...")
                self.picam2.stop()
                self.get_logger().info("Camera stopped.")
            except Exception as e:
                self.get_logger().error(f"Camera stop error: {e}")
        self.cleanup_gpio()

    # --- Main Control Loop ---
    def control_loop(self):
        """Main state machine and control logic for the robot."""
        if self.state in [
            RobotState.ERROR,
            RobotState.CAMERA_ERROR,
            RobotState.GPIO_ERROR,
            RobotState.AIRTABLE_ERROR,  # Also stop if in Airtable Error state
        ]:
            self.stop_moving()
            return

        display_frame, mask_frame, color_detected = None, None, False
        if self.picam2 and self.state != RobotState.CAMERA_ERROR:
            try:
                raw_frame = self.picam2.capture_array()
                _det, display_frame, mask_frame = self.check_for_station_color(
                    raw_frame
                )
                # Only consider color detection relevant when moving
                if self.state == RobotState.MOVING_TO_STATION:
                    color_detected = _det
            except Exception as e:
                self.get_logger().error(
                    f"Camera capture or processing error: {e}", exc_info=True
                )
                self.state = RobotState.CAMERA_ERROR
                self.stop_moving()
                # No return here, allow display update in finally block

        try:
            current_state = (
                self.state
            )  # Record state at start of loop for error reporting

            # --- State: IDLE ---
            if self.state == RobotState.IDLE:
                self.stop_moving()
                # Reset order-specific variables
                self.current_order = None
                self.ordered_station_sequence = []
                self.current_order_station_index = -1
                self.target_station_physical_index = -1
                self.physical_stations_visited_count = 0
                self.initial_line_found = False
                # Reset cooldowns
                self.last_color_detection_times = {
                    idx: 0.0 for idx in STATION_COLORS_HSV.keys()
                }
                # Attempt to get a new order
                self.current_order = self.fetch_order_from_airtable()
                if self.state == RobotState.AIRTABLE_ERROR:
                    self.get_logger().error(
                        "Airtable error while fetching order in IDLE. Halting. Check connection/API key."
                    )
                    # Don't automatically retry, require restart or external intervention
                elif self.current_order:
                    self.get_logger().info(
                        f"Order '{self.current_order['order_name']}' received. Planning route..."
                    )
                    self.state = RobotState.PLANNING_ROUTE
                else:
                    # No error, but no order found
                    self.state = RobotState.ALL_ORDERS_COMPLETE

            # --- State: ALL_ORDERS_COMPLETE ---
            elif self.state == RobotState.ALL_ORDERS_COMPLETE:
                self.get_logger().info("No pending orders found. Waiting...")
                self.play_sound([(440, 200), (440, 200)])
                self.stop_moving()
                # Wait for a bit before checking again
                time.sleep(5.0)  # Consider making this configurable
                self.state = RobotState.IDLE  # Go back to check for orders

            # --- State: PLANNING_ROUTE ---
            elif self.state == RobotState.PLANNING_ROUTE:
                if not self.current_order:
                    self.get_logger().error(
                        "PLANNING: No current_order data! Returning to IDLE."
                    )
                    self.state = RobotState.IDLE
                    return  # Exit loop for this cycle

                # Build the sequence of *required* stations based on Airtable status = WAITING
                self.ordered_station_sequence = []
                status_dict = self.current_order["station_status"]
                for station_idx in PHYSICAL_STATION_SEQUENCE_INDICES:
                    field_name = STATION_INDEX_TO_FIELD.get(station_idx)
                    # IMPORTANT: Only add stations that are explicitly WAITING (0)
                    if field_name and status_dict.get(field_name) == STATUS_WAITING:
                        self.ordered_station_sequence.append(station_idx)

                if not self.ordered_station_sequence:
                    # This can happen if an order was fetched but all items were already DONE or in another state.
                    self.get_logger().warning(
                        f"Order '{self.current_order['order_name']}' has no stations marked as WAITING (0). Marking order complete and returning to IDLE."
                    )
                    # Optional: Mark pickup as DONE just in case
                    if self.current_order:
                        pickup_field = AIRTABLE_PICKUP_STATUS_FIELD
                        # Check current status before updating unnecessarily
                        pickup_status = self.get_station_status(
                            self.current_order["record_id"], pickup_field
                        )
                        if pickup_status != STATUS_DONE:
                            self.get_logger().info(
                                f"Updating {pickup_field} to DONE as no stations were waiting."
                            )
                            self.update_station_status(
                                self.current_order["record_id"],
                                pickup_field,
                                STATUS_DONE,
                            )
                        else:
                            self.get_logger().info(f"{pickup_field} already DONE.")

                    self.state = RobotState.ORDER_COMPLETE  # Treat as complete
                    return

                # Ensure Pickup (0) is the last station if it's required
                if (
                    0 in self.ordered_station_sequence
                    and self.ordered_station_sequence[-1] != 0
                ):
                    self.get_logger().warning(
                        "Pickup station required but not last in planned sequence. Reordering."
                    )
                    self.ordered_station_sequence.remove(0)
                    self.ordered_station_sequence.append(0)

                # Initialize navigation pointers
                self.current_order_station_index = 0
                self.target_station_physical_index = self.ordered_station_sequence[0]
                self.physical_stations_visited_count = 0
                self.initial_line_found = False  # Reset for new journey

                self.get_logger().info(
                    f"Route planned for order '{self.current_order['order_name']}': {[STATION_INDEX_TO_FIELD.get(idx, f'Idx {idx}') for idx in self.ordered_station_sequence]}"
                )
                self.get_logger().info(
                    f"First target station (physical index): {self.target_station_physical_index} ({STATION_INDEX_TO_FIELD.get(self.target_station_physical_index, 'Unknown')})"
                )
                self.state = RobotState.MOVING_TO_STATION
                self.play_sound([(440, 100), (550, 100), (660, 100)])

            # --- State: MOVING_TO_STATION ---
            elif self.state == RobotState.MOVING_TO_STATION:
                # ** Check for arrival via color detection FIRST **
                if color_detected:
                    self.get_logger().info(
                        "Detected Green Line. Stopping to check status."
                    )
                    self.play_sound([(523, 100), (659, 150)])  # Arrival sound
                    self.stop_moving()

                    # Increment visit count *before* checking status
                    self.physical_stations_visited_count += 1

                    # Update cooldown time for the station just passed
                    if (
                        0
                        < self.physical_stations_visited_count
                        <= len(PHYSICAL_STATION_SEQUENCE_INDICES)
                    ):
                        # Get the index of the station we *just passed*
                        passed_station_idx = PHYSICAL_STATION_SEQUENCE_INDICES[
                            self.physical_stations_visited_count - 1
                        ]
                        current_time = time.time()
                        self.last_color_detection_times[passed_station_idx] = (
                            current_time
                        )
                        self.get_logger().debug(
                            f"Updated cooldown for station index {passed_station_idx} at {current_time}"
                        )
                    else:
                        self.get_logger().warning(
                            f"Visited count {self.physical_stations_visited_count} out of bounds for cooldown update."
                        )

                    # Transition to check the status
                    self.state = RobotState.CHECKING_ARRIVAL_STATUS
                    return  # Exit loop, handle check in next cycle

                # ** Check navigation target validity **
                if (
                    self.target_station_physical_index == -1  # No target set
                    or self.current_order_station_index < 0  # Order index invalid
                    or self.current_order_station_index
                    >= len(self.ordered_station_sequence)  # Order index out of bounds
                ):
                    self.get_logger().error(
                        f"MOVING: Invalid target ({self.target_station_physical_index}) or order index ({self.current_order_station_index}). Halting."
                    )
                    self.stop_moving()
                    self.state = RobotState.ERROR
                    return

                # *** Line Following & Recovery Logic ***
                left_on, right_on = self.read_ir_sensors()
                if self.state == RobotState.GPIO_ERROR:
                    return  # IR read failed, state set, exit

                if not left_on and not right_on:  # === Both OFF line ===
                    # If we haven't found the line initially, keep searching
                    # If we *had* the line and lost it, search
                    if not self.initial_line_found:
                        # Debug log frequency reduced
                        # self.get_logger().info("Searching for initial line (turning RIGHT)...")
                        pass
                    else:
                        # Only log warning if line was previously found
                        self.get_logger().warning(
                            "Line lost! Turning RIGHT to search..."
                        )
                        # self.play_sound([(330, 100)]) # Optional lost sound

                    # Command the right rotation (NEGATIVE speed for right turn)
                    self.move_robot(0.0, -LOST_LINE_ROTATE_SPEED)

                else:  # === At least one sensor ON line ===
                    if not self.initial_line_found:
                        # First time finding the line on this journey segment
                        self.get_logger().info("Initial line found!")
                        self.play_sound([(660, 100)])  # Optional found sound
                        self.initial_line_found = True

                    # --- Normal Line Following (HIGH=ON) ---
                    if left_on and right_on:
                        # Both ON -> Drive straight
                        # self.get_logger().debug("Line Follow: Straight") # Less verbose debug
                        self.move_robot(BASE_DRIVE_SPEED, 0.0)
                    elif left_on and not right_on:
                        # Left ON, Right OFF -> Robot drifted RIGHT -> Turn LEFT to correct
                        # self.get_logger().debug("Line Follow: Correct Left") # Less verbose debug
                        # Turn Left (Positive angular z)
                        self.move_robot(
                            BASE_DRIVE_SPEED * TURN_FACTOR, -BASE_ROTATE_SPEED
                        )
                    elif not left_on and right_on:
                        # Left OFF, Right ON -> Robot drifted LEFT -> Turn RIGHT to correct
                        # self.get_logger().debug("Line Follow: Correct Right") # Less verbose debug
                        # Turn Right (Negative angular z)
                        self.move_robot(
                            BASE_DRIVE_SPEED * TURN_FACTOR, BASE_ROTATE_SPEED
                        )
            # --- End of MOVING_TO_STATION ---

            # --- State: CHECKING_ARRIVAL_STATUS ---
            elif self.state == RobotState.CHECKING_ARRIVAL_STATUS:
                # Validate visit count
                if (
                    self.physical_stations_visited_count <= 0
                    or self.physical_stations_visited_count
                    > len(PHYSICAL_STATION_SEQUENCE_INDICES)
                ):
                    self.get_logger().error(
                        f"CHECKING: Invalid physical_stations_visited_count: {self.physical_stations_visited_count}. Resetting to Error."
                    )
                    self.state = RobotState.ERROR
                    return

                # Determine which physical station we *just arrived at*
                current_physical_station_idx = PHYSICAL_STATION_SEQUENCE_INDICES[
                    self.physical_stations_visited_count - 1
                ]
                station_field = STATION_INDEX_TO_FIELD.get(current_physical_station_idx)
                station_name = (
                    station_field.replace(" Status", "")
                    if station_field
                    else f"Unknown Station {current_physical_station_idx}"
                )
                self.get_logger().info(
                    f"Arrived at physical station {self.physical_stations_visited_count}: {station_name} (Index {current_physical_station_idx})."
                )

                # Is this station actually needed for the current order?
                is_station_required_for_order = (
                    current_physical_station_idx in self.ordered_station_sequence
                    # Check if it's the *specific* one we are currently targeting in the order
                    and self.current_order_station_index
                    < len(self.ordered_station_sequence)
                    and self.ordered_station_sequence[self.current_order_station_index]
                    == current_physical_station_idx
                )

                if not is_station_required_for_order:
                    # We detected a green line, but it's not the station we need *next* for this order.
                    self.get_logger().info(
                        f"Physical station {station_name} is not the *next required* station ({STATION_INDEX_TO_FIELD.get(self.target_station_physical_index, 'Unknown')}). Skipping and moving on."
                    )
                    # Check if we somehow passed the last physical station without finding the target
                    if self.physical_stations_visited_count >= len(
                        PHYSICAL_STATION_SEQUENCE_INDICES
                    ):
                        # This case should ideally not happen if routing is correct
                        self.get_logger().error(
                            f"Passed the last physical station ({station_name}) but haven't completed the order. Target was {STATION_INDEX_TO_FIELD.get(self.target_station_physical_index, 'Unknown')}. Order state error."
                        )
                        self.state = RobotState.ERROR
                    else:
                        # We need to continue to the *next* required station.
                        # The target_station_physical_index should already be set correctly from PLANNING or previous completion.
                        self.get_logger().info(
                            f"Continuing towards target: {STATION_INDEX_TO_FIELD.get(self.target_station_physical_index, 'Unknown')}"
                        )
                        # Move forward slightly to clear the current line before searching again
                        self.state = RobotState.LEAVING_STATION
                        self.leaving_station_start_time = time.time()
                    return  # Exit loop for this cycle

                # --- Station IS the required one ---
                self.get_logger().info(
                    f"Station {station_name} is the required target."
                )
                if (
                    not station_field
                ):  # Should not happen if required check passed, but safety first
                    self.get_logger().error(
                        f"CHECKING: Logic Error! Cannot find field name for required station index {current_physical_station_idx}. Error."
                    )
                    self.state = RobotState.ERROR
                    return

                # Get the status from Airtable (or cache)
                current_airtable_status = self.get_station_status(
                    self.current_order["record_id"], station_field
                )

                if current_airtable_status is None:  # Failed to get status
                    self.get_logger().warning(
                        f"CHECKING: Failed to get Airtable status for {station_field}. Will retry check shortly."
                    )
                    # Don't change state, remain in CHECKING_ARRIVAL_STATUS to retry get_station_status
                    time.sleep(1.0)  # Brief pause before retry
                    return

                # --- Process based on the retrieved Airtable status ---
                if current_airtable_status == STATUS_DONE:
                    # Station required, but already done. Skip it.
                    self.get_logger().info(
                        f"Station {station_name} is required but already marked DONE (99) in Airtable. Skipping."
                    )
                    self.play_sound([(440, 50), (330, 100)])
                    # Advance the order index
                    self.current_order_station_index += 1
                    # Check if order is now complete
                    if self.current_order_station_index >= len(
                        self.ordered_station_sequence
                    ):
                        self.get_logger().info(
                            "Skipped the last required station. Order is now complete."
                        )
                        self.state = RobotState.ORDER_COMPLETE
                    else:
                        # Set the next target station from the order sequence
                        self.target_station_physical_index = (
                            self.ordered_station_sequence[
                                self.current_order_station_index
                            ]
                        )
                        self.get_logger().info(
                            f"Next target station (from order): {STATION_INDEX_TO_FIELD.get(self.target_station_physical_index, 'Unknown')}"
                        )
                        # Leave the current station location
                        self.state = RobotState.LEAVING_STATION
                        self.leaving_station_start_time = time.time()
                    return  # Exit loop for this cycle

                # **** MODIFICATION START ****
                # Check if the status means we should ARRIVE and WAIT:
                # WAITING (0), ARRIVED (1), or an intermediate error code (2-7)
                elif (
                    current_airtable_status == STATUS_WAITING
                    or current_airtable_status == STATUS_ARRIVED
                    or (
                        STATUS_INTERMEDIATE_ERROR_MIN
                        <= current_airtable_status
                        <= STATUS_INTERMEDIATE_ERROR_MAX
                    )
                ):
                    log_msg_base = f"Station {station_name} requires processing."
                    log_level = self.get_logger().info  # Default to info

                    if (
                        STATUS_INTERMEDIATE_ERROR_MIN
                        <= current_airtable_status
                        <= STATUS_INTERMEDIATE_ERROR_MAX
                    ):
                        # Log as warning if it's an intermediate "error" code, but proceed normally
                        log_msg_base = f"Station {station_name} has intermediate status {current_airtable_status}. Robot will wait."
                        log_level = self.get_logger().warning

                    log_level(
                        log_msg_base + " Updating status to ARRIVED (1) in Airtable."
                    )

                    # Regardless of waiting/arrived/intermediate, set status to ARRIVED (1)
                    # This signals the robot is present and waiting for the process.
                    if not self.update_station_status(
                        self.current_order["record_id"], station_field, STATUS_ARRIVED
                    ):
                        # If update fails, go to Airtable Error state
                        self.get_logger().error(
                            f"CHECKING: Failed to update Airtable status to ARRIVED for {station_field}. Setting Airtable error state."
                        )
                        self.state = RobotState.AIRTABLE_ERROR
                        return  # Exit loop

                    # Successfully updated status, now transition to wait
                    self.state = RobotState.ARRIVED_AT_STATION
                    return  # Exit loop for this cycle
                # **** MODIFICATION END ****

                else:  # Handles any other unexpected status code
                    self.get_logger().error(
                        f"CHECKING: Station {station_name} has unexpected status {current_airtable_status} (expected 0, 1, 2-7, or 99). Error."
                    )
                    self.state = RobotState.ERROR
                    return  # Exit loop

            # --- State: ARRIVED_AT_STATION ---
            elif self.state == RobotState.ARRIVED_AT_STATION:
                # This state confirms arrival and prepares for waiting.
                # It primarily ensures the station context is correct and starts the timer.
                if (  # Validate visit count again
                    self.physical_stations_visited_count <= 0
                    or self.physical_stations_visited_count
                    > len(PHYSICAL_STATION_SEQUENCE_INDICES)
                ):
                    self.get_logger().error(
                        f"ARRIVED: Invalid physical_stations_visited_count: {self.physical_stations_visited_count}. Error."
                    )
                    self.state = RobotState.ERROR
                    return

                current_physical_station_idx = PHYSICAL_STATION_SEQUENCE_INDICES[
                    self.physical_stations_visited_count - 1
                ]
                station_field = STATION_INDEX_TO_FIELD.get(current_physical_station_idx)

                if not station_field:  # Safety check
                    self.get_logger().error(
                        f"ARRIVED: Cannot find field name for station index {current_physical_station_idx}. Error."
                    )
                    self.state = RobotState.ERROR
                    return

                # Log confirmation and start the wait timer
                self.get_logger().info(
                    f"Confirmed arrival at {station_field}. Starting wait for DONE (99) status."
                )
                self.play_sound([(440, 100), (440, 100)])  # Arrival confirmation sound
                self.wait_start_time = time.time()  # Start timeout timer
                self._last_airtable_check_time = (
                    0.0  # Ensure immediate check in next state
                )
                self.state = RobotState.WAITING_FOR_STATION_COMPLETION

            # --- State: WAITING_FOR_STATION_COMPLETION ---
            elif self.state == RobotState.WAITING_FOR_STATION_COMPLETION:
                # Validate station context first
                if (
                    self.physical_stations_visited_count <= 0
                    or self.physical_stations_visited_count
                    > len(PHYSICAL_STATION_SEQUENCE_INDICES)
                ):
                    self.get_logger().error(
                        f"WAITING: Invalid physical_stations_visited_count: {self.physical_stations_visited_count}. Error."
                    )
                    self.state = RobotState.ERROR
                    return

                current_physical_station_idx = PHYSICAL_STATION_SEQUENCE_INDICES[
                    self.physical_stations_visited_count - 1
                ]
                station_field = STATION_INDEX_TO_FIELD.get(current_physical_station_idx)

                if not station_field:  # Safety check
                    self.get_logger().error(
                        f"WAITING: Cannot find field name for station index {current_physical_station_idx}. Error."
                    )
                    self.state = RobotState.ERROR
                    return

                # Check for timeout
                elapsed = time.time() - self.wait_start_time
                if elapsed > STATION_WAIT_TIMEOUT_SEC:
                    self.get_logger().warning(
                        f"WAIT TIMEOUT ({elapsed:.1f}s > {STATION_WAIT_TIMEOUT_SEC}s) for station {station_field}. Aborting order."
                    )
                    self.play_sound([(330, 500), (220, 500)])  # Timeout sound
                    self.state = RobotState.STATION_TIMED_OUT
                    return  # Exit loop

                # Check Airtable status periodically based on poll rate
                now = time.time()
                if now - self._last_airtable_check_time >= AIRTABLE_POLL_RATE:
                    self._last_airtable_check_time = now
                    self.get_logger().debug(
                        f"Polling Airtable: Is {station_field} DONE (99) yet?"
                    )
                    current_status = self.get_station_status(
                        self.current_order["record_id"], station_field
                    )

                    if current_status is None:  # Handle failed status check
                        self.get_logger().warning(
                            f"WAITING: Failed to get Airtable status for {station_field}. Continuing wait, will retry polling."
                        )
                        # Stay in this state, loop will retry getting status

                    elif current_status == STATUS_DONE:
                        # Success! Station is complete.
                        self.get_logger().info(
                            f"Station {station_field} reported DONE (99) by Airtable."
                        )
                        self.play_sound([(659, 150), (784, 200)])  # Success sound
                        # Advance the order index
                        self.current_order_station_index += 1
                        # Check if the entire order is now complete
                        if self.current_order_station_index >= len(
                            self.ordered_station_sequence
                        ):
                            self.get_logger().info(
                                "Completed the last required station. Order finished."
                            )
                            self.state = RobotState.ORDER_COMPLETE
                        else:
                            # Set the next target station
                            self.target_station_physical_index = (
                                self.ordered_station_sequence[
                                    self.current_order_station_index
                                ]
                            )
                            self.get_logger().info(
                                f"Proceeding to next target station (from order): {STATION_INDEX_TO_FIELD.get(self.target_station_physical_index, 'Unknown')}"
                            )
                            # Leave the current station
                            self.state = RobotState.LEAVING_STATION
                            self.leaving_station_start_time = time.time()
                        return  # Exit loop

                    # **** MODIFICATION START ****
                    # Check if status means we should KEEP waiting:
                    # ARRIVED (1), WAITING (0), or an intermediate error code (2-7)
                    elif (
                        current_status == STATUS_ARRIVED
                        or current_status == STATUS_WAITING
                        or (
                            STATUS_INTERMEDIATE_ERROR_MIN
                            <= current_status
                            <= STATUS_INTERMEDIATE_ERROR_MAX
                        )
                    ):
                        log_level = self.get_logger().debug  # Default log level
                        log_msg = f"Station {station_field} not yet DONE (Status: {current_status}). Continuing wait."

                        if (
                            STATUS_INTERMEDIATE_ERROR_MIN
                            <= current_status
                            <= STATUS_INTERMEDIATE_ERROR_MAX
                        ):
                            # Log intermediate states as warnings for visibility
                            log_level = self.get_logger().warning
                            log_msg = f"Station {station_field} has intermediate status {current_status}. Continuing wait."

                        log_level(log_msg)
                        # No state change, just continue waiting. The loop will poll again later.
                    # **** MODIFICATION END ****

                    else:  # Handle any other unexpected status received while waiting
                        self.get_logger().error(
                            f"WAITING: Station {station_field} has unexpected status {current_status} (expected 0, 1, 2-7, or 99). Error."
                        )
                        self.state = RobotState.ERROR
                        return  # Exit loop

            # --- State: LEAVING_STATION ---
            elif self.state == RobotState.LEAVING_STATION:
                # Drive forward for a short duration to clear the current station's line marker
                elapsed = time.time() - self.leaving_station_start_time
                if elapsed < LEAVING_STATION_DURATION_SEC:
                    # Move forward slowly
                    self.move_robot(BASE_DRIVE_SPEED * 0.7, 0.0)
                else:
                    # Finished moving forward, stop and prepare for next segment
                    self.get_logger().info(
                        f"Finished leaving station ({LEAVING_STATION_DURATION_SEC}s). Resuming search for next station."
                    )
                    self.stop_moving()
                    self.initial_line_found = False  # Need to re-find the line
                    self.state = RobotState.MOVING_TO_STATION

            # --- State: ORDER_COMPLETE ---
            elif self.state == RobotState.ORDER_COMPLETE:
                # Log completion and play success sound
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

                # Final check: Ensure Pickup status is DONE for the completed order
                pickup_field = AIRTABLE_PICKUP_STATUS_FIELD
                if self.current_order:  # Check if order data exists
                    # Use get_station_status which checks cache first
                    pickup_status = self.get_station_status(
                        self.current_order["record_id"], pickup_field
                    )
                    if pickup_status is None:
                        self.get_logger().warning(
                            f"Could not verify final status of {pickup_field} for {order_name}. Attempting to set DONE."
                        )
                        self.update_station_status(
                            self.current_order["record_id"], pickup_field, STATUS_DONE
                        )
                    elif pickup_status != STATUS_DONE:
                        self.get_logger().info(
                            f"Ensuring {pickup_field} status is set to DONE (99) for completed order {order_name}."
                        )
                        self.update_station_status(
                            self.current_order["record_id"], pickup_field, STATUS_DONE
                        )
                    else:
                        self.get_logger().info(
                            f"{pickup_field} was already DONE for completed order {order_name}."
                        )
                else:
                    self.get_logger().warning(
                        "ORDER_COMPLETE state reached but current_order is None. Cannot verify final Pickup status."
                    )

                # Transition back to IDLE to look for the next order
                self.state = RobotState.IDLE

            # --- State: STATION_TIMED_OUT ---
            elif self.state == RobotState.STATION_TIMED_OUT:
                # Log the timeout event
                order_name = (
                    self.current_order.get("order_name", "Unknown Order")
                    if self.current_order
                    else "Unknown Order"
                )
                timed_out_station_idx = -1
                timed_out_field = "Unknown"
                # Try to identify the station that timed out
                if (
                    0
                    < self.physical_stations_visited_count
                    <= len(PHYSICAL_STATION_SEQUENCE_INDICES)
                ):
                    timed_out_station_idx = PHYSICAL_STATION_SEQUENCE_INDICES[
                        self.physical_stations_visited_count - 1
                    ]
                    timed_out_field = STATION_INDEX_TO_FIELD.get(
                        timed_out_station_idx, "Unknown"
                    )
                self.get_logger().error(
                    f"STATION TIMED OUT waiting for {timed_out_field} (Physical Idx: {timed_out_station_idx}). Aborting order '{order_name}'."
                )
                self.stop_moving()

                # Reset order-specific data and return to IDLE
                # Consider adding logic here to potentially mark the order as failed in Airtable if needed
                self.current_order = None  # Discard the failed order
                self.state = RobotState.IDLE  # Go back to look for new orders

            # --- State: AIRTABLE_ERROR ---
            elif self.state == RobotState.AIRTABLE_ERROR:
                # A persistent Airtable error occurred (e.g., bad API key, network issue, update failed)
                self.get_logger().error(
                    "Persistent AIRTABLE ERROR detected. Robot requires intervention. System halted."
                )
                self.play_sound([(330, 300), (330, 300), (330, 300)])  # Error sound
                self.stop_moving()
                # The loop will exit naturally as the state check at the top will stop movement

        except Exception as e:
            # Catch any unhandled exceptions during state processing
            self.get_logger().fatal(
                f"!!! Unhandled exception in state {current_state.name}: {e} !!!",
                exc_info=True,  # Log the full traceback
            )
            self.state = RobotState.ERROR  # Set error state
            self.stop_moving()

        finally:
            # --- Display Update (Executed every loop iteration) ---
            if self.debug_windows:
                try:
                    # Display camera feed if available
                    if display_frame is not None:
                        cv2.imshow("Camera Feed", display_frame)
                    else:  # Show blank feed if frame is missing
                        blank_feed = np.zeros(
                            (CAMERA_RESOLUTION[1], CAMERA_RESOLUTION[0], 3),
                            dtype=np.uint8,
                        )
                        cv2.putText(
                            blank_feed,
                            f"State: {self.state.name}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                        )
                        cv2.putText(
                            blank_feed,
                            "No Camera Frame / Camera Error?",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                        )
                        cv2.imshow("Camera Feed", blank_feed)

                    # Display mask frame if available
                    if mask_frame is not None:
                        # Ensure mask is 3-channel for display if it's grayscale
                        mask_display = (
                            cv2.cvtColor(mask_frame, cv2.COLOR_GRAY2BGR)
                            if len(mask_frame.shape) == 2
                            else mask_frame
                        )
                        cv2.imshow("Color Detection Mask", mask_display)
                    else:  # Show blank mask if missing
                        blank_mask = np.zeros(
                            (CAMERA_RESOLUTION[1], CAMERA_RESOLUTION[0], 3),
                            dtype=np.uint8,
                        )
                        cv2.putText(
                            blank_mask,
                            "No Mask Frame",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                        )
                        cv2.imshow("Color Detection Mask", blank_mask)

                    # Check for quit key ('q')
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        self.get_logger().info(
                            "Quit key ('q') pressed in OpenCV window. Initiating shutdown."
                        )
                        self.state = (
                            RobotState.ERROR
                        )  # Treat 'q' as an error shutdown trigger
                        self.stop_moving()
                        rclpy.try_shutdown()  # Request ROS shutdown

                except Exception as e:
                    # Catch errors specifically from the display update
                    self.get_logger().error(
                        f"Display update error: {e}",
                        exc_info=False,  # Don't need full traceback usually
                    )
                    # Consider disabling debug windows if this happens frequently
                    # self.debug_windows = False


# --- Main Function ---
def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = None
    exit_code = 0
    try:
        node = PancakeRobotNode()
        # Check for critical init failures
        if (
            node.state
            in [
                RobotState.GPIO_ERROR,
                RobotState.CAMERA_ERROR,
                RobotState.ERROR,  # General init error
                RobotState.AIRTABLE_ERROR,  # Added: Airtable failure during init (e.g., bad credentials)
            ]
        ):
            node.get_logger().fatal(
                f"Node initialization failed critically with state: {node.state.name}. Aborting execution."
            )
            exit_code = 1
            # Perform minimal cleanup possible without assuming full initialization
            if node:
                node.cleanup_hardware()  # Try to clean up GPIO/Camera if partially initialized
                if node.debug_windows:
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                if not node.is_destroyed:
                    node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            return exit_code  # Exit early

        # If initialization was successful, proceed with executor
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        node.get_logger().info("Starting ROS2 executor spin...")

        # Main execution loop - Spin as long as ROS is OK and the node is not in a fatal error state
        while rclpy.ok() and node.state not in [
            RobotState.ERROR,
            RobotState.AIRTABLE_ERROR,  # Stop spinning on persistent Airtable error
            RobotState.GPIO_ERROR,  # Stop spinning on GPIO error
            RobotState.CAMERA_ERROR,  # Stop spinning on Camera error
        ]:
            executor.spin_once(timeout_sec=0.1)  # Process callbacks, timers

            # Check state again after spin_once, in case a callback set an error state
            if node.state in [
                RobotState.ERROR,
                RobotState.AIRTABLE_ERROR,
                RobotState.GPIO_ERROR,
                RobotState.CAMERA_ERROR,
            ]:
                node.get_logger().error(
                    f"Node entered fatal error state ({node.state.name}) during spin. Shutting down executor."
                )
                exit_code = 1
                break  # Exit the while loop

        # If loop exited because rclpy is not ok (e.g., Ctrl+C)
        if not rclpy.ok():
            node.get_logger().info("RCLPY shutdown requested. Exiting spin loop.")
        else:
            node.get_logger().info(
                f"Spin loop exited due to node state: {node.state.name}"
            )

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received.")
        if node:
            node.get_logger().info(
                "KeyboardInterrupt: Initiating controlled shutdown..."
            )
            node.state = (
                RobotState.ERROR
            )  # Set error state to ensure cleanup happens correctly
        exit_code = 0  # Typically exit code 0 for user interrupt
    except Exception as e:
        print(f"\nFATAL Unhandled Error in main execution scope: {e}")
        if node:
            node.get_logger().fatal(
                f"FATAL Unhandled Error in main: {e}", exc_info=True
            )
            node.state = RobotState.ERROR  # Ensure error state is set
        exit_code = 1  # Non-zero exit code for unexpected errors
    finally:
        print("--- Initiating Final Cleanup Sequence ---")
        if node:
            node.get_logger().info("Stopping robot movement (final command)...")
            try:
                # Attempt to send stop command even if node is in error state
                node.stop_moving()
                time.sleep(0.2)  # Allow time for command to be sent/processed
            except Exception as stop_err:
                node.get_logger().error(f"Error during final stop command: {stop_err}")

            if executor:
                node.get_logger().info("Shutting down ROS2 executor...")
                try:
                    executor.shutdown()
                except Exception as exec_shut_err:
                    node.get_logger().error(
                        f"Error shutting down executor: {exec_shut_err}"
                    )

            node.get_logger().info("Cleaning up hardware resources (GPIO, Camera)...")
            node.cleanup_hardware()  # Call hardware cleanup

            if node.debug_windows:
                node.get_logger().info("Closing any remaining OpenCV windows...")
                try:
                    cv2.destroyAllWindows()
                    # Add multiple waitKeys to help ensure windows close on all systems
                    cv2.waitKey(1)
                    cv2.waitKey(1)
                    cv2.waitKey(1)
                except Exception as cv_err:
                    node.get_logger().error(f"Error closing OpenCV windows: {cv_err}")

            # Check if node exists and is not already destroyed before destroying
            if node and not node.is_destroyed:
                node.get_logger().info("Destroying ROS2 node instance...")
                try:
                    node.destroy_node()
                except Exception as node_destroy_err:
                    # Log error but continue shutdown
                    node.get_logger().error(
                        f"Error destroying node: {node_destroy_err}"
                    )

        # Final ROS cleanup
        if rclpy.ok():
            print("Shutting down rclpy context...")
            try:
                rclpy.shutdown()
            except Exception as rclpy_shut_err:
                print(
                    f"Error during final rclpy shutdown: {rclpy_shut_err}"
                )  # Use print as logger might be gone

        print(f"--- Shutdown sequence finished. Exiting with code: {exit_code} ---")
        # Explicitly exit with the determined code
        # sys.exit(exit_code) # Uncomment if you want to force exit code in scripts


if __name__ == "__main__":
    main()
