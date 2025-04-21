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

# --- Map Airtable Fields to Station Indices ---
# Pickup is 0 for color detection purposes, but logically last
STATION_FIELD_TO_INDEX = {
    AIRTABLE_COOKING_1_STATUS_FIELD: 1,
    AIRTABLE_COOKING_2_STATUS_FIELD: 2,
    AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD: 3,  # Assuming Choco Chips is physically after Cooking 2
    AIRTABLE_WHIPPED_CREAM_STATUS_FIELD: 4,  # Assuming Whipped Cream is after Choco Chips
    AIRTABLE_SPRINKLES_STATUS_FIELD: 5,  # Assuming Sprinkles is after Whipped Cream
    AIRTABLE_PICKUP_STATUS_FIELD: 0,  # Pickup is physically last, uses index 0 for color
}
STATION_INDEX_TO_FIELD = {v: k for k, v in STATION_FIELD_TO_INDEX.items()}

# Define the PHYSICAL sequence of stations the robot encounters
# This is crucial because the color is the same.
# The robot relies on detecting green N times to reach the Nth station.
PHYSICAL_STATION_SEQUENCE_INDICES = [
    1,  # Cooking 1
    2,  # Cooking 2
    3,  # Choco Chips
    4,  # Whipped Cream
    5,  # Sprinkles
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

# --- Navigation & Control Parameters ---
AIRTABLE_POLL_RATE = 2.0
BASE_DRIVE_SPEED = 0.01  # m/s - Tune as needed - SLOWED DOWN FOR TESTING
BASE_ROTATE_SPEED = 0.2  # rad/s - Speed for *correction* turns (tune as needed)
TURN_FACTOR = 0.7  # Speed multiplier during correction turns
LOST_LINE_ROTATE_SPEED = 0.15  # rad/s - Speed for *search* rotation (tune as needed) - USE NEGATIVE FOR RIGHT TURN
COLOR_DETECTION_THRESHOLD = 2000
COLOR_COOLDOWN_SEC = 5.0  # Cooldown *after successful arrival*
STATION_WAIT_TIMEOUT_SEC = 120.0
LEAVING_STATION_DURATION_SEC = (
    2.0  # Time to drive forward after completing/skipping a station
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
    )  # NEW state: check status immediately on green detect
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
        try:
            self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
            self.audio_publisher = self.create_publisher(
                AudioNoteVector, "/cmd_audio", 10
            )
            # Action clients might be useful later but aren't used for line following
            # self.drive_client = ActionClient(self, DriveDistance, "/drive_distance")
            # self.rotate_client = ActionClient(self, RotateAngle, "/rotate_angle")
            if not self.audio_publisher or not self.cmd_vel_pub:
                raise RuntimeError("Publisher creation failed.")
            self.get_logger().info("ROS2 ok.")
        except Exception as e:
            self.get_logger().error(f"FATAL: ROS2 init: {e}", exc_info=True)
            self.state = RobotState.ERROR

    # --- Airtable Functions ---
    def fetch_order_from_airtable(self):
        # Fetches the next available order (Cooking 1 = 0, Pickup = 0)
        self.get_logger().info("Attempting to fetch new order from Airtable...")
        try:
            params = {
                "maxRecords": 1,
                # Ensure we only grab orders that haven't started and aren't finished pickup
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
                return None  # No orders waiting

            record = records[0]
            record_id = record.get("id")
            fields = record.get("fields", {})
            order_name = fields.get(AIRTABLE_ORDER_NAME_COLUMN)

            if not record_id or not order_name:
                self.get_logger().error(f"Airtable record invalid: {record}")
                return None

            # Store the full status details
            order_data = {
                "record_id": record_id,
                "order_name": order_name,
                "station_status": {
                    field: fields.get(
                        field, STATUS_WAITING
                    )  # Default to WAITING if field missing
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
        # Updates the status of a specific field for a given record ID
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

        data = {"fields": {field: status}}
        url = f"{AIRTABLE_URL}/{record_id}"
        try:
            response = requests.patch(
                url=url, headers=AIRTABLE_HEADERS, json=data, timeout=10
            )
            response.raise_for_status()  # Raise exception for bad status codes (4xx or 5xx)
            self.get_logger().info(
                f"Airtable: Updated '{field}' to {status} for order {record_id}."
            )
            return True
        except requests.exceptions.Timeout:
            self.get_logger().error(f"Airtable update timed out for '{field}'.")
            self.state = RobotState.AIRTABLE_ERROR
            return False
        except requests.exceptions.RequestException as e:
            self.get_logger().error(
                f"Airtable update error for '{field}': {e} (Status code: {e.response.status_code if e.response else 'N/A'})"
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
        # Gets the current status of a specific field for a given record ID
        # Returns status code (int) or None if error occurs
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
            # self.get_logger().debug(f"Airtable get status '{field}': Status is {current_status}") # DEBUG
            if current_status is None:
                self.get_logger().warning(
                    f"Airtable field '{field}' not found for record {record_id}. Assuming {STATUS_WAITING}."
                )
                return STATUS_WAITING  # Or handle as error? Defaulting to 0 for now.
            return int(current_status)  # Ensure it's an integer
        except requests.exceptions.Timeout:
            self.get_logger().warning(f"Airtable get status timed out for '{field}'.")
            # Don't set global error state here, allow retry
            return None  # Indicate error occurred during this check
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Airtable get status error ({field}): {e}")
            # Don't set global error state here, allow retry
            return None  # Indicate error occurred during this check
        except Exception as e:
            self.get_logger().error(
                f"Airtable get status unexpected error: {e}", exc_info=True
            )
            # Don't set global error state here, allow retry
            return None  # Indicate error occurred during this check

    # --- Robot Movement and Sensors ---
    def move_robot(self, linear_x, angular_z):
        # Publishes Twist messages to control robot movement
        if (
            self.state
            in [
                RobotState.ERROR,
                RobotState.CAMERA_ERROR,
                RobotState.GPIO_ERROR,
                RobotState.AIRTABLE_ERROR,  # Stop if persistent Airtable error
            ]
            or not self.cmd_vel_pub
        ):
            # Ensure robot stops if in an error state
            if self.cmd_vel_pub:
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)
            return  # Don't send movement commands in error states

        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        # self.get_logger().debug(f"Publish Twist: Lin={linear_x:.3f}, Ang={angular_z:.3f}")
        try:
            self.cmd_vel_pub.publish(twist)
        except Exception as e:
            self.get_logger().error(f"Failed to publish Twist: {e}")
            # Should we set an error state here? Maybe ROS is down.
            self.state = RobotState.ERROR  # Let's assume this is critical

    def stop_moving(self):
        # Sends stop commands repeatedly for robustness
        # self.get_logger().debug("Stopping movement...")
        for _ in range(3):  # Send multiple times
            self.move_robot(0.0, 0.0)
            time.sleep(0.02)  # Small delay between commands

    def read_ir_sensors(self):
        """Reads IR sensors. Returns (left_on_line, right_on_line). HIGH = ON LINE."""
        try:
            left_val = GPIO.input(LEFT_IR_PIN)
            right_val = GPIO.input(RIGHT_IR_PIN)
            # self.get_logger().debug(f"IR Raw: L={left_val}, R={right_val}") # DEBUG
            return (left_val == IR_LINE_DETECT_SIGNAL), (
                right_val == IR_LINE_DETECT_SIGNAL
            )
        except Exception as e:
            self.get_logger().error(f"IR sensor read error: {e}", exc_info=True)
            self.state = RobotState.GPIO_ERROR  # Set error state on read failure
            return False, False

    # --- Color Detection ---
    def check_for_station_color(self, frame):
        # Detects the common green color, returns flag and annotated frames
        detected_flag = False
        display_frame = frame.copy()
        mask_frame = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        # Basic state info on frame
        cv2.putText(
            display_frame,
            f"State: {self.state.name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        # Show expected *next* physical station index
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
        color_bgr = COMMON_COLOR_BGR

        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
            mask_frame = mask  # Keep the mask for display/debugging
            pixels = cv2.countNonZero(mask)

            # Display pixel count for tuning
            # cv2.putText(
            #     display_frame, f"Green Pixels: {pixels}",
            #     (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)

            now = time.time()
            # Use a general cooldown based on the *last time any* green was detected and processed
            # Find the most recent detection time across all station indices
            last_processed_detection = (
                max(self.last_color_detection_times.values())
                if self.last_color_detection_times
                else 0.0
            )

            if pixels > COLOR_DETECTION_THRESHOLD and (
                now - last_processed_detection > COLOR_COOLDOWN_SEC
            ):
                # Don't update last_color_detection_times here yet. Update it only when
                # we confirm arrival at CHECKING_ARRIVAL_STATUS or ARRIVED_AT_STATION.
                detected_flag = True
                self.get_logger().debug(
                    f"Detected significant green area ({pixels} px). Potential arrival."
                )
                # Visual indicator on frame
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
            return (
                False,
                display_frame,
                np.zeros_like(mask_frame),
            )  # Return blank mask on error
        except Exception as e:
            self.get_logger().error(
                f"Color detection unexpected error: {e}", exc_info=True
            )
            return False, display_frame, np.zeros_like(mask_frame)

    # --- Sound ---
    def play_sound(self, notes):
        # Plays a sequence of audio notes
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
        self.get_logger().info("Cleaning up GPIO...")
        try:
            GPIO.cleanup()
            self.get_logger().info("GPIO cleanup successful.")
        except Exception as e:
            self.get_logger().error(f"GPIO cleanup error: {e}")

    def cleanup_hardware(self):
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
        # --- Pre-State Machine Checks ---
        if self.state in [
            RobotState.ERROR,
            RobotState.CAMERA_ERROR,
            RobotState.GPIO_ERROR,
            # Note: AIRTABLE_ERROR is handled within states now, allowing retries
        ]:
            self.stop_moving()  # Ensure stopped in critical error states
            # No return here, allow loop to run for potential recovery or shutdown?
            # For now, let's halt activity.
            return

        # --- Camera Frame Processing ---
        display_frame, mask_frame, color_detected = None, None, False
        if self.picam2 and self.state != RobotState.CAMERA_ERROR:
            try:
                raw_frame = self.picam2.capture_array()
                # Perform color detection regardless of state for display purposes,
                # but only use the 'color_detected' flag in relevant states.
                _det, display_frame, mask_frame = self.check_for_station_color(
                    raw_frame
                )
                # Only consider color detected if we are actually looking for a station
                if self.state == RobotState.MOVING_TO_STATION:
                    color_detected = _det

            except Exception as e:
                self.get_logger().error(
                    f"Camera capture or processing error: {e}", exc_info=True
                )
                self.state = RobotState.CAMERA_ERROR
                self.stop_moving()
                # Display update will likely fail too, handled in finally block

        # --- State Machine Logic ---
        try:
            current_state = self.state  # Store state at start of cycle

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
                self.last_color_detection_times = {
                    idx: 0.0 for idx in STATION_COLORS_HSV.keys()
                }  # Reset cooldowns

                # Try fetching a new order
                self.current_order = self.fetch_order_from_airtable()

                if self.state == RobotState.AIRTABLE_ERROR:
                    self.get_logger().error(
                        "Airtable error while fetching order in IDLE. Will retry."
                    )
                    # Stay in IDLE, maybe add a longer delay?
                    time.sleep(5.0)  # Wait before retrying fetch

                elif self.current_order:
                    self.get_logger().info(
                        f"Order '{self.current_order['order_name']}' received. Planning route..."
                    )
                    self.state = RobotState.PLANNING_ROUTE
                else:
                    # No error and no order means all done for now
                    self.state = RobotState.ALL_ORDERS_COMPLETE

            # --- State: ALL_ORDERS_COMPLETE ---
            elif self.state == RobotState.ALL_ORDERS_COMPLETE:
                self.get_logger().info("No pending orders found. Waiting...")
                self.play_sound([(440, 200), (440, 200)])
                self.stop_moving()
                time.sleep(5.0)  # Wait for a bit before checking again
                self.state = RobotState.IDLE  # Go back to check for new orders

            # --- State: PLANNING_ROUTE ---
            elif self.state == RobotState.PLANNING_ROUTE:
                if not self.current_order:
                    self.get_logger().error(
                        "PLANNING: No current_order data! Returning to IDLE."
                    )
                    self.state = RobotState.IDLE
                    return  # Exit loop for this cycle

                # Build the sequence of *required* stations for THIS order
                self.ordered_station_sequence = []
                status_dict = self.current_order["station_status"]

                # Iterate through the PHYSICAL sequence and add to order if status is WAITING
                for station_idx in PHYSICAL_STATION_SEQUENCE_INDICES:
                    field_name = STATION_INDEX_TO_FIELD.get(station_idx)
                    if field_name and status_dict.get(field_name) == STATUS_WAITING:
                        self.ordered_station_sequence.append(station_idx)

                # Validate the plan
                if not self.ordered_station_sequence:
                    self.get_logger().error(
                        f"Order '{self.current_order['order_name']}' has no stations marked as WAITING. Assuming complete or error. Returning to IDLE."
                    )
                    # Maybe update pickup to DONE? For now, just go idle.
                    # self.update_station_status(self.current_order['record_id'], AIRTABLE_PICKUP_STATUS_FIELD, STATUS_DONE) # Optional
                    self.state = RobotState.IDLE
                    self.current_order = None  # Clear the problematic order
                    return

                # Ensure Pickup (index 0) is always the last *required* station if present
                if (
                    0 in self.ordered_station_sequence
                    and self.ordered_station_sequence[-1] != 0
                ):
                    self.get_logger().warning(
                        "Pickup station is required but not last in planned sequence. Reordering."
                    )
                    self.ordered_station_sequence.remove(0)
                    self.ordered_station_sequence.append(0)

                # Initialize tracking for the first station
                self.current_order_station_index = 0
                # The target physical station is the first one in our order list
                self.target_station_physical_index = self.ordered_station_sequence[0]
                self.physical_stations_visited_count = (
                    0  # Reset visit count for the new trip
                )
                self.initial_line_found = False  # Need to find the line again

                self.get_logger().info(
                    f"Route planned for order '{self.current_order['order_name']}': {self.ordered_station_sequence}"
                )
                self.get_logger().info(
                    f"First target station (physical index): {self.target_station_physical_index}"
                )

                self.state = RobotState.MOVING_TO_STATION
                self.play_sound([(440, 100), (550, 100), (660, 100)])

            # --- State: MOVING_TO_STATION ---
            elif self.state == RobotState.MOVING_TO_STATION:
                # **Primary Check: Color Detection**
                if color_detected:
                    self.get_logger().info(
                        f"Detected Green Line. Stopping to check status."
                    )
                    self.play_sound([(523, 100), (659, 150)])  # Arrival sound
                    self.stop_moving()
                    # Increment the count of physical stations visited *now*
                    self.physical_stations_visited_count += 1
                    # Update the color detection cooldown timestamp for the detected station index
                    # Which index was it? The one we *expected* based on count.
                    expected_station_idx = PHYSICAL_STATION_SEQUENCE_INDICES[
                        self.physical_stations_visited_count - 1
                    ]
                    self.last_color_detection_times[expected_station_idx] = time.time()

                    self.state = RobotState.CHECKING_ARRIVAL_STATUS
                    return  # Exit loop, handle check in next cycle

                # **Secondary Check: Target Validity**
                if (
                    self.target_station_physical_index == -1
                    or self.current_order_station_index < 0
                ):
                    self.get_logger().error(
                        f"MOVING: Invalid target ({self.target_station_physical_index}) or order index ({self.current_order_station_index}). Halting."
                    )
                    self.stop_moving()
                    self.state = RobotState.ERROR
                    return

                # --- Line Following Logic ---
                left_on, right_on = self.read_ir_sensors()
                if self.state == RobotState.GPIO_ERROR:
                    return  # IR read failed

                if not left_on and not right_on:  # === Both OFF line ===
                    if not self.initial_line_found:
                        # self.get_logger().debug("Searching for initial line (turning RIGHT)...") # Less verbose
                        self.move_robot(
                            0.0, -LOST_LINE_ROTATE_SPEED
                        )  # Turn RIGHT (negative Z)
                    else:
                        self.get_logger().warning(
                            "Line lost! Turning RIGHT to search..."
                        )
                        # self.play_sound([(330, 100)]) # Optional lost sound
                        self.move_robot(0.0, -LOST_LINE_ROTATE_SPEED)  # Turn RIGHT

                else:  # === At least one sensor ON line ===
                    if not self.initial_line_found:
                        self.get_logger().info("Initial line found!")
                        self.play_sound([(660, 100)])  # Found sound
                        self.initial_line_found = True

                    # --- Normal Line Following ---
                    if left_on and right_on:  # Both ON -> Straight
                        # self.get_logger().debug("Line Follow: Straight")
                        self.move_robot(BASE_DRIVE_SPEED, 0.0)
                    elif left_on and not right_on:  # Left ON only -> Turn LEFT
                        # self.get_logger().debug("Line Follow: Correct Left")
                        self.move_robot(
                            BASE_DRIVE_SPEED * TURN_FACTOR, BASE_ROTATE_SPEED
                        )  # Positive Z is Left? Check Create3 docs
                    elif not left_on and right_on:  # Right ON only -> Turn RIGHT
                        # self.get_logger().debug("Line Follow: Correct Right")
                        self.move_robot(
                            BASE_DRIVE_SPEED * TURN_FACTOR, -BASE_ROTATE_SPEED
                        )  # Negative Z is Right?

            # --- State: CHECKING_ARRIVAL_STATUS ---
            elif self.state == RobotState.CHECKING_ARRIVAL_STATUS:
                # We stopped because we detected green. Now, figure out which station we *think* we are at.
                # This is based on how many green lines we've passed *this trip*.
                if (
                    self.physical_stations_visited_count <= 0
                    or self.physical_stations_visited_count
                    > len(PHYSICAL_STATION_SEQUENCE_INDICES)
                ):
                    self.get_logger().error(
                        f"CHECKING: Invalid physical_stations_visited_count: {self.physical_stations_visited_count}. Resetting."
                    )
                    self.state = RobotState.ERROR  # Or try to recover?
                    return

                # Get the index of the station we just physically arrived at
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

                # Is this station actually needed for the *current order*?
                is_station_required_for_order = (
                    current_physical_station_idx in self.ordered_station_sequence
                )

                if not is_station_required_for_order:
                    self.get_logger().info(
                        f"Station {station_name} not required for this order. Skipping and moving on."
                    )
                    # We don't need to increment current_order_station_index here.
                    # We just need to set the *next* physical target.
                    if self.physical_stations_visited_count >= len(
                        PHYSICAL_STATION_SEQUENCE_INDICES
                    ):
                        self.get_logger().error(
                            "Passed the last physical station but didn't find required target? Error."
                        )
                        self.state = RobotState.ERROR  # Should have found pickup by now
                    else:
                        # Update the *next* expected physical station index
                        self.target_station_physical_index = (
                            PHYSICAL_STATION_SEQUENCE_INDICES[
                                self.physical_stations_visited_count
                            ]
                        )
                        self.get_logger().info(
                            f"Next expected physical station: {STATION_INDEX_TO_FIELD.get(self.target_station_physical_index, 'Unknown')}"
                        )
                        self.state = (
                            RobotState.LEAVING_STATION
                        )  # Briefly move forward off the line
                        self.leaving_station_start_time = time.time()
                    return

                # --- This station IS required for the order ---
                # Now check its CURRENT status in Airtable
                if not station_field:
                    self.get_logger().error(
                        f"CHECKING: Cannot find field name for required station index {current_physical_station_idx}. Error."
                    )
                    self.state = RobotState.ERROR
                    return

                current_airtable_status = self.get_station_status(
                    self.current_order["record_id"], station_field
                )

                if current_airtable_status is None:
                    self.get_logger().warning(
                        f"CHECKING: Failed to get Airtable status for {station_field}. Will retry check shortly."
                    )
                    # Stay in this state, maybe add a small delay?
                    time.sleep(1.0)
                    return  # Retry getting status next cycle

                # --- Process based on Airtable status ---
                if current_airtable_status == STATUS_DONE:
                    # Station is required, but ALREADY marked DONE (e.g., 99). Skip it.
                    self.get_logger().info(
                        f"Station {station_name} is required but already marked DONE in Airtable. Skipping."
                    )
                    self.play_sound([(440, 50), (330, 100)])  # Skip sound

                    # Increment the index for the *order sequence*
                    self.current_order_station_index += 1

                    # Check if that was the last station for the ORDER
                    if self.current_order_station_index >= len(
                        self.ordered_station_sequence
                    ):
                        self.get_logger().info(
                            "Skipped the last required station. Order should be complete."
                        )
                        # This case implies the last station (e.g. Pickup) was already 99.
                        self.state = RobotState.ORDER_COMPLETE
                    else:
                        # Set the next target station based on the ORDER sequence
                        self.target_station_physical_index = (
                            self.ordered_station_sequence[
                                self.current_order_station_index
                            ]
                        )
                        self.get_logger().info(
                            f"Next target station (from order): {STATION_INDEX_TO_FIELD.get(self.target_station_physical_index, 'Unknown')}"
                        )
                        # Now, find this target in the physical sequence to know which *physical* station to expect next
                        try:
                            next_physical_target_list_index = (
                                PHYSICAL_STATION_SEQUENCE_INDICES.index(
                                    self.target_station_physical_index,
                                    self.physical_stations_visited_count,
                                )
                            )
                        except ValueError:
                            self.get_logger().error(
                                f"Could not find next target {self.target_station_physical_index} in physical sequence after index {self.physical_stations_visited_count}. Error."
                            )
                            self.state = RobotState.ERROR
                            return

                        self.get_logger().info(
                            f"Continuing movement, expecting physical station {next_physical_target_list_index + 1}"
                        )
                        self.state = (
                            RobotState.LEAVING_STATION
                        )  # Move off the current line
                        self.leaving_station_start_time = time.time()
                    return

                elif (
                    current_airtable_status == STATUS_WAITING
                    or current_airtable_status == STATUS_ARRIVED
                ):
                    # Station is required and is WAITING (or already marked as ARRIVED). Process it.
                    self.get_logger().info(
                        f"Station {station_name} requires processing. Updating status to ARRIVED."
                    )
                    # Update status to ARRIVED (even if it was already 1, doesn't hurt)
                    if not self.update_station_status(
                        self.current_order["record_id"], station_field, STATUS_ARRIVED
                    ):
                        self.get_logger().error(
                            f"CHECKING: Failed to update Airtable status to ARRIVED for {station_field}. Setting Airtable error state."
                        )
                        # Don't transition, let the main AIRTABLE_ERROR handler kick in if persistent
                        self.state = RobotState.AIRTABLE_ERROR
                        return

                    # Transition to the actual waiting state
                    self.state = RobotState.ARRIVED_AT_STATION  # Confirmation state
                    return

                else:
                    # Unexpected status value
                    self.get_logger().error(
                        f"CHECKING: Station {station_name} has unexpected status {current_airtable_status}. Error."
                    )
                    self.state = RobotState.ERROR
                    return

            # --- State: ARRIVED_AT_STATION ---
            # This state now simply confirms we need to wait. The checks were done in CHECKING_ARRIVAL_STATUS.
            elif self.state == RobotState.ARRIVED_AT_STATION:
                # We already updated Airtable to ARRIVED in the previous state.
                # Get the station details again for logging/waiting.
                if (
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
                if not station_field:
                    self.get_logger().error(
                        f"ARRIVED: Cannot find field name for station index {current_physical_station_idx}. Error."
                    )
                    self.state = RobotState.ERROR
                    return

                self.get_logger().info(
                    f"Confirmed arrival at {station_field}. Starting wait for DONE status."
                )
                self.play_sound([(440, 100), (440, 100)])  # Waiting sound
                self.wait_start_time = time.time()  # Start the timeout timer
                self._last_airtable_check_time = 0.0  # Reset check timer
                self.state = RobotState.WAITING_FOR_STATION_COMPLETION

            # --- State: WAITING_FOR_STATION_COMPLETION ---
            elif self.state == RobotState.WAITING_FOR_STATION_COMPLETION:
                # Get the details of the station we are waiting for
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
                if not station_field:
                    self.get_logger().error(
                        f"WAITING: Cannot find field name for station index {current_physical_station_idx}. Error."
                    )
                    self.state = RobotState.ERROR
                    return

                # Check for timeout first
                elapsed = time.time() - self.wait_start_time
                if elapsed > STATION_WAIT_TIMEOUT_SEC:
                    self.get_logger().warning(
                        f"WAIT TIMEOUT ({elapsed:.1f}s > {STATION_WAIT_TIMEOUT_SEC}s) for station {station_field}. Aborting order."
                    )
                    self.play_sound([(330, 500), (220, 500)])  # Timeout sound
                    self.state = RobotState.STATION_TIMED_OUT
                    return  # Exit loop for this cycle

                # Periodically check Airtable status
                now = time.time()
                if now - self._last_airtable_check_time >= AIRTABLE_POLL_RATE:
                    self._last_airtable_check_time = now
                    self.get_logger().debug(
                        f"Checking Airtable if {station_field} is DONE..."
                    )
                    current_status = self.get_station_status(
                        self.current_order["record_id"], station_field
                    )

                    if current_status is None:
                        self.get_logger().warning(
                            f"WAITING: Failed to get Airtable status for {station_field}. Continuing wait."
                        )
                        # Stay in this state, rely on timeout or next successful check

                    elif current_status == STATUS_DONE:
                        self.get_logger().info(
                            f"Station {station_field} reported DONE by Airtable."
                        )
                        self.play_sound([(659, 150), (784, 200)])  # Success sound

                        # Increment the index for the *order sequence*
                        self.current_order_station_index += 1

                        # Check if that was the last station for the ORDER
                        if self.current_order_station_index >= len(
                            self.ordered_station_sequence
                        ):
                            self.get_logger().info(
                                "Completed the last required station. Order finished."
                            )
                            self.state = RobotState.ORDER_COMPLETE
                        else:
                            # Set the next target station based on the ORDER sequence
                            self.target_station_physical_index = (
                                self.ordered_station_sequence[
                                    self.current_order_station_index
                                ]
                            )
                            self.get_logger().info(
                                f"Proceeding to next target station (from order): {STATION_INDEX_TO_FIELD.get(self.target_station_physical_index, 'Unknown')}"
                            )

                            # Find this target in the physical sequence to know which *physical* station to expect next
                            try:
                                next_physical_target_list_index = (
                                    PHYSICAL_STATION_SEQUENCE_INDICES.index(
                                        self.target_station_physical_index,
                                        self.physical_stations_visited_count,
                                    )
                                )
                            except ValueError:
                                self.get_logger().error(
                                    f"Could not find next target {self.target_station_physical_index} in physical sequence after index {self.physical_stations_visited_count}. Error."
                                )
                                self.state = RobotState.ERROR
                                return

                            self.get_logger().info(
                                f"Leaving station, expecting physical station {next_physical_target_list_index + 1}"
                            )
                            self.state = (
                                RobotState.LEAVING_STATION
                            )  # Move off the current line
                            self.leaving_station_start_time = time.time()
                        return  # Exit loop for this cycle

                    elif (
                        current_status == STATUS_ARRIVED
                        or current_status == STATUS_WAITING
                    ):
                        # Still waiting, do nothing, continue loop
                        self.get_logger().debug(
                            f"Station {station_field} not yet DONE (Status: {current_status}). Continuing wait."
                        )
                    else:
                        # Unexpected status during wait
                        self.get_logger().error(
                            f"WAITING: Station {station_field} has unexpected status {current_status}. Error."
                        )
                        self.state = RobotState.ERROR
                        return

            # --- State: LEAVING_STATION ---
            # Added state to briefly drive forward after completing or skipping a station
            # This helps ensure the color sensor is clear of the line before searching again.
            elif self.state == RobotState.LEAVING_STATION:
                elapsed = time.time() - self.leaving_station_start_time
                if elapsed < LEAVING_STATION_DURATION_SEC:
                    # self.get_logger().debug(f"Leaving station... moving forward slightly. {elapsed:.1f}s elapsed.")
                    # Move forward slowly, no turning
                    self.move_robot(BASE_DRIVE_SPEED * 0.7, 0.0)
                else:
                    self.get_logger().info(
                        f"Finished leaving station. Resuming search for next station."
                    )
                    self.stop_moving()
                    # We already determined the next target physical index in the previous state
                    self.initial_line_found = False  # Must acquire line again
                    self.state = RobotState.MOVING_TO_STATION
                    # Optional: reset color cooldown here? No, reset happens on detection.

            # --- State: ORDER_COMPLETE ---
            elif self.state == RobotState.ORDER_COMPLETE:
                order_name = (
                    self.current_order.get("order_name", "Unknown Order")
                    if self.current_order
                    else "Unknown Order"
                )
                self.get_logger().info(f"Order '{order_name}' successfully COMPLETED.")
                self.play_sound(
                    [(784, 150), (880, 150), (1047, 250)]
                )  # Completion fanfare
                self.stop_moving()
                # Optionally update Pickup status to DONE here if it wasn't the last processed station
                # (e.g., if the last action was cooking and pickup was already 99)
                pickup_field = AIRTABLE_PICKUP_STATUS_FIELD
                if (
                    self.current_order
                    and self.get_station_status(
                        self.current_order["record_id"], pickup_field
                    )
                    != STATUS_DONE
                ):
                    self.get_logger().info(
                        f"Ensuring Pickup status is set to DONE for completed order {order_name}."
                    )
                    self.update_station_status(
                        self.current_order["record_id"], pickup_field, STATUS_DONE
                    )
                    # Handle potential error from update? For now, proceed to IDLE.

                # Reset for next order search
                self.state = RobotState.IDLE

            # --- State: STATION_TIMED_OUT ---
            elif self.state == RobotState.STATION_TIMED_OUT:
                order_name = (
                    self.current_order.get("order_name", "Unknown Order")
                    if self.current_order
                    else "Unknown Order"
                )
                # Figure out which station timed out based on visited count
                timed_out_station_idx = -1
                timed_out_field = "Unknown"
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
                # Reset as if order failed/aborted
                self.current_order = None  # Clear order data
                self.state = RobotState.IDLE  # Go back to look for new orders

            # --- State: AIRTABLE_ERROR ---
            # This state is entered if Airtable updates/fetches fail critically
            elif self.state == RobotState.AIRTABLE_ERROR:
                self.get_logger().error(
                    "Persistent AIRTABLE ERROR detected. System halted. Check API token, base/table ID, network, and Airtable service status."
                )
                self.play_sound([(330, 300), (330, 300), (330, 300)])  # Error sound
                self.stop_moving()
                # Stay in this state until resolved externally? Or maybe try to transition back to IDLE after a long delay?
                # For now, halt indefinitely.

        except Exception as e:
            # Catch-all for unexpected errors within the state machine logic
            self.get_logger().fatal(
                f"!!! Unhandled exception in state {current_state.name}: {e} !!!",
                exc_info=True,  # Log the full traceback
            )
            self.state = RobotState.ERROR  # Enter critical error state
            self.stop_moving()

        # --- Post-State Machine Updates (Display) ---
        finally:
            if self.debug_windows:
                try:
                    if display_frame is not None:
                        cv2.imshow("Camera Feed", display_frame)
                    else:  # Ensure window exists even if frame processing failed
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
                            "No Camera Frame",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                        )
                        cv2.imshow("Camera Feed", blank_feed)

                    if mask_frame is not None:
                        mask_display = (
                            cv2.cvtColor(mask_frame, cv2.COLOR_GRAY2BGR)
                            if len(mask_frame.shape) == 2
                            else mask_frame
                        )
                        cv2.imshow("Color Detection Mask", mask_display)
                    else:  # Ensure window exists
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

                    # Check for quit key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        self.get_logger().info(
                            "Quit key ('q') pressed. Initiating shutdown."
                        )
                        self.state = RobotState.ERROR  # Trigger shutdown path
                        self.stop_moving()
                        # Request ROS shutdown
                        if rclpy.ok():
                            # Need to run shutdown in main thread, cannot call directly here
                            # Setting state to ERROR and stopping should allow main loop to exit.
                            # Consider using a dedicated shutdown flag if needed.
                            rclpy.try_shutdown()

                except Exception as e:
                    # Error during display update itself
                    self.get_logger().error(
                        f"Display update error: {e}",
                        exc_info=False,  # Keep log concise
                    )
                    # If display fails, maybe disable it?
                    # self.debug_windows = False


# --- Main Function ---
def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = None
    exit_code = 0  # Assume success unless error occurs

    try:
        node = PancakeRobotNode()
        # Check if node initialization failed critically
        if node.state in [
            RobotState.GPIO_ERROR,
            RobotState.CAMERA_ERROR,
            RobotState.ERROR,
        ]:
            node.get_logger().fatal(
                f"Node initialization failed critically with state: {node.state.name}. Aborting execution."
            )
            exit_code = 1  # Indicate failure
            # Cleanup might have already happened in init, but call again to be safe
            if node:
                node.cleanup_hardware()
                if node.debug_windows:
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)  # Allow windows to close
                node.destroy_node()  # Explicitly destroy node
            if rclpy.ok():
                rclpy.shutdown()
            return exit_code  # Exit script

        # Initialization succeeded, proceed with spinning
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        node.get_logger().info("Starting ROS2 executor spin...")

        # Spin until shutdown is requested (e.g., by Ctrl+C, 'q' key, or error state)
        while rclpy.ok() and node.state not in [
            RobotState.ERROR,
            RobotState.AIRTABLE_ERROR,
        ]:  # Halt on critical errors
            executor.spin_once(timeout_sec=0.1)  # Process callbacks, check state
            if (
                node.state == RobotState.ERROR
                or node.state == RobotState.AIRTABLE_ERROR
            ):
                node.get_logger().error(
                    f"Node entered error state ({node.state.name}). Shutting down executor."
                )
                exit_code = 1  # Indicate failure
                break  # Exit the spin loop

        node.get_logger().info("Spin loop exited.")

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received.")
        if node:
            node.get_logger().info("KeyboardInterrupt: Initiating shutdown...")
        exit_code = 0  # User initiated stop is not an error

    except Exception as e:
        print(f"\nFATAL Unhandled Error in main: {e}")
        if node:
            node.get_logger().fatal(
                f"FATAL Unhandled Error in main: {e}", exc_info=True
            )
        exit_code = 1  # Indicate failure

    finally:
        print("--- Initiating Final Cleanup ---")
        if node:
            node.get_logger().info("Stopping robot movement...")
            # Ensure stop command is sent even if ROS is shutting down
            try:
                node.stop_moving()
                # Short delay to allow command to be sent/processed
                time.sleep(0.2)
            except Exception as stop_err:
                node.get_logger().error(f"Error during final stop command: {stop_err}")

            # Shutdown ROS executor if it exists and wasn't already shutdown
            if executor and not executor.shutdown_called:
                node.get_logger().info("Shutting down ROS2 executor...")
                try:
                    executor.shutdown()
                    node.get_logger().info("Executor shutdown complete.")
                except Exception as exec_shut_err:
                    node.get_logger().error(
                        f"Error shutting down executor: {exec_shut_err}"
                    )

            node.get_logger().info("Cleaning up hardware resources...")
            node.cleanup_hardware()

            if node.debug_windows:
                node.get_logger().info("Closing OpenCV windows...")
                try:
                    cv2.destroyAllWindows()
                    cv2.waitKey(50)  # Short delay for windows to close
                except Exception as cv_err:
                    node.get_logger().error(f"Error closing OpenCV windows: {cv_err}")

            # Explicitly destroy the node after executor is down
            if not node.is_destroyed:
                node.get_logger().info("Destroying ROS2 node...")
                try:
                    node.destroy_node()
                    node.get_logger().info("Node destroyed.")
                except Exception as node_destroy_err:
                    node.get_logger().error(
                        f"Error destroying node: {node_destroy_err}"
                    )

        # Shutdown ROS context if it's still running
        if rclpy.ok():
            print("Shutting down rclpy...")
            try:
                rclpy.shutdown()
                print("rclpy shutdown complete.")
            except Exception as rclpy_shut_err:
                print(f"Error during rclpy shutdown: {rclpy_shut_err}")

        print(f"--- Shutdown sequence finished. Exit code: {exit_code} ---")
        # sys.exit(exit_code) # Uncomment if running as a script to return exit code


if __name__ == "__main__":
    main()
