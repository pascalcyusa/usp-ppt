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

# --- Field names in Airtable base (MUST match exactly, case-sensitive) ---
AIRTABLE_ORDER_NAME_COLUMN = "Order Name"
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
    AIRTABLE_PICKUP_STATUS_FIELD: 0,
}
STATION_INDEX_TO_FIELD = {v: k for k, v in STATION_FIELD_TO_INDEX.items()}

# --- Hardware Configuration ---
LEFT_IR_PIN = 16
RIGHT_IR_PIN = 18
# --- Corrected IR Interpretation: HIGH (1) = ON LINE (Dark), LOW (0) = OFF LINE (Light) ---
IR_LINE_DETECT_SIGNAL = GPIO.HIGH
IR_OFF_LINE_SIGNAL = GPIO.LOW

CAMERA_RESOLUTION = (640, 480)
CAMERA_TRANSFORM = Transform(hflip=True, vflip=True)

# --- Color Detection Configuration (Common Green) ---
COMMON_HSV_LOWER = (35, 100, 100)
COMMON_HSV_UPPER = (85, 255, 255)
COMMON_COLOR_BGR = (0, 255, 0)  # BGR for Green

STATION_COLORS_HSV = {
    idx: {
        "name": STATION_INDEX_TO_FIELD.get(idx, f"Station {idx}").replace(
            " Status", ""
        ),
        "hsv_lower": COMMON_HSV_LOWER,
        "hsv_upper": COMMON_HSV_UPPER,
        "color_bgr": COMMON_COLOR_BGR,
    }
    for idx in STATION_FIELD_TO_INDEX.values()  # Auto-populate based on defined stations
}

# --- Navigation & Control Parameters ---
AIRTABLE_POLL_RATE = 2.0

BASE_DRIVE_SPEED = 0.01  # m/s - Start slow, increase carefully
BASE_ROTATE_SPEED = 0.2  # rad/s - Start slow, increase carefully
TURN_FACTOR = 0.7  # Multiplier for speed reduction during turns (0.0-1.0)

# --- NEW: Simplified Line Search Parameter ---
LOST_LINE_ROTATE_SPEED = (
    0.15  # rad/s - Speed for rotating right when line is lost (tune this)
)
# --- End Search Parameter ---

COLOR_DETECTION_THRESHOLD = 2000
COLOR_COOLDOWN_SEC = 5.0
STATION_WAIT_TIMEOUT_SEC = 120.0
LEAVING_STATION_DURATION_SEC = 2.0


class RobotState(Enum):
    IDLE = auto()
    FETCHING_ORDER = auto()
    PLANNING_ROUTE = auto()
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


class PancakeRobotNode(Node):
    def __init__(self):
        super().__init__("pancake_robot_node")
        self.get_logger().info("Pancake Robot Node Initializing...")

        # State and Order Variables
        self.state = RobotState.IDLE
        self.current_order = None
        self.station_sequence = []
        self.current_sequence_index = -1
        self.target_station_index = -1

        # Timers and Cooldowns
        self.last_color_detection_times = {
            idx: 0.0 for idx in STATION_COLORS_HSV.keys()
        }
        self.wait_start_time = 0.0
        self.leaving_station_start_time = 0.0
        self._last_airtable_check_time = 0.0

        # Flags
        self.initial_line_found = False  # Tracks if line has been seen at least once

        # Hardware and Display
        self.picam2 = None
        self.debug_windows = True  # Set to False to disable CV windows

        # ROS Publishers/Clients
        self.cmd_vel_pub = None
        self.audio_publisher = None
        self.drive_client = None  # Action client (not used for line following)
        self.rotate_client = None  # Action client (not used for line following)

        # Initialization
        self._init_hardware()
        if self.state in [RobotState.GPIO_ERROR, RobotState.CAMERA_ERROR]:
            self.get_logger().fatal("Hardware init failed.")
            return
        self._init_ros2()
        if self.state == RobotState.ERROR:
            self.get_logger().fatal("ROS2 init failed.")
            self.cleanup_hardware()
            return

        # Control Loop Timer (20 Hz -> 0.05s interval)
        self.control_timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info("Pancake Robot Node Initialized and Ready.")
        self.play_sound([(440, 150), (550, 200)])  # Initial sound

    def _init_hardware(self):
        """Initialize GPIO and Camera"""
        # GPIO
        try:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(LEFT_IR_PIN, GPIO.IN)
            GPIO.setup(RIGHT_IR_PIN, GPIO.IN)
            self.get_logger().info(
                f"GPIO initialized (Pins: L={LEFT_IR_PIN}, R={RIGHT_IR_PIN}). Expecting {IR_LINE_DETECT_SIGNAL} (HIGH) on line."
            )
        except Exception as e:
            self.get_logger().error(f"FATAL: GPIO init failed: {e}", exc_info=True)
            self.state = RobotState.GPIO_ERROR
            return
        # Camera
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": CAMERA_RESOLUTION}, transform=CAMERA_TRANSFORM
            )
            self.picam2.configure(config)
            self.picam2.set_controls(
                {"AfMode": controls.AfModeEnum.Continuous, "LensPosition": 0.0}
            )
            self.picam2.start()
            time.sleep(2)  # Allow camera to stabilize
            self.get_logger().info("Pi Camera initialized.")
            if self.debug_windows:
                cv2.namedWindow("Camera Feed")
                cv2.namedWindow("Color Detection Mask")
        except Exception as e:
            self.get_logger().error(f"FATAL: Camera init failed: {e}", exc_info=True)
            self.cleanup_gpio()  # Cleanup GPIO if camera fails after GPIO init
            self.state = RobotState.CAMERA_ERROR
            return

    def _init_ros2(self):
        """Initialize ROS2 components"""
        try:
            self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
            self.audio_publisher = self.create_publisher(
                AudioNoteVector, "/cmd_audio", 10
            )
            # Action clients are not used in the line following part, but kept for potential future use
            self.drive_client = ActionClient(self, DriveDistance, "/drive_distance")
            self.rotate_client = ActionClient(self, RotateAngle, "/rotate_angle")

            if not self.audio_publisher or not self.cmd_vel_pub:
                raise RuntimeError("Publisher creation failed.")

            self.get_logger().info("ROS2 components initialized.")
        except Exception as e:
            self.get_logger().error(f"FATAL: ROS2 init failed: {e}", exc_info=True)
            self.state = RobotState.ERROR

    # --- Airtable Functions (Unchanged) ---
    def fetch_order_from_airtable(self):
        try:
            params = {
                "maxRecords": 1,
                "filterByFormula": f"AND({{{AIRTABLE_COOKING_1_STATUS_FIELD}}}=0, {{{AIRTABLE_PICKUP_STATUS_FIELD}}}=0)",
                "sort[0][field]": AIRTABLE_ORDER_NAME_COLUMN,
                "sort[0][direction]": "asc",
            }
            response = requests.get(
                url=AIRTABLE_URL, headers=AIRTABLE_HEADERS, params=params, timeout=15
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
                    field: fields.get(field, 0)
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
            response = requests.patch(
                url=url, headers=AIRTABLE_HEADERS, json=data, timeout=10
            )
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            self.get_logger().info(f"Airtable: Updated {field} to {status}.")
            return True
        except requests.exceptions.Timeout:
            self.get_logger().error(f"Airtable update timed out for {field}.")
            self.state = RobotState.AIRTABLE_ERROR
            return False
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Airtable update error for {field}: {e}")
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
            return False  # Assume not done on timeout
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Airtable check error ({field}): {e}")
            return False  # Assume not done on error
        except Exception as e:
            self.get_logger().error(
                f"Airtable check unexpected error: {e}", exc_info=True
            )
            return False  # Assume not done on error

    # --- Robot Movement and Sensors ---
    def move_robot(self, linear_x, angular_z):
        """Publishes Twist message to /cmd_vel"""
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
                self.cmd_vel_pub.publish(twist)
            return

        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        # self.get_logger().debug(f"Publishing Twist: Lin={linear_x:.3f}, Ang={angular_z:.3f}")
        try:
            self.cmd_vel_pub.publish(twist)
        except Exception as e:
            self.get_logger().error(f"Failed to publish Twist: {e}")

    def stop_moving(self):
        """Sends zero velocity Twist commands multiple times."""
        # self.get_logger().debug("Stopping movement...")
        for _ in range(3):  # Send multiple times for reliability
            self.move_robot(0.0, 0.0)
            time.sleep(0.02)  # Small delay between publishes

    def read_ir_sensors(self):
        """Reads IR sensors. Returns (left_on_line, right_on_line). HIGH = ON LINE."""
        try:
            left_val = GPIO.input(LEFT_IR_PIN)
            right_val = GPIO.input(RIGHT_IR_PIN)
            # self.get_logger().debug(f"IR Raw: L={left_val}, R={right_val}") # Optional raw logging
            # HIGH signal means the sensor is over the dark line
            return (left_val == IR_LINE_DETECT_SIGNAL), (
                right_val == IR_LINE_DETECT_SIGNAL
            )
        except Exception as e:
            self.get_logger().error(f"IR sensor read error: {e}", exc_info=True)
            self.state = RobotState.GPIO_ERROR  # Treat GPIO read error as critical
            return False, False

    # --- Color Detection (Unchanged) ---
    def check_for_station_color(self, frame, target_idx):
        """Checks frame for the color associated with target_idx."""
        detected_flag = False
        display_frame = frame.copy()
        mask_frame = np.zeros(
            (frame.shape[0], frame.shape[1]), dtype=np.uint8
        )  # Initialize blank mask

        # Put state text on display frame
        cv2.putText(
            display_frame,
            f"State: {self.state.name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        if target_idx not in STATION_COLORS_HSV:
            cv2.putText(
                display_frame,
                f"Target: Invalid ({target_idx})",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            # Return original frame and blank mask if target is invalid
            return False, display_frame, mask_frame

        # Get target color info (all stations use common green now)
        color_info = STATION_COLORS_HSV[target_idx]
        target_name = color_info["name"]
        target_bgr = COMMON_COLOR_BGR
        lower_bound = np.array(COMMON_HSV_LOWER)
        upper_bound = np.array(COMMON_HSV_UPPER)

        try:
            # Convert to HSV and create mask
            hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            color_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
            mask_frame = color_mask  # Assign the calculated mask

            # Count non-zero pixels in the mask
            detected_pixels = cv2.countNonZero(color_mask)

            # Display target info and pixel count
            text = f"Target: {target_name} ({detected_pixels} px)"
            cv2.putText(
                display_frame,
                text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                target_bgr,
                2,
            )

            # Check detection threshold and cooldown
            current_time = time.time()
            last_detection_time = self.last_color_detection_times.get(target_idx, 0.0)

            if detected_pixels > COLOR_DETECTION_THRESHOLD and (
                current_time - last_detection_time > COLOR_COOLDOWN_SEC
            ):
                self.last_color_detection_times[target_idx] = current_time
                detected_flag = True
                self.get_logger().info(f"Detected color for {target_name}!")
                # Add "DETECTED!" text to display frame
                cv2.putText(
                    display_frame,
                    "DETECTED!",
                    (frame.shape[1] // 2 - 50, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            return detected_flag, display_frame, mask_frame

        except cv2.error as e:
            self.get_logger().error(f"OpenCV error during color detection: {e}")
            # Return original frame and blank mask on OpenCV error
            return False, display_frame, np.zeros_like(mask_frame)
        except Exception as e:
            self.get_logger().error(
                f"Unexpected error during color detection: {e}", exc_info=True
            )
            # Return original frame and blank mask on other errors
            return False, display_frame, np.zeros_like(mask_frame)

    # --- Sound (Unchanged from fix) ---
    def play_sound(self, notes):
        """Publishes a sequence of audio notes."""
        if not hasattr(self, "audio_publisher") or self.audio_publisher is None:
            self.get_logger().warning(
                "Audio publisher not initialized. Cannot play sound."
            )
            return

        note_msg = AudioNoteVector()
        note_list_str = []  # For logging

        for freq, dur in notes:
            # Basic validation for frequency and duration
            if freq <= 0 or dur <= 0:
                self.get_logger().warning(
                    f"Skipping invalid note: Freq={freq}, Dur={dur}"
                )
                continue

            note = AudioNote()
            note.frequency = int(freq)
            # Convert duration from milliseconds to seconds and nanoseconds
            duration_sec = int(dur / 1000)
            duration_nanosec = int((dur % 1000) * 1e6)
            note.max_runtime = Duration(sec=duration_sec, nanosec=duration_nanosec)

            note_msg.notes.append(note)
            note_list_str.append(f"({freq}Hz,{dur}ms)")

        if not note_msg.notes:
            self.get_logger().warning("No valid notes provided to play_sound.")
            return

        self.get_logger().info(f"Playing Audio Sequence: {', '.join(note_list_str)}")
        try:
            self.audio_publisher.publish(note_msg)
            # self.get_logger().debug("Audio message published.")
        except Exception as e:
            self.get_logger().error(f"Audio publish failed: {e}", exc_info=True)

    # --- Cleanup ---
    def cleanup_gpio(self):
        """Cleans up GPIO resources."""
        self.get_logger().info("Cleaning up GPIO...")
        try:
            GPIO.cleanup()
            self.get_logger().info("GPIO cleanup successful.")
        except Exception as e:
            # Log error but don't crash shutdown
            self.get_logger().error(f"GPIO cleanup error: {e}")

    def cleanup_hardware(self):
        """Stops camera and cleans up GPIO."""
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

    # --- Main Control Loop ---
    def control_loop(self):
        """Main state machine and control logic for the robot."""
        # --- Pre-State Machine Checks ---
        # If in a critical error state, ensure robot is stopped and do nothing else.
        if self.state in [
            RobotState.ERROR,
            RobotState.CAMERA_ERROR,
            RobotState.GPIO_ERROR,
            RobotState.AIRTABLE_ERROR,  # Also halt on persistent Airtable errors
        ]:
            self.stop_moving()
            return

        # --- Camera Frame Processing ---
        display_frame, mask_frame, color_detected = None, None, False
        process_color_flag = self.state == RobotState.MOVING_TO_STATION

        if self.picam2 and self.state != RobotState.CAMERA_ERROR:
            try:
                raw_frame = self.picam2.capture_array()
                if self.target_station_index != -1:
                    _detected_flag, display_frame, mask_frame = (
                        self.check_for_station_color(
                            raw_frame, self.target_station_index
                        )
                    )
                    if process_color_flag:
                        color_detected = _detected_flag
                else:
                    display_frame = raw_frame.copy()
                    cv2.putText(
                        display_frame,
                        f"State: {self.state.name}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        "Target: None",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    mask_frame = np.zeros(
                        (raw_frame.shape[0], raw_frame.shape[1]), dtype=np.uint8
                    )

            except Exception as e:
                self.get_logger().error(
                    f"Camera frame processing error: {e}", exc_info=True
                )
                self.state = RobotState.CAMERA_ERROR  # Transition to camera error state
                self.stop_moving()
                return  # Exit control loop iteration

        # --- State Machine Logic ---
        try:
            # State: IDLE
            if self.state == RobotState.IDLE:
                self.stop_moving()
                self.current_order = None
                self.station_sequence = []
                self.current_sequence_index = -1
                self.target_station_index = -1
                self.initial_line_found = False  # Reset line flag

                self.current_order = self.fetch_order_from_airtable()
                if self.current_order:
                    if self.state != RobotState.AIRTABLE_ERROR:
                        self.get_logger().info(
                            f"Order '{self.current_order['order_name']}' received. Planning route."
                        )
                        self.state = RobotState.PLANNING_ROUTE
                elif self.state != RobotState.AIRTABLE_ERROR:
                    self.state = RobotState.ALL_ORDERS_COMPLETE

            # State: PLANNING_ROUTE
            elif self.state == RobotState.PLANNING_ROUTE:
                if not self.current_order:
                    self.get_logger().error(
                        "PLANNING: No current order available! Returning to IDLE."
                    )
                    self.state = RobotState.IDLE
                    return

                self.station_sequence = []
                order_status = self.current_order["station_status"]
                # Simplified planning
                if order_status.get(AIRTABLE_COOKING_1_STATUS_FIELD) == STATUS_WAITING:
                    self.station_sequence.append(
                        STATION_FIELD_TO_INDEX[AIRTABLE_COOKING_1_STATUS_FIELD]
                    )
                if order_status.get(AIRTABLE_COOKING_2_STATUS_FIELD) == STATUS_WAITING:
                    self.station_sequence.append(
                        STATION_FIELD_TO_INDEX[AIRTABLE_COOKING_2_STATUS_FIELD]
                    )
                if (
                    order_status.get(AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD)
                    == STATUS_WAITING
                ):
                    self.station_sequence.append(
                        STATION_FIELD_TO_INDEX[AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD]
                    )
                if (
                    order_status.get(AIRTABLE_WHIPPED_CREAM_STATUS_FIELD)
                    == STATUS_WAITING
                ):
                    self.station_sequence.append(
                        STATION_FIELD_TO_INDEX[AIRTABLE_WHIPPED_CREAM_STATUS_FIELD]
                    )
                if order_status.get(AIRTABLE_SPRINKLES_STATUS_FIELD) == STATUS_WAITING:
                    self.station_sequence.append(
                        STATION_FIELD_TO_INDEX[AIRTABLE_SPRINKLES_STATUS_FIELD]
                    )
                self.station_sequence.append(
                    STATION_FIELD_TO_INDEX[AIRTABLE_PICKUP_STATUS_FIELD]
                )  # Always add Pickup

                if not self.station_sequence:
                    self.get_logger().error(
                        "Route planning resulted in an empty sequence! Returning to IDLE."
                    )
                    self.state = RobotState.IDLE
                    self.current_order = None
                else:
                    self.current_sequence_index = 0
                    self.target_station_index = self.station_sequence[0]
                    self.get_logger().info(
                        f"Route planned: {self.station_sequence}. Next target: Station {self.target_station_index}"
                    )
                    self.initial_line_found = False  # Reset for the new journey
                    self.state = RobotState.MOVING_TO_STATION
                    self.play_sound(
                        [(440, 100), (550, 100), (660, 100)]
                    )  # Planning complete sound

            # State: LEAVING_STATION
            elif self.state == RobotState.LEAVING_STATION:
                elapsed_time = time.time() - self.leaving_station_start_time
                if elapsed_time < LEAVING_STATION_DURATION_SEC:
                    self.move_robot(
                        BASE_DRIVE_SPEED * 0.5, 0.0
                    )  # Drive straight slowly
                else:
                    self.get_logger().info(
                        f"Finished leaving station. Transitioning to move to next station: {self.target_station_index}."
                    )
                    self.stop_moving()
                    self.initial_line_found = False  # Must find the line again
                    self.state = RobotState.MOVING_TO_STATION

            # State: MOVING_TO_STATION
            elif self.state == RobotState.MOVING_TO_STATION:
                # --- Check for Station Arrival (Color Detection) FIRST ---
                if color_detected:
                    self.get_logger().info(
                        f"Color detected for target station {self.target_station_index}. Arriving."
                    )
                    self.play_sound([(523, 100), (659, 150)])  # Arrival sound
                    self.stop_moving()
                    self.state = RobotState.ARRIVED_AT_STATION
                    return  # Exit loop for this cycle

                # --- Check Target Validity ---
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

                # --- Line Following Logic ---
                left_on, right_on = self.read_ir_sensors()

                # === CASE 1: Both Sensors OFF Line ===
                if not left_on and not right_on:
                    # --- MODIFIED: Always Rotate Right to Search ---
                    if not self.initial_line_found:
                        # Still searching for the line for the first time in this segment
                        self.get_logger().debug(
                            "Searching for initial line (turning right)..."
                        )
                    else:
                        # Line was previously found, but now lost.
                        self.get_logger().warning(
                            "Line lost! Rotating right to search..."
                        )
                        self.play_sound([(330, 100)])  # Lost line sound (optional)

                    # Command the right rotation
                    self.move_robot(0.0, -LOST_LINE_ROTATE_SPEED)
                    # --- End Modification ---

                # === CASE 2: At least one Sensor ON Line ===
                else:
                    # Mark initial line found if it wasn't already
                    if not self.initial_line_found:
                        self.get_logger().info("Initial line found!")
                        self.initial_line_found = True
                        self.play_sound([(660, 100)])  # Found line sound (optional)

                    # --- Normal Line Following ---
                    if left_on and right_on:  # Both ON line -> Drive straight
                        # self.get_logger().debug("Line Follow: Straight")
                        self.move_robot(BASE_DRIVE_SPEED, 0.0)
                    elif (
                        left_on and not right_on
                    ):  # Left ON, Right OFF -> Turn Left (Correct Right Drift)
                        # self.get_logger().debug("Line Follow: Correct Left")
                        self.move_robot(
                            BASE_DRIVE_SPEED * TURN_FACTOR, -BASE_ROTATE_SPEED
                        )
                    elif (
                        not left_on and right_on
                    ):  # Left OFF, Right ON -> Turn Right (Correct Left Drift)
                        # self.get_logger().debug("Line Follow: Correct Right")
                        self.move_robot(
                            BASE_DRIVE_SPEED * TURN_FACTOR, BASE_ROTATE_SPEED
                        )

            # State: ARRIVED_AT_STATION
            elif self.state == RobotState.ARRIVED_AT_STATION:
                self.stop_moving()
                if (
                    self.current_sequence_index < 0
                    or self.current_sequence_index >= len(self.station_sequence)
                ):
                    self.get_logger().error(
                        f"ARRIVED: Invalid sequence index {self.current_sequence_index}. Halting."
                    )
                    self.state = RobotState.ERROR
                    return

                current_station_idx = self.station_sequence[self.current_sequence_index]
                if current_station_idx not in STATION_INDEX_TO_FIELD:
                    self.get_logger().error(
                        f"ARRIVED: No Airtable field defined for station index {current_station_idx}. Halting."
                    )
                    self.state = RobotState.ERROR
                    return

                station_field = STATION_INDEX_TO_FIELD[current_station_idx]
                self.get_logger().info(
                    f"Arrived at Station {current_station_idx} ({station_field}). Updating status to ARRIVED."
                )

                if self.update_station_status(
                    self.current_order["record_id"], station_field, STATUS_ARRIVED
                ):
                    self.wait_start_time = time.time()
                    self.state = RobotState.WAITING_FOR_STATION_COMPLETION
                    self.get_logger().info(
                        f"Now waiting for station {current_station_idx} ({station_field}) to be marked DONE."
                    )
                    self.play_sound([(440, 100), (440, 100)])  # Waiting sound
                else:
                    self.get_logger().error(
                        f"Failed to update Airtable status for {station_field}. State is now {self.state.name}"
                    )

            # State: WAITING_FOR_STATION_COMPLETION
            elif self.state == RobotState.WAITING_FOR_STATION_COMPLETION:
                elapsed_wait_time = time.time() - self.wait_start_time
                if elapsed_wait_time > STATION_WAIT_TIMEOUT_SEC:
                    self.get_logger().warning(
                        f"WAIT TIMEOUT ({elapsed_wait_time:.1f}s > {STATION_WAIT_TIMEOUT_SEC}s) for station {self.target_station_index}. Aborting order."
                    )
                    self.play_sound([(330, 500), (220, 500)])  # Timeout sound
                    self.state = RobotState.STATION_TIMED_OUT
                    return

                if time.time() - self._last_airtable_check_time >= AIRTABLE_POLL_RATE:
                    self._last_airtable_check_time = time.time()

                    if (
                        self.current_sequence_index < 0
                        or self.current_sequence_index >= len(self.station_sequence)
                    ):
                        self.get_logger().error(
                            f"WAITING: Invalid sequence index {self.current_sequence_index}. Halting."
                        )
                        self.state = RobotState.ERROR
                        return
                    current_station_idx = self.station_sequence[
                        self.current_sequence_index
                    ]
                    if current_station_idx not in STATION_INDEX_TO_FIELD:
                        self.get_logger().error(
                            f"WAITING: No field for index {current_station_idx}. Halting."
                        )
                        self.state = RobotState.ERROR
                        return
                    station_field = STATION_INDEX_TO_FIELD[current_station_idx]

                    self.get_logger().debug(
                        f"Checking Airtable if {station_field} is DONE..."
                    )
                    if self.wait_for_station_completion(
                        self.current_order["record_id"], station_field
                    ):
                        self.get_logger().info(
                            f"Station {current_station_idx} ({station_field}) reported DONE."
                        )
                        self.play_sound(
                            [(659, 150), (784, 200)]
                        )  # Station complete sound
                        self.current_sequence_index += 1

                        if self.current_sequence_index >= len(self.station_sequence):
                            self.get_logger().info(
                                "All stations for this order are complete."
                            )
                            self.state = RobotState.ORDER_COMPLETE
                        else:
                            self.target_station_index = self.station_sequence[
                                self.current_sequence_index
                            ]
                            self.get_logger().info(
                                f"Proceeding to next station: {self.target_station_index}. Starting LEAVING_STATION procedure."
                            )
                            self.last_color_detection_times[current_station_idx] = 0.0
                            self.leaving_station_start_time = time.time()
                            self.state = RobotState.LEAVING_STATION
                    elif self.state == RobotState.AIRTABLE_ERROR:
                        self.get_logger().error(
                            "Airtable error occurred during status check. Halting."
                        )
                    else:
                        self.get_logger().debug(
                            f"Station {station_field} not yet DONE. Continuing wait."
                        )

            # State: ORDER_COMPLETE
            elif self.state == RobotState.ORDER_COMPLETE:
                order_name = (
                    self.current_order.get("order_name", "Unknown Order")
                    if self.current_order
                    else "Unknown Order"
                )
                self.get_logger().info(f"Order '{order_name}' successfully completed.")
                self.play_sound(
                    [(784, 150), (880, 150), (1047, 250)]
                )  # Order complete fanfare
                self.stop_moving()
                self.current_order = None
                self.station_sequence = []
                self.current_sequence_index = -1
                self.target_station_index = -1
                self.initial_line_found = False
                self.state = RobotState.IDLE  # Go back to idle to check for more orders

            # State: ALL_ORDERS_COMPLETE
            elif self.state == RobotState.ALL_ORDERS_COMPLETE:
                self.get_logger().info("No pending orders found. Entering idle wait.")
                self.play_sound([(440, 200), (440, 200)])  # Idle waiting sound
                self.stop_moving()
                time.sleep(5.0)  # Wait for a bit before checking again
                self.state = RobotState.IDLE  # Go back to idle to re-check Airtable

            # State: STATION_TIMED_OUT
            elif self.state == RobotState.STATION_TIMED_OUT:
                order_name = (
                    self.current_order.get("order_name", "Unknown Order")
                    if self.current_order
                    else "Unknown Order"
                )
                self.get_logger().error(
                    f"STATION TIMED OUT waiting for station {self.target_station_index}. Aborting order '{order_name}'."
                )
                self.stop_moving()
                self.current_order = None
                self.station_sequence = []
                self.current_sequence_index = -1
                self.target_station_index = -1
                self.initial_line_found = False
                self.state = RobotState.IDLE  # Go back to idle after timeout

            # State: AIRTABLE_ERROR (Handled at the top, but log here if somehow reached)
            elif self.state == RobotState.AIRTABLE_ERROR:
                self.get_logger().error(
                    "Persistent AIRTABLE ERROR detected. System halted."
                )
                self.play_sound([(330, 300), (330, 300), (330, 300)])  # Error sound
                self.stop_moving()

        except Exception as e:
            self.get_logger().fatal(
                f"Unhandled exception in state machine ({self.state.name}): {e}",
                exc_info=True,
            )
            self.state = RobotState.ERROR  # Go to generic error state
            self.stop_moving()

        # --- Post-State Machine: Display Update ---
        finally:
            if self.debug_windows:
                try:
                    if display_frame is not None:
                        cv2.imshow("Camera Feed", display_frame)
                    if mask_frame is not None:
                        if len(mask_frame.shape) == 2:
                            mask_display = cv2.cvtColor(mask_frame, cv2.COLOR_GRAY2BGR)
                        else:
                            mask_display = mask_frame
                        cv2.imshow("Color Detection Mask", mask_display)
                    else:
                        blank_mask = np.zeros(
                            (CAMERA_RESOLUTION[1], CAMERA_RESOLUTION[0], 3),
                            dtype=np.uint8,
                        )
                        cv2.imshow("Color Detection Mask", blank_mask)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        self.get_logger().info("Quit key ('q') pressed. Shutting down.")
                        self.state = RobotState.ERROR
                        self.stop_moving()
                        if rclpy.ok():
                            rclpy.try_shutdown()

                except Exception as e:
                    self.get_logger().error(
                        f"Display update error: {e}", exc_info=False
                    )


# --- Main Function ---
def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = None
    try:
        node = PancakeRobotNode()
        if node.state not in [
            RobotState.GPIO_ERROR,
            RobotState.CAMERA_ERROR,
            RobotState.ERROR,
            RobotState.AIRTABLE_ERROR,
        ]:
            executor = SingleThreadedExecutor()
            executor.add_node(node)
            node.get_logger().info("Starting ROS2 executor spin...")
            executor.spin()
        else:
            node.get_logger().fatal(
                f"Node initialization failed with state: {node.state.name}. Aborting execution."
            )
            if node:
                node.cleanup_hardware()
                if node.debug_windows:
                    cv2.destroyAllWindows()

    except KeyboardInterrupt:
        print("KeyboardInterrupt received.")
        if node:
            node.get_logger().info("KeyboardInterrupt received, initiating shutdown...")
    except Exception as e:
        print(f"FATAL Unhandled Error in main: {e}")
        if node:
            node.get_logger().fatal(
                f"FATAL Unhandled Error in main: {e}", exc_info=True
            )
    finally:
        print("Initiating final cleanup...")
        if node:
            node.get_logger().info("Stopping robot movement...")
            node.stop_moving()
            if executor:
                node.get_logger().info("Shutting down ROS2 executor...")
                executor.shutdown()
                node.get_logger().info("Executor shutdown.")
            node.get_logger().info("Cleaning up hardware resources...")
            node.cleanup_hardware()
            if node.debug_windows:
                node.get_logger().info("Closing OpenCV windows...")
                cv2.destroyAllWindows()
                cv2.waitKey(50)
            node.get_logger().info("Destroying ROS2 node...")
            node.destroy_node()
            node.get_logger().info("Node destroyed.")
        if rclpy.ok():
            print("Shutting down rclpy...")
            rclpy.shutdown()
            print("rclpy shutdown complete.")
        print("Shutdown sequence finished.")


if __name__ == "__main__":
    main()
