# Code works as of 4/15/2025 @ 11:43 PM EST
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
AIRTABLE_URL = (
    f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
)
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
STATUS_DONE = 99

# --- Map Airtable Fields to Station Indices ---
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
BASE_DRIVE_SPEED = 0.01  # m/s
BASE_ROTATE_SPEED = 0.2  # rad/s
TURN_FACTOR = 0.7  # Speed multiplier during turns
LOST_LINE_ROTATE_SPEED = 0.1  # rad/s - Speed for search rotation
COLOR_DETECTION_THRESHOLD = 2000
COLOR_COOLDOWN_SEC = 5.0
STATION_WAIT_TIMEOUT_SEC = 120.0
LEAVING_STATION_DURATION_SEC = 2.0


# --- Robot States ---
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


# --- Main Node Class ---
class PancakeRobotNode(Node):
    def __init__(self):
        super().__init__("pancake_robot_node")
        self.get_logger().info("Pancake Robot Node Initializing...")
        # Initialize attributes...
        self.state = RobotState.IDLE
        self.current_order = None
        self.station_sequence = []
        self.current_sequence_index = -1
        self.target_station_index = -1
        self.last_color_detection_times = {
            idx: 0.0 for idx in STATION_COLORS_HSV.keys()
        }
        self.wait_start_time = 0.0
        self.leaving_station_start_time = 0.0
        self._last_airtable_check_time = 0.0
        self.initial_line_found = False
        self.picam2 = None
        self.debug_windows = True
        self.cmd_vel_pub = None
        self.audio_publisher = None
        self.drive_client = None
        self.rotate_client = None
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
        self.control_timer = self.create_timer(
            0.05, self.control_loop
        )  # 20 Hz loop
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
            self.drive_client = ActionClient(
                self, DriveDistance, "/drive_distance"
            )
            self.rotate_client = ActionClient(
                self, RotateAngle, "/rotate_angle"
            )
            if not self.audio_publisher:
                raise RuntimeError("Audio pub failed.")
                self.get_logger().info("ROS2 ok.")
        except Exception as e:
            self.get_logger().error(f"FATAL: ROS2 init: {e}", exc_info=True)
            self.state = RobotState.ERROR

    # --- Airtable Functions ---
    def fetch_order_from_airtable(self):
        try:
            params = {
                "maxRecords": 1,
                "filterByFormula": f"AND({{{AIRTABLE_COOKING_1_STATUS_FIELD}}}=0, {{{AIRTABLE_PICKUP_STATUS_FIELD}}}=0)",
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
            self.get_logger().info(
                f"Fetched order '{order_name}' (ID: {record_id})."
            )
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
            self.get_logger().error(
                "Airtable update error: Missing ID or field"
            )
            return False
        data = {"fields": {field: status}}
        url = f"{AIRTABLE_URL}/{record_id}"
        try:
            requests.patch(
                url=url, headers=AIRTABLE_HEADERS, json=data, timeout=10
            ).raise_for_status()
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
            response = requests.get(
                url=url, headers=AIRTABLE_HEADERS, timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return data.get("fields", {}).get(field) == STATUS_DONE
        except requests.exceptions.Timeout:
            self.get_logger().warning(f"Airtable check timed out for {field}.")
            return False
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Airtable check error ({field}): {e}")
            return False
        except Exception as e:
            self.get_logger().error(
                f"Airtable check unexpected error: {e}", exc_info=True
            )
            return False

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
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            if self.cmd_vel_pub:
                self.cmd_vel_pub.publish(twist)
                return
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(twist)

    def stop_moving(self):
        for _ in range(3):
            self.move_robot(0.0, 0.0)
            time.sleep(0.02)

    def read_ir_sensors(self):
        """Reads IR sensors. Returns (left_on_line, right_on_line). HIGH = ON LINE."""
        try:
            left_val = GPIO.input(LEFT_IR_PIN)
            right_val = GPIO.input(RIGHT_IR_PIN)
            return (left_val == IR_LINE_DETECT_SIGNAL), (
                right_val == IR_LINE_DETECT_SIGNAL
            )
        except Exception as e:
            self.get_logger().error(f"IR sensor read error: {e}", exc_info=True)
            return False, False

    # --- Color Detection ---
    def check_for_station_color(self, frame, target_idx):
        detected_flag = False
        display_frame = frame.copy()
        mask_frame = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
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
            return False, display_frame, mask_frame
        color_info = STATION_COLORS_HSV[target_idx]
        target_name = color_info["name"]
        target_bgr = COMMON_COLOR_BGR
        lower = np.array(COMMON_HSV_LOWER)
        upper = np.array(COMMON_HSV_UPPER)
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            mask_frame = mask
            pixels = cv2.countNonZero(mask)
            text = f"Target: {target_name} ({pixels} px)"
            cv2.putText(
                display_frame,
                text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                target_bgr,
                2,
            )
            now = time.time()
            if pixels > COLOR_DETECTION_THRESHOLD and (
                now - self.last_color_detection_times.get(target_idx, 0.0)
                > COLOR_COOLDOWN_SEC
            ):
                self.last_color_detection_times[target_idx] = now
                detected_flag = True
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
            self.get_logger().error(f"OpenCV color error: {e}")
            return False, display_frame, np.zeros_like(mask_frame)
        except Exception as e:
            self.get_logger().error(f"Color detect error: {e}", exc_info=True)
            return False, display_frame, np.zeros_like(mask_frame)

    # --- Sound ---
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
            n.max_runtime = Duration(
                sec=int(d / 1000), nanosec=int((d % 1000) * 1e6)
            )
            msg.notes.append(n)
            log_str.append(f"({f}Hz,{d}ms)")
        if not msg.notes:
            self.get_logger().warning("No valid notes.")
            return
        self.get_logger().info(f"Audio Cmd: {','.join(log_str)}")
        try:
            self.audio_publisher.publish(msg)
            self.get_logger().debug("Audio published.")
        except Exception as e:
            self.get_logger().error(f"Audio publish failed: {e}", exc_info=True)

    # --- Cleanup ---
    def cleanup_gpio(self):
        try:
            GPIO.cleanup()
        except Exception as e:
            self.get_logger().error(f"GPIO cleanup error: {e}")

    def cleanup_hardware(self):
        self.cleanup_gpio()
        if self.picam2:
            try:
                self.picam2.stop()
            except Exception as e:
                self.get_logger().error(f"Camera stop error: {e}")

    # --- Main Control Loop ---
    def control_loop(self):
        """Main state machine and control logic for the robot."""
        if self.state in [
            RobotState.ERROR,
            RobotState.CAMERA_ERROR,
            RobotState.GPIO_ERROR,
        ]:
            self.stop_moving()
            return

        display_frame, mask_frame, color_detected = None, None, False
        process_color = self.state == RobotState.MOVING_TO_STATION

        # Camera Handling
        if self.picam2 and self.state != RobotState.CAMERA_ERROR:
            try:
                raw = self.picam2.capture_array()
                if self.target_station_index != -1:
                    _det, display_frame, mask_frame = (
                        self.check_for_station_color(
                            raw, self.target_station_index
                        )
                    )
                if process_color:
                    color_detected = _det
                else:
                    display_frame = raw.copy()
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
            except Exception as e:
                self.get_logger().error(
                    f"Camera loop error: {e}", exc_info=True
                )
                self.state = RobotState.CAMERA_ERROR
                self.stop_moving()
                return

        # State Machine
        try:
            # Handle States
            if self.state == RobotState.IDLE:
                self.stop_moving()
                self.current_order = None
                self.station_sequence = []
                self.current_sequence_index = -1
                self.target_station_index = -1
                self.initial_line_found = False
                self.current_order = self.fetch_order_from_airtable()
                if self.current_order:
                    if self.state != RobotState.AIRTABLE_ERROR:
                        self.get_logger().info(
                            f"Order '{self.current_order['order_name']}' received."
                        )
                        self.state = RobotState.PLANNING_ROUTE
                elif self.state != RobotState.AIRTABLE_ERROR:
                    self.state = RobotState.ALL_ORDERS_COMPLETE

            elif self.state == RobotState.PLANNING_ROUTE:
                if not self.current_order:
                    self.get_logger().error("PLANNING: No order!")
                    self.state = RobotState.IDLE
                    return
                self.station_sequence = []
                status = self.current_order["station_status"]
                if (
                    status.get(AIRTABLE_COOKING_1_STATUS_FIELD)
                    == STATUS_WAITING
                ):
                    self.station_sequence.append(1)
                else:
                    self.get_logger().error(f"Order invalid start.")
                    self.state = RobotState.IDLE
                    self.current_order = None
                    return
                if (
                    status.get(AIRTABLE_COOKING_2_STATUS_FIELD)
                    == STATUS_WAITING
                ):
                    self.station_sequence.append(2)
                if (
                    status.get(AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD)
                    == STATUS_WAITING
                ):
                    self.station_sequence.append(3)
                if (
                    status.get(AIRTABLE_WHIPPED_CREAM_STATUS_FIELD)
                    == STATUS_WAITING
                ):
                    self.station_sequence.append(4)
                if (
                    status.get(AIRTABLE_SPRINKLES_STATUS_FIELD)
                    == STATUS_WAITING
                ):
                    self.station_sequence.append(5)
                self.station_sequence.append(0)  # Pickup
                if not self.station_sequence:
                    self.get_logger().error("Route plan failed!")
                    self.state = RobotState.IDLE
                    self.current_order = None
                else:
                    self.current_sequence_index = 0
                    self.target_station_index = self.station_sequence[0]
                    self.initial_line_found = False
                    self.get_logger().info(
                        f"Route: {self.station_sequence}. Next: {self.target_station_index}"
                    )
                    self.state = RobotState.MOVING_TO_STATION
                    self.play_sound([(440, 100), (550, 100), (660, 100)])

            elif self.state == RobotState.LEAVING_STATION:
                elapsed = time.time() - self.leaving_station_start_time
                if elapsed < LEAVING_STATION_DURATION_SEC:
                    self.move_robot(BASE_DRIVE_SPEED, 0.0)
                else:
                    self.get_logger().info(
                        f"Finished leaving. Moving to {self.target_station_index}."
                    )
                    self.stop_moving()
                    self.initial_line_found = False
                    self.state = RobotState.MOVING_TO_STATION

            elif self.state == RobotState.MOVING_TO_STATION:
                if color_detected:
                    self.get_logger().info(
                        f"Color detected for {self.target_station_index}. Arriving."
                    )
                    self.play_sound([(523, 100), (659, 150)])
                    self.stop_moving()
                    self.state = RobotState.ARRIVED_AT_STATION
                    return
                if (
                    self.target_station_index == -1
                    or self.current_sequence_index >= len(self.station_sequence)
                ):
                    self.get_logger().error(f"MOVING: Invalid target/index.")
                    self.stop_moving()
                    self.state = RobotState.ERROR
                    return

                left_on, right_on = self.read_ir_sensors()
                if not left_on and not right_on:  # Both OFF line (LOW)
                    if not self.initial_line_found:
                        self.get_logger().info(
                            "Searching initial line (RIGHT)..."
                        )
                        self.move_robot(0.0, LOST_LINE_ROTATE_SPEED)
                    else:  # Line lost, simple right turn
                        self.get_logger().debug("Line lost, turning right...")
                        self.move_robot(0.0, -LOST_LINE_ROTATE_SPEED)
                else:  # At least one sensor ON line (HIGH)
                    if not self.initial_line_found:
                        self.get_logger().info("Initial line found!")
                        self.initial_line_found = True
                    # Normal Line Following (HIGH=ON)
                    if left_on and right_on:
                        self.move_robot(BASE_DRIVE_SPEED, 0.0)  # -> Straight
                    elif left_on and not right_on:
                        self.move_robot(
                            BASE_DRIVE_SPEED * TURN_FACTOR, BASE_ROTATE_SPEED
                        )  # -> Turn Right
                    elif not left_on and right_on:
                        self.move_robot(
                            BASE_DRIVE_SPEED * TURN_FACTOR, -BASE_ROTATE_SPEED
                        )  # -> Turn Left

            elif self.state == RobotState.ARRIVED_AT_STATION:
                self.stop_moving()
                if (
                    self.current_sequence_index < 0
                    or self.current_sequence_index >= len(self.station_sequence)
                ):
                    self.get_logger().error("ARRIVED: Invalid index.")
                    self.state = RobotState.ERROR
                    return
                idx = self.station_sequence[self.current_sequence_index]
                if idx not in STATION_INDEX_TO_FIELD:
                    self.get_logger().error(
                        f"ARRIVED: No field for index {idx}."
                    )
                    self.state = RobotState.ERROR
                    return
                field = STATION_INDEX_TO_FIELD[idx]
                if self.update_station_status(
                    self.current_order["record_id"], field, STATUS_ARRIVED
                ):
                    self.wait_start_time = time.time()
                    self.state = RobotState.WAITING_FOR_STATION_COMPLETION
                    self.get_logger().info(f"At {field}. Waiting for DONE.")

            elif self.state == RobotState.WAITING_FOR_STATION_COMPLETION:
                elapsed = time.time() - self.wait_start_time
                if elapsed > STATION_WAIT_TIMEOUT_SEC:
                    self.get_logger().warning(
                        f"WAIT TIMEOUT ({elapsed:.1f}s) for {self.target_station_index}."
                    )
                    self.play_sound([(330, 500), (220, 500)])
                    self.state = RobotState.STATION_TIMED_OUT
                    return
                if (
                    time.time() - self._last_airtable_check_time
                    >= AIRTABLE_POLL_RATE
                ):
                    self._last_airtable_check_time = time.time()
                    if (
                        self.current_sequence_index < 0
                        or self.current_sequence_index
                        >= len(self.station_sequence)
                    ):
                        self.get_logger().error("WAITING: Invalid index.")
                        self.state = RobotState.ERROR
                        return
                    idx = self.station_sequence[self.current_sequence_index]
                    field = STATION_INDEX_TO_FIELD.get(idx)
                    if not field:
                        self.get_logger().error(
                            f"WAITING: No field for index {idx}."
                        )
                        self.state = RobotState.ERROR
                        return
                    if self.wait_for_station_completion(
                        self.current_order["record_id"], field
                    ):
                        self.get_logger().info(f"Station {idx} ({field}) DONE.")
                        self.play_sound([(659, 150), (784, 200)])
                        last_idx = self.current_sequence_index
                        self.current_sequence_index += 1
                        if self.current_sequence_index >= len(
                            self.station_sequence
                        ):
                            self.get_logger().info("All stations complete.")
                            self.state = RobotState.ORDER_COMPLETE
                        else:
                            self.target_station_index = self.station_sequence[
                                self.current_sequence_index
                            ]
                            self.get_logger().info(
                                f"Leaving station {idx} before moving to {self.target_station_index}."
                            )
                            self.leaving_station_start_time = time.time()
                            self.state = RobotState.LEAVING_STATION
                            self.last_color_detection_times[idx] = 0.0

            elif self.state == RobotState.ORDER_COMPLETE:
                self.get_logger().info(
                    f"Order '{self.current_order['order_name']}' COMPLETE."
                )
                self.play_sound([(784, 150), (880, 150), (1047, 250)])
                self.stop_moving()
                self.current_order = None
                self.station_sequence = []
                self.current_sequence_index = -1
                self.target_station_index = -1
                self.initial_line_found = False
                self.state = RobotState.IDLE

            elif self.state == RobotState.ALL_ORDERS_COMPLETE:
                self.get_logger().info("No orders. Waiting...")
                self.play_sound([(440, 200), (440, 200)])
                self.stop_moving()
                time.sleep(5.0)
                self.state = RobotState.IDLE

            elif self.state == RobotState.STATION_TIMED_OUT:
                self.get_logger().error("STATION TIMED OUT.")
                self.stop_moving()
                if self.current_order:
                    self.get_logger().error(
                        f"Aborting order '{self.current_order['order_name']}'."
                    )
                self.current_order = None
                self.station_sequence = []
                self.current_sequence_index = -1
                self.target_station_index = -1
                self.initial_line_found = False
                self.state = RobotState.IDLE

            elif self.state == RobotState.AIRTABLE_ERROR:
                self.get_logger().error("AIRTABLE ERROR. Halting.")
                self.play_sound([(330, 300), (330, 300), (330, 300)])
                self.stop_moving()

        except Exception as e:
            self.get_logger().error(
                f"State machine exception: {e}", exc_info=True
            )
            self.state = RobotState.ERROR
            self.stop_moving()

        # Display Update
        finally:
            if self.debug_windows:
                try:
                    if display_frame is not None:
                        cv2.imshow("Camera Feed", display_frame)
                    if mask_frame is not None:
                        cv2.imshow("Color Detection Mask", mask_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        self.get_logger().info("Quit key pressed.")
                        self.state = RobotState.ERROR
                        self.stop_moving()
                        rclpy.try_shutdown()
                except Exception as e:
                    self.get_logger().error(f"Display update error: {e}")


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
        ]:
            executor = SingleThreadedExecutor()
            executor.add_node(node)
            node.get_logger().info("Starting ROS2 executor spin...")
            executor.spin()
        else:
            node.get_logger().fatal(f"Node init failed: {node.state.name}.")
    except KeyboardInterrupt:
        print("KeyboardInterrupt.")
    except Exception as e:
        print(f"FATAL Error: {e}")
    finally:
        if node:
            node.get_logger().info("Cleanup...")
            node.stop_moving()
            if executor:
                executor.shutdown()
                node.cleanup_hardware()
            if node.debug_windows:
                cv2.destroyAllWindows()
                node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
