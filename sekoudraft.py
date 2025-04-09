#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor

import time
from enum import Enum, auto
import math
import cv2  # OpenCV for image processing
import numpy as np
from picamera2 import Picamera2  # Pi Camera library
from libcamera import controls  # For camera controls like autofocus
import requests  # For Airtable API calls
import json  # For Airtable API calls

# iRobot Create 3 specific messages
from irobot_create_msgs.action import DriveDistance, RotateAngle
from irobot_create_msgs.msg import AudioNoteVector, AudioNote
from builtin_interfaces.msg import Duration

# --- Configuration Constants ---
# --- Airtable Configuration ---
AIRTABLE_API_TOKEN = "your_token"
AIRTABLE_BASE_ID = "your_base_id"
AIRTABLE_TABLE_NAME = "PancakesOrders" 
AIRTABLE_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
AIRTABLE_HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_API_TOKEN}",
    "Content-Type": "application/json",
}

# --- Status values expected in the AIRTABLE_STATUS_FIELD ---
STATUS_PENDING = "Pending"
STATUS_AT_ORDER = "At Order Station"
STATUS_AT_BATTER = "At Batter Station"
STATUS_AT_TOPPING_1 = "At Topping 1"
STATUS_AT_TOPPING_2 = "At Topping 2"
STATUS_AT_TOPPING_3 = "At Topping 3"
STATUS_COMPLETED = "Completed"
STATUS_FAILED = "Failed"

# --- Camera Configuration ---
CAMERA_RESOLUTION = (640, 480)

# --- Color Detection Configuration ---
STATION_COLORS_HSV = {
    0: {"name": "Order Station", "process_time": 2.0, "hsv_lower": (100, 100, 50), "hsv_upper": (130, 255, 255)},  # Blue
    1: {"name": "Batter Station", "process_time": 5.0, "hsv_lower": (0, 100, 100), "hsv_upper": (10, 255, 255)},   # Red
    2: {"name": "Topping 1", "process_time": 3.0, "hsv_lower": (20, 100, 100), "hsv_upper": (40, 255, 255)},       # Yellow
    3: {"name": "Topping 2", "process_time": 3.0, "hsv_lower": (50, 100, 50), "hsv_upper": (80, 255, 255)},        # Green
    4: {"name": "Topping 3", "process_time": 3.0, "hsv_lower": (140, 100, 50), "hsv_upper": (170, 255, 255)},       # Pink
}

NUM_STATIONS = len(STATION_COLORS_HSV)

# --- Navigation Parameters ---
COLOR_DETECTION_THRESHOLD = 500
DRIVE_INCREMENT = 0.03          # Meters to drive forward when searching for color
DRIVE_SPEED = 0.1  # m/s
ROTATE_SPEED = 0.8  # rad/s

# --- State Machine Definition ---
class RobotState(Enum):
    IDLE = auto()
    FETCHING_ORDER = auto()
    MOVING_TO_STATION = auto()
    PROCESSING_AT_STATION = auto()
    RETURNING_TO_START = auto()
    STOPPING_BEFORE_PROCESS = auto()
    STOPPING_BEFORE_IDLE = auto()
    CYCLE_COMPLETE = auto()
    ERROR = auto()
    CAMERA_ERROR = auto()
    AIRTABLE_ERROR = auto()
    LINE_FOLLOWING = auto()
    LINE_LOST = auto()

# --- Main Robot Control Class ---
class PancakeRobotNode(Node):
    """
    Manages the iCreate3 robot for pancake making. Fetches orders from Airtable,
    navigates using camera color detection, simulates processing, and updates Airtable status.
    """
    def __init__(self):
        super().__init__('pancake_robot_node')
        self.get_logger().info("Pancake Robot Node Initializing...")

        # --- Camera Setup ---
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(main={"size": CAMERA_RESOLUTION})
            self.picam2.configure(config)
            self.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
            self.picam2.start()
            time.sleep(2)  # Allow camera to initialize and focus
            self.get_logger().info("Pi Camera initialized successfully.")
        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to initialize Pi Camera: {e}")
            self.state = RobotState.CAMERA_ERROR
            return  # Cannot proceed without camera

        # --- Robot State Initialization ---
        self.state = RobotState.IDLE
        self.current_station_index = 0  # Start at the 'Order Station'
        self.target_station_index = 0  # Station we are currently moving towards
        self.pancakes_made_count = 0  # Counter for completed pancakes in this run
        self.current_order = None  # Stores details of the order being processed

        # --- ROS2 Action Clients and Publisher Initialization ---
        self.drive_client = ActionClient(self, DriveDistance, '/drive_distance')
        self.rotate_client = ActionClient(self, RotateAngle, '/rotate_angle')
        self.audio_publisher = self.create_publisher(AudioNoteVector, '/cmd_audio', 10)

        # --- Wait for Action Servers ---
        if not self.drive_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('FATAL: DriveDistance action server not available!')
            self.state = RobotState.ERROR
            return
        if not self.rotate_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('FATAL: RotateAngle action server not available!')
            self.state = RobotState.ERROR
            return
        self.get_logger().info("Action servers found.")

        # Main control loop timer
        self.timer_period = 0.2  # seconds (Check camera/state frequently)
        self.timer = self.create_timer(self.timer_period, self.control_loop)

        self.get_logger().info("Pancake Robot Node Initialized and Ready.")
        self.play_sound([(440, 200), (550, 300)])  # Play startup sound

    # --- Camera Feed Handling ---
    def capture_image(self):
        """Capture and flip the image from the camera."""
        image = self.picam2.capture_array()
        image = cv2.flip(image, 0)  # Flip vertically
        return image

    def detect_target_color(self, target_idx):
        """Checks if the color marker for the target station index is detected."""
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.AIRTABLE_ERROR]:
            return False

        if target_idx not in STATION_COLORS_HSV:
            self.get_logger().error(f"Invalid target index {target_idx} for color detection.")
            return False

        color_info = STATION_COLORS_HSV[target_idx]
        lower_bound = np.array(color_info["hsv_lower"])
        upper_bound = np.array(color_info["hsv_upper"])
        target_color_name = color_info["name"]

        try:
            # Capture image (flip image if needed)
            image = self.capture_image()

            # Convert to HSV
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Create a mask for the target color range
            mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

            # Show the processed image and detected colors
            cv2.imshow("Detected Colors", mask)
            cv2.imshow("Camera Feed", image)  # Display the flipped feed

            # Count non-zero pixels in the mask (pixels matching the color)
            detected_pixels = cv2.countNonZero(mask)
            self.get_logger().debug(f"Detecting {target_color_name}: Pixels={detected_pixels} (Threshold={COLOR_DETECTION_THRESHOLD})")

            return detected_pixels > COLOR_DETECTION_THRESHOLD

        except cv2.error as cv_err:
            self.get_logger().error(f"OpenCV error during color detection: {cv_err}")
            return False
        except Exception as e:
            self.get_logger().error(f"Unexpected error during color detection: {e}")
            return False

    def control_loop(self):
        """The core state machine logic, called periodically by the timer."""
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.CYCLE_COMPLETE, RobotState.AIRTABLE_ERROR]:
            return

        self.get_logger().debug(f"Current State: {self.state.name}")

        if self.state == RobotState.IDLE:
            self.get_logger().info("State: IDLE. Checking for new orders...")
            self.state = RobotState.FETCHING_ORDER
            self.current_order = None
            self.current_station_index = 0
            self.target_station_index = 0

        elif self.state == RobotState.FETCHING_ORDER:
            # Fetch order logic here
            self.get_logger().info("Fetching orders from Airtable...")
            # Simulate fetching orders here, then transition to MOVING_TO_STATION

            self.state = RobotState.MOVING_TO_STATION  # Transition to next state

        elif self.state == RobotState.MOVING_TO_STATION:
            if self.detect_target_color(self.target_station_index):
                self.get_logger().info(f"Target color detected for Station {self.target_station_index}")
                self.state = RobotState.STOPPING_BEFORE_PROCESS
            else:
                self.get_logger().info("Target color not detected, continuing search.")
                # Logic to continue moving until the target color is found

        elif self.state == RobotState.STOPPING_BEFORE_PROCESS:
            self.get_logger().info("Stopping before processing...")
            # Logic for stopping before processing
            self.state = RobotState.PROCESSING_AT_STATION

        # Further state transitions and actions...

    # --- Cleanup ---
    def cleanup(self):
        """Stops the robot and the camera feed."""
        self.picam2.stop()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = PancakeRobotNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt detected, shutting down.")
    finally:
        node.cleanup()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
