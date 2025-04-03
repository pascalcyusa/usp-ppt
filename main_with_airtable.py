#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor # To wait for actions

import time
from enum import Enum, auto
import math
import cv2 # OpenCV for image processing
import numpy as np
from picamera2 import Picamera2 # Pi Camera library
from libcamera import controls # For camera controls like autofocus
import requests # For Airtable API calls
import json     # For Airtable API calls

# iRobot Create 3 specific messages
from irobot_create_msgs.action import DriveDistance, RotateAngle
from irobot_create_msgs.msg import InterfaceButtons, IrIntensityVector # Example sensor msgs
from builtin_interfaces.msg import Duration
from irobot_create_msgs.msg import AudioNoteVector, AudioNote

# --- Configuration Constants ---

# --- Airtable Configuration ---
# !!! SECURITY WARNING: Avoid hardcoding tokens in production. Use environment variables! !!!
AIRTABLE_API_TOKEN = "patU8FCP20PtCQfA2.32f6fc12175797c8c6f67985395aef9fb168d3c177ecc092feec853724fd35bd" # Replace with your actual token (starts with 'pat...')
AIRTABLE_BASE_ID = "https://api.airtable.com/v0/app7psTY3i95TjiYI/Table%201"             # Replace with your Base ID (starts with 'app...')
AIRTABLE_TABLE_NAME = "PancakeOrders"       # Replace with your Table Name or ID (starts with 'tbl...')
AIRTABLE_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
AIRTABLE_HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_API_TOKEN}",
    "Content-Type": "application/json",
}
# Field names in your Airtable base (Case sensitive!)
AIRTABLE_ORDER_ID_FIELD = "OrderID"
AIRTABLE_STATUS_FIELD = "Status"
AIRTABLE_TOPPINGS_FIELD = "Toppings" # Optional

# Status values expected in Airtable
STATUS_PENDING = "Pending"
STATUS_STARTED = "Started"
STATUS_READY = "Ready"


# --- Camera Configuration ---
CAMERA_RESOLUTION = (640, 480) # Width, Height
# --- Color Detection Configuration ---
# Define HSV Lower and Upper bounds for each station's target color
STATION_COLORS_HSV = {
    3: {"name": "Order Station", "process_time": 2.0,
        "hsv_lower": (100, 100, 50), "hsv_upper": (130, 255, 255)}, # Example: Blue (for return)
    1: {"name": "Batter/Cook", "process_time": 5.0,
        "hsv_lower": (0, 100, 100), "hsv_upper": (10, 255, 255)},   # Example: Red
    2: {"name": "Topping 1", "process_time": 3.0,
        "hsv_lower": (20, 100, 100), "hsv_upper": (40, 255, 255)},   # Example: Yellow
    0: {"name": "Topping 2", "process_time": 3.0,
        "hsv_lower": (50, 100, 50), "hsv_upper": (80, 255, 255)},   # Example: Green
    4: {"name": "Topping 3", "process_time": 3.0,
        "hsv_lower": (140, 100, 50), "hsv_upper": (170, 255, 255)},  # Example: Magenta/Pink
}
NUM_STATIONS = len(STATION_COLORS_HSV)

# Threshold for color detection (minimum number of white pixels in the mask)
COLOR_DETECTION_THRESHOLD = 500
DRIVE_INCREMENT = 0.03

# Robot Control Parameters
DRIVE_SPEED = 0.1
ROTATE_SPEED = 0.8
PANCAKES_TO_MAKE = 3 # How many times to run the full cycle (or until no more orders)

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
    AIRTABLE_ERROR = auto() # New state for Airtable issues

# --- Main Robot Control Class ---
class PancakeRobotNode(Node):
    """
    Manages the iCreate3 robot for the pancake making process.
    Fetches orders from Airtable, uses camera color detection for navigation,
    and updates order status in Airtable.
    """
    def __init__(self):
        super().__init__('pancake_robot_node')
        self.get_logger().info("Pancake Robot Node Initializing...")

        # Airtable Check (Basic check if config looks okay)
        if "YOUR_" in AIRTABLE_API_TOKEN or "YOUR_" in AIRTABLE_BASE_ID or "YOUR_" in AIRTABLE_TABLE_NAME:
             self.get_logger().warn("Airtable configuration seems incomplete. Please update constants.")
             # Decide if this is fatal or just a warning
             # self.state = RobotState.AIRTABLE_ERROR
             # return

        # Robot State
        self.state = RobotState.IDLE
        self.current_station_index = 0
        self.target_station_index = 0
        self.pancakes_made = 0
        self.current_order = None # Will store fetched order details {'record_id': ..., 'pancake_id': ..., 'toppings': ...}

        # Action Clients
        self.drive_client = ActionClient(self, DriveDistance, '/drive_distance')
        self.rotate_client = ActionClient(self, RotateAngle, '/rotate_angle')
        self.audio_publisher = self.create_publisher(AudioNoteVector, '/cmd_audio', 10)

        # --- Camera Setup ---
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(main={"size": CAMERA_RESOLUTION})
            self.picam2.configure(config)
            self.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous, "LensPosition": 0.0})
            self.picam2.start()
            time.sleep(2)
            self.get_logger().info("Pi Camera initialized successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize Pi Camera: {e}")
            self.state = RobotState.CAMERA_ERROR
            return

        # Wait for Action Servers
        if not self.drive_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('DriveDistance action server not available!')
            self.state = RobotState.ERROR
            return
        if not self.rotate_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('RotateAngle action server not available!')
            self.state = RobotState.ERROR
            return

        self.get_logger().info("Action servers found.")

        # Main control loop timer
        self.timer_period = 0.2
        self.timer = self.create_timer(self.timer_period, self.control_loop)

        self.get_logger().info("Pancake Robot Node Initialized.")
        self.play_sound([(440, 200), (550, 300)])

    # --- Movement Actions (drive_distance, stop_moving, rotate_angle) ---
    # ... (Keep the implementations from the previous version) ...
    def drive_distance(self, distance):
        """Sends a DriveDistance goal and waits for completion."""
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.AIRTABLE_ERROR]: return False

        goal_msg = DriveDistance.Goal()
        goal_msg.distance = float(distance)
        goal_msg.max_translation_speed = DRIVE_SPEED

        self.get_logger().debug(f"Sending Drive goal: distance={distance:.3f}m")
        executor = SingleThreadedExecutor()
        rclpy.spin_once(self, executor=executor, timeout_sec=0.1)
        future = self.drive_client.send_goal_async(goal_msg)

        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, executor=executor, timeout_sec=0.1)

        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error('Drive goal rejected or failed to get handle.')
            return False

        result_future = goal_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
             rclpy.spin_once(self, executor=executor, timeout_sec=0.1)

        result = result_future.result()
        if result and result.status == result.GoalStatus.STATUS_SUCCEEDED:
             self.get_logger().debug(f'Drive completed.')
             return True
        else:
             status_str = result.status if result else 'Unknown'
             self.get_logger().warn(f'Drive action did not succeed. Status: {status_str}')
             return False

    def stop_moving(self):
        """Sends a DriveDistance goal with 0 distance to stop the robot."""
        self.get_logger().info("Sending stop command.")
        return self.drive_distance(0.0)

    def rotate_angle(self, angle):
        """Sends a RotateAngle goal and waits for completion."""
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.AIRTABLE_ERROR]: return False

        goal_msg = RotateAngle.Goal()
        goal_msg.angle = float(angle) # Radians
        goal_msg.max_rotation_speed = ROTATE_SPEED
        self.get_logger().info(f"Rotating angle: {math.degrees(angle):.1f} degrees")
        executor = SingleThreadedExecutor()
        rclpy.spin_once(self, executor=executor, timeout_sec=0.1)
        future = self.rotate_client.send_goal_async(goal_msg)

        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, executor=executor, timeout_sec=0.1)

        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error('Rotate goal rejected or failed to get handle.')
            return False

        self.get_logger().info('Rotate goal accepted.')
        result_future = goal_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
             rclpy.spin_once(self, executor=executor, timeout_sec=0.1)

        result = result_future.result()
        if result and result.status == result.GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Rotate completed.')
            return True
        else:
            status_str = result.status if result else 'Unknown'
            self.get_logger().warn(f'Rotate action did not succeed. Status: {status_str}')
            return False

    # --- Color Detection (detect_target_color) ---
    # ... (Keep the implementation from the previous version) ...
    def detect_target_color(self, target_idx):
        """Checks if the color for the target station index is detected by the camera."""
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.AIRTABLE_ERROR]: return False

        if target_idx not in STATION_COLORS_HSV:
            self.get_logger().error(f"Invalid target index {target_idx} for color detection.")
            return False

        color_info = STATION_COLORS_HSV[target_idx]
        lower_bound = np.array(color_info["hsv_lower"])
        upper_bound = np.array(color_info["hsv_upper"])
        target_color_name = color_info["name"]

        try:
            image = self.picam2.capture_array()
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
            detected_pixels = cv2.countNonZero(mask)
            self.get_logger().debug(f"Detecting {target_color_name}: Pixels={detected_pixels} (Threshold={COLOR_DETECTION_THRESHOLD})")
            return detected_pixels > COLOR_DETECTION_THRESHOLD
        except Exception as e:
            self.get_logger().error(f"Error during color detection: {e}")
            return False

    # --- Station & Process Simulation ---
    # ... (Keep simulate_processing) ...
    def simulate_processing(self, duration):
        """Placeholder to simulate work at a station."""
        self.get_logger().info(f"Processing at {STATION_COLORS_HSV[self.current_station_index]['name']} for {duration:.1f}s...")
        time.sleep(duration)
        self.get_logger().info("Processing complete.")

    # --- Airtable Communication ---
    def fetch_order_from_airtable(self):
        """Fetches the oldest 'Pending' order from Airtable."""
        self.get_logger().info("Attempting to fetch order from Airtable...")
        params = {
            "maxRecords": 1,
            "view": "Grid view", # Or your specific view name
            "filterByFormula": f"({{Status}}='{STATUS_PENDING}')",
             # Optional: Sort by 'Created' time field if you have one, oldest first
            # "sort[0][field]": "CreatedTime",
            # "sort[0][direction]": "asc"
        }
        try:
            response = requests.get(AIRTABLE_URL, headers=AIRTABLE_HEADERS, params=params, timeout=10)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            data = response.json()
            records = data.get("records", [])

            if not records:
                self.get_logger().info("No pending orders found in Airtable.")
                return None

            # Process the first record found
            record = records[0]
            record_id = record.get("id")
            fields = record.get("fields", {})
            order_id = fields.get(AIRTABLE_ORDER_ID_FIELD)
            toppings = fields.get(AIRTABLE_TOPPINGS_FIELD, "") # Handle missing toppings field

            if not record_id or not order_id:
                 self.get_logger().error(f"Fetched record missing 'id' or '{AIRTABLE_ORDER_ID_FIELD}'. Record: {record}")
                 return None

            self.get_logger().info(f"Fetched order: ID={order_id}, RecordID={record_id}, Toppings='{toppings}'")
            return {
                "record_id": record_id,
                "pancake_id": order_id,
                "toppings": toppings.split(',') if isinstance(toppings, str) else toppings # Basic parsing if toppings is comma-sep string
            }

        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Airtable request failed: {e}")
            # Consider setting state to AIRTABLE_ERROR temporarily
            # self.state = RobotState.AIRTABLE_ERROR
            return None
        except Exception as e:
             self.get_logger().error(f"Error processing Airtable response: {e}")
             return None

    def update_order_status(self, record_id, new_status):
        """Updates the status of a specific order in Airtable using its record_id."""
        if not record_id:
            self.get_logger().error("Cannot update status: record_id is missing.")
            return False

        self.get_logger().info(f"Updating Airtable record {record_id} status to '{new_status}'...")
        url = f"{AIRTABLE_URL}/{record_id}"
        payload = json.dumps({
            "fields": {
                AIRTABLE_STATUS_FIELD: new_status
            }
        })

        try:
            response = requests.patch(url, headers=AIRTABLE_HEADERS, data=payload, timeout=10)
            response.raise_for_status() # Raise an exception for bad status codes

            self.get_logger().info(f"Airtable record {record_id} updated successfully.")
            return True

        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Airtable update request failed for {record_id}: {e}")
            # self.state = RobotState.AIRTABLE_ERROR # Consider consequences
            return False
        except Exception as e:
             self.get_logger().error(f"Error processing Airtable update response for {record_id}: {e}")
             return False

    # --- Sound Utility (play_sound) ---
    # ... (Keep the implementation from the previous version) ...
    def play_sound(self, notes):
        """Publishes a sequence of notes to the audio topic."""
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.AIRTABLE_ERROR]: return

        audio_msg = AudioNoteVector()
        audio_msg.append = False

        for freq, duration_ms in notes:
            note = AudioNote()
            note.frequency = int(freq)
            note.max_runtime = Duration(sec=0, nanosec=int(duration_ms * 1_000_000))
            audio_msg.notes.append(note)

        self.get_logger().debug(f"Playing sound: {notes}")
        self.audio_publisher.publish(audio_msg)

    # --- Main Control Loop ---
    def control_loop(self):
        """The core state machine logic, called periodically by the timer."""

        # Handle terminal/error states first
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.CYCLE_COMPLETE, RobotState.AIRTABLE_ERROR]:
            if self.state == RobotState.ERROR:
                self.get_logger().error("Robot in ERROR state. Halting.", throttle_duration_sec=5)
            elif self.state == RobotState.CAMERA_ERROR:
                self.get_logger().error("Robot in CAMERA_ERROR state. Halting.", throttle_duration_sec=5)
            elif self.state == RobotState.AIRTABLE_ERROR:
                 self.get_logger().error("Robot in AIRTABLE_ERROR state. Check connection/config. Halting.", throttle_duration_sec=5)
            # self.timer.cancel() # Consider stopping timer in error states
            return

        self.get_logger().debug(f"State: {self.state.name}, Current Station: {self.current_station_index}, Target: {self.target_station_index}, Pancakes Made: {self.pancakes_made}")

        # --- State Transitions ---
        if self.state == RobotState.IDLE:
            # Instead of fixed number, we try to fetch an order
            self.get_logger().info("State: IDLE. Checking for new orders...")
            self.state = RobotState.FETCHING_ORDER

        elif self.state == RobotState.FETCHING_ORDER:
            fetched_order = self.fetch_order_from_airtable()
            if fetched_order:
                self.current_order = fetched_order # Store {'record_id': ..., 'pancake_id': ...}
                # Update status to Started immediately after fetching
                if self.update_order_status(self.current_order["record_id"], STATUS_STARTED):
                    self.current_station_index = 0 # Explicitly start at 0
                    self.target_station_index = 1 # Target first processing station
                    self.get_logger().info(f"Order {self.current_order['pancake_id']} started. Moving from {STATION_COLORS_HSV[self.current_station_index]['name']} to find {STATION_COLORS_HSV[self.target_station_index]['name']} color.")
                    self.state = RobotState.MOVING_TO_STATION
                else:
                    self.get_logger().error(f"Failed to update status for order {self.current_order['pancake_id']} to '{STATUS_STARTED}'. Retrying fetch cycle.")
                    self.current_order = None # Clear failed order
                    self.state = RobotState.IDLE # Go back to idle to retry fetch later
                    time.sleep(5) # Wait before immediate retry
            else:
                # No orders found, or error during fetch
                self.get_logger().info("No pending orders found or error during fetch. Cycle complete for now.")
                # Check if *any* pancakes were made in this run before declaring "all done"
                if self.pancakes_made > 0:
                     self.play_sound([(600,100), (700,100), (800, 300)]) # Sound for finishing a batch
                else:
                     self.play_sound([(400, 500)]) # Sound for finding no orders
                self.timer.cancel() # Stop the timer if no more orders
                self.state = RobotState.CYCLE_COMPLETE

        # --- MOVING_TO_STATION ---
        elif self.state == RobotState.MOVING_TO_STATION:
            if self.detect_target_color(self.target_station_index):
                self.get_logger().info(f"Target color for Station {self.target_station_index} detected!")
                self.play_sound([(500, 150)])
                self.state = RobotState.STOPPING_BEFORE_PROCESS
            else:
                self.get_logger().debug(f"Color not detected, driving forward {DRIVE_INCREMENT}m...")
                if not self.drive_distance(DRIVE_INCREMENT):
                    self.get_logger().warn("Failed to drive increment, retrying detection/drive.")
                    # Consider adding a failure counter or timeout

        # --- STOPPING_BEFORE_PROCESS ---
        elif self.state == RobotState.STOPPING_BEFORE_PROCESS:
            if self.stop_moving():
                self.current_station_index = self.target_station_index
                self.state = RobotState.PROCESSING_AT_STATION
            else:
                 self.get_logger().error("Failed to stop before processing! Retrying stop.")

        # --- PROCESSING_AT_STATION ---
        elif self.state == RobotState.PROCESSING_AT_STATION:
            station_info = STATION_COLORS_HSV[self.current_station_index]
            self.simulate_processing(station_info["process_time"])

            if self.current_station_index < (NUM_STATIONS - 1):
                self.target_station_index = self.current_station_index + 1
                self.get_logger().info(f"Processing done. Moving from {station_info['name']} to find {STATION_COLORS_HSV[self.target_station_index]['name']} color.")
                self.state = RobotState.MOVING_TO_STATION
            else: # Finished the last station
                 # Update status to Ready
                 if self.update_order_status(self.current_order["record_id"], STATUS_READY):
                     self.get_logger().info(f"Order {self.current_order['pancake_id']} marked as '{STATUS_READY}'. Returning to start.")
                     self.target_station_index = 0 # Target is start station color
                     self.state = RobotState.RETURNING_TO_START
                 else:
                     self.get_logger().error(f"Failed to update status for order {self.current_order['pancake_id']} to '{STATUS_READY}'. Cannot proceed. Entering Airtable Error state.")
                     self.state = RobotState.AIRTABLE_ERROR # Treat failure to update final status as critical

        # --- RETURNING_TO_START ---
        elif self.state == RobotState.RETURNING_TO_START:
            if self.detect_target_color(self.target_station_index): # Target index is 0
                self.get_logger().info(f"Return color for Station {self.target_station_index} detected!")
                self.play_sound([(800, 100), (700, 100), (600, 200)])
                self.state = RobotState.STOPPING_BEFORE_IDLE
            else:
                self.get_logger().debug(f"Return color not detected, driving forward {DRIVE_INCREMENT}m...")
                if not self.drive_distance(DRIVE_INCREMENT):
                    self.get_logger().warn("Failed to drive increment while returning, retrying detection/drive.")

        # --- STOPPING_BEFORE_IDLE ---
        elif self.state == RobotState.STOPPING_BEFORE_IDLE:
             if self.stop_moving():
                self.current_station_index = 0 # Officially back at start
                self.pancakes_made += 1
                self.get_logger().info(f"Pancake order {self.current_order['pancake_id']} cycle complete. Stopped at start. Returning to IDLE to check for more orders.")
                self.current_order = None # Clear current order
                self.state = RobotState.IDLE # Go back to IDLE to fetch next order
             else:
                 self.get_logger().error("Failed to stop before idle! Retrying stop.")


    def shutdown_camera(self):
        """Safely stops the camera."""
        if hasattr(self, 'picam2') and self.picam2:
            try:
                self.get_logger().info("Stopping Pi Camera...")
                self.picam2.stop()
                self.get_logger().info("Pi Camera stopped.")
            except Exception as e:
                self.get_logger().error(f"Error stopping camera: {e}")

# --- Main Function ---
def main(args=None):
    rclpy.init(args=args)
    pancake_robot_node = None
    try:
        pancake_robot_node = PancakeRobotNode()

        # Check for initialization errors
        if pancake_robot_node.state not in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.AIRTABLE_ERROR]:
            rclpy.spin(pancake_robot_node)
        else:
            pancake_robot_node.get_logger().error("Node initialization failed. Shutting down.")

    except KeyboardInterrupt:
        if pancake_robot_node:
            pancake_robot_node.get_logger().info("Keyboard interrupt detected, shutting down.")
    except Exception as e:
         if pancake_robot_node:
             pancake_robot_node.get_logger().error(f"An unexpected error occurred during spin: {e}")
             import traceback
             traceback.print_exc()
         else:
             print(f"An unexpected error occurred before node was fully initialized: {e}")
    finally:
        if pancake_robot_node:
            pancake_robot_node.get_logger().info("Initiating cleanup...")
            pancake_robot_node.shutdown_camera()
            # Attempt graceful stop if node exists and actions might be running
            try:
                 if pancake_robot_node.state not in [RobotState.IDLE, RobotState.CYCLE_COMPLETE] and pancake_robot_node.drive_client.server_is_ready():
                      pancake_robot_node.get_logger().info("Attempting to stop robot movement...")
                      pancake_robot_node.stop_moving() # Try to stop if interrupted mid-movement
            except Exception as stop_e:
                 pancake_robot_node.get_logger().error(f"Error during final stop attempt: {stop_e}")

            pancake_robot_node.destroy_node()
            pancake_robot_node.get_logger().info("Node destroyed.")
        if rclpy.ok():
             rclpy.shutdown()
        print("ROS Cleanup complete.")


if __name__ == '__main__':
    main()