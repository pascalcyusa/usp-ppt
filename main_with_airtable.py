#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor

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
# ==============================================================================
# !!! IMPORTANT: REPLACE THESE VALUES WITH YOUR ACTUAL AIRTABLE DETAILS !!!
# ==============================================================================
# Go to https://airtable.com/create/tokens to create a token
# Scopes needed: data.records:read, data.records:write
# Access needed: Grant access to the specific Base below
AIRTABLE_API_TOKEN = "patjjcSjgPC5BCHTe.035e36f241e324d484e939a6e2a06e8ae986f8ba0e9b9a7acb6dac928c9d5d76"  

# Find Base ID in the URL when viewing your base (e.g., https://airtable.com/YOUR_BASE_ID/...)
AIRTABLE_BASE_ID = "app7psTY3i95TjiYI/PancakesOrders"              

# The exact name of the table containing your pancake orders
AIRTABLE_TABLE_NAME = "PancakesOrders"                
# ==============================================================================

# --- Construct Airtable URL and Headers from Config Above ---
# Ensure placeholders are replaced before using these
AIRTABLE_URL = f"https://api.airtable.com/v0/app7psTY3i95TjiYI/PancakesOrders"
AIRTABLE_HEADERS = {
    "Authorization": f"Bearer patjjcSjgPC5BCHTe.035e36f241e324d484e939a6e2a06e8ae986f8ba0e9b9a7acb6dac928c9d5d76",
    "Content-Type": "application/json",
}

# --- Field names in your Airtable base (MUST match exactly, case-sensitive) ---
# Mapping Airtable Columns to Script Variables:
AIRTABLE_NAME_COLUMN = "Name"          # Airtable column to fetch for 'pancake_id' (OrderID)
AIRTABLE_TOPPINGS_COLUMNS = ["sprinkles", "Whipped Cream", "Chocolate Chips"]  # Columns for toppings
AIRTABLE_STATUS_FIELD = "Status"       # The field storing "Done", "In Progress", "To do"

# --- Status values expected in the AIRTABLE_STATUS_FIELD ---
STATUS_PENDING = "To do"
STATUS_STARTED = "In Progress"
STATUS_READY = "Done"

# --- Camera Configuration ---
CAMERA_RESOLUTION = (640, 480) # Width, Height

# --- Color Detection Configuration ---
# Define HSV Lower and Upper bounds for each station's target color marker
# --> Tune these values for your specific colors and lighting conditions <--
STATION_COLORS_HSV = {
    # Index 0: Color marker to detect when returning to the start/order station
    0: {"name": "Order Station", "process_time": 2.0,
        "hsv_lower": (100, 100, 50), "hsv_upper": (130, 255, 255)}, # Example: Blue

    # Index 1-4: Color markers for processing stations
    1: {"name": "Batter/Cook", "process_time": 5.0,
        "hsv_lower": (0, 100, 100), "hsv_upper": (10, 255, 255)},   # Example: Red
    2: {"name": "Topping 1", "process_time": 3.0,
        "hsv_lower": (20, 100, 100), "hsv_upper": (40, 255, 255)},   # Example: Yellow
    3: {"name": "Topping 2", "process_time": 3.0,
        "hsv_lower": (50, 100, 50), "hsv_upper": (80, 255, 255)},   # Example: Green
    4: {"name": "Topping 3", "process_time": 3.0,
        "hsv_lower": (140, 100, 50), "hsv_upper": (170, 255, 255)},  # Example: Magenta/Pink
}
NUM_STATIONS = len(STATION_COLORS_HSV) # Should be 5 (indices 0 to 4)

# --- Navigation Parameters ---
COLOR_DETECTION_THRESHOLD = 500 # Min pixels of target color to trigger detection (tune this!)
DRIVE_INCREMENT = 0.03          # Meters to drive forward when searching for color

# --- Robot Control Parameters ---
DRIVE_SPEED = 0.1  # m/s
ROTATE_SPEED = 0.8  # rad/s

# --- State Machine Definition ---
class RobotState(Enum):
    IDLE = auto()
    FETCHING_ORDER = auto()
    MOVING_TO_STATION = auto()
    PROCESSING_AT_STATION = auto()
    RETURNING_TO_START = auto()
    STOPPING_BEFORE_PROCESS = auto() # Intermediate state to ensure stop before processing
    STOPPING_BEFORE_IDLE = auto()    # Intermediate state to ensure stop before idling
    CYCLE_COMPLETE = auto()          # Finished all available orders
    ERROR = auto()                   # General error
    CAMERA_ERROR = auto()            # Camera initialization failed
    AIRTABLE_ERROR = auto()          # Airtable communication error or config issue

# --- Main Robot Control Class ---
class PancakeRobotNode(Node):
    """
    Manages the iCreate3 robot for pancake making. Fetches orders from Airtable,
    navigates using camera color detection, simulates processing, and updates Airtable status.
    """
    def __init__(self):
        super().__init__('pancake_robot_node')
        self.get_logger().info("Pancake Robot Node Initializing...")

        # Airtable Configuration Check
        if "YOUR_" in AIRTABLE_API_TOKEN or "YOUR_" in AIRTABLE_BASE_ID or AIRTABLE_TABLE_NAME == "PancakeOrders":
             self.get_logger().warn("="*60)
             self.get_logger().warn("!!! POSSIBLE AIRTABLE CONFIG ISSUE DETECTED !!!")
             self.get_logger().warn("Please verify AIRTABLE_API_TOKEN, AIRTABLE_BASE_ID, and AIRTABLE_TABLE_NAME constants.")
             self.get_logger().warn("If configuration is correct, you can ignore this warning.")
             self.get_logger().warn("="*60)
             # Decide if you want to halt on potential config error, e.g.:
             # self.state = RobotState.AIRTABLE_ERROR
             # return

        # Robot State Initialization
        self.state = RobotState.IDLE
        self.current_station_index = 0 # Start at the 'Order Station'
        self.target_station_index = 0 # Station we are currently moving towards
        self.pancakes_made_count = 0  # Counter for completed pancakes in this run
        self.current_order = None # Stores details of the order being processed {'record_id': ..., 'pancake_id': ..., 'toppings': ...}

        # ROS2 Action Clients and Publisher Initialization
        self.drive_client = ActionClient(self, DriveDistance, '/drive_distance')
        self.rotate_client = ActionClient(self, RotateAngle, '/rotate_angle')
        self.audio_publisher = self.create_publisher(AudioNoteVector, '/cmd_audio', 10)

        # --- Camera Setup ---
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(main={"size": CAMERA_RESOLUTION})
            self.picam2.configure(config)
            # Enable continuous autofocus, start near focus
            self.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous, "LensPosition": 0.0})
            self.picam2.start()
            time.sleep(2) # Allow camera to initialize and focus
            self.get_logger().info("Pi Camera initialized successfully.")
        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to initialize Pi Camera: {e}")
            self.state = RobotState.CAMERA_ERROR
            return # Cannot proceed without camera

        # Wait for ROS2 Action Servers
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
        self.play_sound([(440, 200), (550, 300)]) # Play startup sound

    # --- Movement Actions ---
    def drive_distance(self, distance):
        """Sends a DriveDistance goal and waits synchronously for completion."""
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.AIRTABLE_ERROR]: return False
        if not self.drive_client.server_is_ready():
            self.get_logger().error("Drive action server not ready.")
            return False

        goal_msg = DriveDistance.Goal()
        goal_msg.distance = float(distance)
        goal_msg.max_translation_speed = DRIVE_SPEED

        self.get_logger().debug(f"Sending Drive goal: distance={distance:.3f}m")
        executor = SingleThreadedExecutor()
        rclpy.spin_once(self, executor=executor, timeout_sec=0.1) # Process incoming messages once
        future = self.drive_client.send_goal_async(goal_msg)

        # Wait for the goal handle
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, executor=executor, timeout_sec=0.1)
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error('Drive goal rejected or failed to get handle.')
            return False

        # Wait for the result
        result_future = goal_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
             rclpy.spin_once(self, executor=executor, timeout_sec=0.1)

        result = result_future.result()
        status = result.status if result else -1 # Get status if result exists
        if status == result.GoalStatus.STATUS_SUCCEEDED:
             self.get_logger().debug(f'Drive completed.')
             return True
        else:
             status_map = {1: 'ACCEPTED', 2: 'EXECUTING', 3: 'CANCELING', 4: 'SUCCEEDED', 5: 'ABORTED', 6: 'CANCELED', -1: 'UNKNOWN'}
             self.get_logger().warn(f'Drive action did not succeed. Status: {status_map.get(status, "INVALID")}')
             # Decide if non-success should trigger error state
             # self.state = RobotState.ERROR
             return False

    def stop_moving(self):
        """Sends a DriveDistance goal with 0 distance to stop the robot."""
        self.get_logger().info("Sending stop command.")
        return self.drive_distance(0.0)

    def rotate_angle(self, angle):
        """Sends a RotateAngle goal and waits synchronously for completion."""
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.AIRTABLE_ERROR]: return False
        if not self.rotate_client.server_is_ready():
            self.get_logger().error("Rotate action server not ready.")
            return False

        goal_msg = RotateAngle.Goal()
        goal_msg.angle = float(angle) # Radians
        goal_msg.max_rotation_speed = ROTATE_SPEED
        self.get_logger().info(f"Rotating angle: {math.degrees(angle):.1f} degrees")

        executor = SingleThreadedExecutor()
        rclpy.spin_once(self, executor=executor, timeout_sec=0.1)
        future = self.rotate_client.send_goal_async(goal_msg)

        # Wait for goal handle
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, executor=executor, timeout_sec=0.1)
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error('Rotate goal rejected or failed to get handle.')
            return False
        self.get_logger().debug('Rotate goal accepted.')

        # Wait for result
        result_future = goal_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
             rclpy.spin_once(self, executor=executor, timeout_sec=0.1)

        result = result_future.result()
        status = result.status if result else -1
        if status == result.GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Rotate completed.')
            return True
        else:
            status_map = {1: 'ACCEPTED', 2: 'EXECUTING', 3: 'CANCELING', 4: 'SUCCEEDED', 5: 'ABORTED', 6: 'CANCELED', -1: 'UNKNOWN'}
            self.get_logger().warn(f'Rotate action did not succeed. Status: {status_map.get(status, "INVALID")}')
            # self.state = RobotState.ERROR # Decide if this is fatal
            return False

    # --- Color Detection ---
    def detect_target_color(self, target_idx):
        """Checks if the color marker for the target station index is detected."""
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.AIRTABLE_ERROR]: return False

        if target_idx not in STATION_COLORS_HSV:
            self.get_logger().error(f"Invalid target index {target_idx} for color detection.")
            return False

        color_info = STATION_COLORS_HSV[target_idx]
        lower_bound = np.array(color_info["hsv_lower"])
        upper_bound = np.array(color_info["hsv_upper"])
        target_color_name = color_info["name"]

        try:
            # Capture image (Picamera2 default is often BGR-like for cv2)
            image = self.picam2.capture_array()
            # If color issues, explicitly convert: image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR) or similar

            # Convert to HSV color space
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Create a mask for the target color range
            mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

            # Optional: Noise reduction using morphological operations
            # kernel = np.ones((5, 5), np.uint8)
            # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Count non-zero pixels in the mask (pixels matching the color)
            detected_pixels = cv2.countNonZero(mask)
            self.get_logger().debug(f"Detecting {target_color_name}: Pixels={detected_pixels} (Threshold={COLOR_DETECTION_THRESHOLD})")

            # Compare detected pixels against the threshold
            return detected_pixels > COLOR_DETECTION_THRESHOLD

        except cv2.error as cv_err:
             self.get_logger().error(f"OpenCV error during color detection: {cv_err}")
             return False
        except Exception as e:
            self.get_logger().error(f"Unexpected error during color detection: {e}")
            # Potentially set state to CAMERA_ERROR if recurring
            return False

    # --- Station & Process Simulation ---
    def simulate_processing(self, duration):
        """Placeholder to simulate work time at a station."""
        self.get_logger().info(f"Processing at {STATION_COLORS_HSV[self.current_station_index]['name']} for {duration:.1f}s...")
        time.sleep(duration) # For simulation, simple sleep is fine
        self.get_logger().info("Processing complete.")

    # --- Airtable Communication ---
    def fetch_order_from_airtable(self):
        """Fetches the oldest 'To do' order from Airtable using configured credentials."""
        self.get_logger().info("Attempting to fetch order from Airtable...")

        params = {
            "maxRecords": 1,
            "filterByFormula": f"({{{AIRTABLE_STATUS_FIELD}}}='{STATUS_PENDING}')",
        }
        order_to_process = None

        try:
            response = requests.get(url=AIRTABLE_URL, headers=AIRTABLE_HEADERS, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            records = data.get("records", [])

            if records:
                record = records[0]
                record_id = record.get("id")
                fields = record.get("fields", {})

                order_id = fields.get(AIRTABLE_NAME_COLUMN)
                toppings_data = {topping: fields.get(topping, "no") for topping in AIRTABLE_TOPPINGS_COLUMNS}

                if not record_id or not order_id:
                    self.get_logger().error(f"Fetched record missing 'id' or '{AIRTABLE_NAME_COLUMN}'. Record data: {record}")
                else:
                    self.get_logger().info(f"Fetched order: ID='{order_id}', RecordID='{record_id}', Toppings={toppings_data}")
                    order_to_process = {
                        "record_id": record_id,
                        "pancake_id": order_id,
                        "toppings": [topping for topping, value in toppings_data.items() if value.lower() == "yes"]
                    }
            else:
                self.get_logger().info(f"No records found with Status='{STATUS_PENDING}'.")

        except requests.exceptions.RequestException as req_err:
            self.get_logger().error(f"Airtable request error: {req_err}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error processing Airtable response: {e}")

        return order_to_process

    def update_order_status(self, record_id, new_status):
        """Updates the status of a specific order in Airtable using its record_id."""
        if not record_id:
            self.get_logger().error("Cannot update status: record_id is missing.")
            return False

        self.get_logger().info(f"Updating Airtable record {record_id} status to '{new_status}'...")
        update_url = f"{AIRTABLE_URL}/{record_id}"
        payload = json.dumps({
            "fields": {
                AIRTABLE_STATUS_FIELD: new_status
            }
        })

        try:
            response = requests.patch(update_url, headers=AIRTABLE_HEADERS, data=payload, timeout=10)
            response.raise_for_status()
            self.get_logger().info(f"Airtable record {record_id} updated successfully to '{new_status}'.")
            return True

        except requests.exceptions.RequestException as req_err:
            self.get_logger().error(f"Airtable update request error: {req_err}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error processing Airtable update for {record_id}: {e}")

        return False

    # --- Sound Utility ---
    def play_sound(self, notes):
        """Publishes a sequence of AudioNotes to the /cmd_audio topic."""
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.AIRTABLE_ERROR]: return

        audio_msg = AudioNoteVector()
        audio_msg.append = False # Play immediately, replace any current sound

        for freq, duration_ms in notes:
            note = AudioNote()
            note.frequency = int(freq)
            # Duration expects nanoseconds
            note.max_runtime = Duration(sec=0, nanosec=int(duration_ms * 1_000_000))
            audio_msg.notes.append(note)

        self.get_logger().debug(f"Playing sound: {notes}")
        self.audio_publisher.publish(audio_msg)

    # --- Main Control Loop (State Machine Logic) ---
    def control_loop(self):
        """The core state machine logic, called periodically by the timer."""

        # Handle terminal/error states first - prevents further actions
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.CYCLE_COMPLETE, RobotState.AIRTABLE_ERROR]:
            # Log error states periodically for awareness
            if self.state == RobotState.ERROR:
                self.get_logger().error("Robot in ERROR state. Halting operations.", throttle_duration_sec=5)
            elif self.state == RobotState.CAMERA_ERROR:
                self.get_logger().error("Robot in CAMERA_ERROR state. Halting operations.", throttle_duration_sec=5)
            elif self.state == RobotState.AIRTABLE_ERROR:
                 self.get_logger().error("Robot in AIRTABLE_ERROR state. Check Airtable connection/config. Halting.", throttle_duration_sec=5)
            # Note: CYCLE_COMPLETE is a normal exit state, no error log needed.
            # Consider stopping the timer in permanent error states:
            # self.timer.cancel()
            return # Exit control loop iteration

        # Log current state for debugging
        self.get_logger().debug(f"State: {self.state.name}, Current St: {self.current_station_index}, Target St: {self.target_station_index}, Orders Done: {self.pancakes_made_count}")

        # --- State Transitions ---
        if self.state == RobotState.IDLE:
            # Ready to start or check for the next order
            self.get_logger().info("State: IDLE. Checking for new orders...")
            self.state = RobotState.FETCHING_ORDER
            # Reset relevant variables for a new potential order cycle if needed
            self.current_order = None
            self.current_station_index = 0 # Ensure we are logically at the start
            self.target_station_index = 0

        elif self.state == RobotState.FETCHING_ORDER:
            fetched_order = self.fetch_order_from_airtable()
            if fetched_order:
                self.current_order = fetched_order # Store {'record_id': ..., 'pancake_id': ..., 'toppings': ...}
                # Update status to Started immediately after fetching
                if self.update_order_status(self.current_order["record_id"], STATUS_STARTED):
                    self.current_station_index = 0 # Logically at start
                    self.target_station_index = 1 # Set target to the first processing station (index 1)
                    self.get_logger().info(f"Order {self.current_order['pancake_id']} '{STATUS_STARTED}'. Moving from {STATION_COLORS_HSV[self.current_station_index]['name']} to find {STATION_COLORS_HSV[self.target_station_index]['name']} color marker.")
                    self.state = RobotState.MOVING_TO_STATION
                else:
                    # Failed to update status - critical problem?
                    self.get_logger().error(f"Failed to update status for order {self.current_order['pancake_id']} to '{STATUS_STARTED}'. Entering Airtable Error state.")
                    self.current_order = None # Clear failed order
                    self.state = RobotState.AIRTABLE_ERROR # Treat failure to update start status as critical error
            else:
                # No orders found, or error during fetch
                self.get_logger().info("No pending orders found or error during fetch.")
                # Check if *any* pancakes were made in this run before declaring "all done"
                if self.pancakes_made_count > 0:
                     self.get_logger().info(f"Completed {self.pancakes_made_count} order(s). No more pending orders found. Cycle complete.")
                     self.play_sound([(600,100), (700,100), (800, 300)]) # Sound for finishing a batch
                else:
                     self.get_logger().info("No pending orders found on startup/check. Idling.")
                     self.play_sound([(400, 500)]) # Sound for finding no orders initially
                # Stop the timer and enter final state as no more work is available
                self.timer.cancel()
                self.state = RobotState.CYCLE_COMPLETE

        elif self.state == RobotState.MOVING_TO_STATION:
            # Check camera for the color marker of the target station
            if self.detect_target_color(self.target_station_index):
                self.get_logger().info(f"Target color marker for Station {self.target_station_index} ({STATION_COLORS_HSV[self.target_station_index]['name']}) detected!")
                self.play_sound([(500, 150)]) # Arrival beep
                self.state = RobotState.STOPPING_BEFORE_PROCESS # Go to intermediate state to stop movement
            else:
                # Color not detected, drive forward a small amount and check again next cycle
                self.get_logger().debug(f"Searching for Station {self.target_station_index} color, driving forward {DRIVE_INCREMENT}m...")
                if not self.drive_distance(DRIVE_INCREMENT):
                    self.get_logger().warn("Failed small drive increment while searching. Retrying detection/drive.")
                    # Consider adding a timeout or failure counter here to prevent infinite loops if stuck

        elif self.state == RobotState.STOPPING_BEFORE_PROCESS:
            # Ensure robot is stopped before proceeding
            if self.stop_moving():
                self.current_station_index = self.target_station_index # Officially arrived at the station
                self.get_logger().info(f"Stopped at Station {self.current_station_index}. Beginning processing.")
                self.state = RobotState.PROCESSING_AT_STATION
            else:
                 self.get_logger().error("Failed to execute stop command before processing! Retrying stop.")
                 # Stay in STOPPING_BEFORE_PROCESS state to retry stopping

        elif self.state == RobotState.PROCESSING_AT_STATION:
            station_info = STATION_COLORS_HSV[self.current_station_index]
            self.simulate_processing(station_info["process_time"]) # Blocking sleep for simulation

            # Decide next step after processing
            if self.current_station_index < (NUM_STATIONS - 1): # If not the last processing station (index 4)
                self.target_station_index = self.current_station_index + 1
                self.get_logger().info(f"Processing done at {station_info['name']}. Moving to find {STATION_COLORS_HSV[self.target_station_index]['name']} color marker.")
                self.state = RobotState.MOVING_TO_STATION # Move to next station
            else: # Finished the last processing station (Station 4)
                 self.get_logger().info(f"Processing done at last station {station_info['name']}.")
                 # Update status to Ready in Airtable
                 if self.update_order_status(self.current_order["record_id"], STATUS_READY):
                     self.get_logger().info(f"Order {self.current_order['pancake_id']} marked as '{STATUS_READY}'. Returning to start.")
                     self.target_station_index = 0 # Target is now the start station marker (index 0)
                     self.state = RobotState.RETURNING_TO_START
                 else:
                     # Failed to update final status - critical error
                     self.get_logger().error(f"Failed to update status for order {self.current_order['pancake_id']} to '{STATUS_READY}'. Entering Airtable Error state.")
                     self.state = RobotState.AIRTABLE_ERROR

        elif self.state == RobotState.RETURNING_TO_START:
            # Check camera for the color marker of the start station (index 0)
            if self.detect_target_color(self.target_station_index): # target_station_index is 0 here
                self.get_logger().info(f"Return color marker for Station {self.target_station_index} ({STATION_COLORS_HSV[self.target_station_index]['name']}) detected!")
                self.play_sound([(800, 100), (700, 100), (600, 200)]) # Completion sound
                self.state = RobotState.STOPPING_BEFORE_IDLE # Go to intermediate state to stop movement
            else:
                # Color not detected, drive forward a small amount
                self.get_logger().debug(f"Searching for return color marker, driving forward {DRIVE_INCREMENT}m...")
                if not self.drive_distance(DRIVE_INCREMENT):
                    self.get_logger().warn("Failed small drive increment while returning. Retrying detection/drive.")
                    # Consider adding timeout/failure counter

        elif self.state == RobotState.STOPPING_BEFORE_IDLE:
             # Ensure robot is stopped before returning to IDLE
             if self.stop_moving():
                self.current_station_index = 0 # Officially back at start station
                self.pancakes_made_count += 1
                self.get_logger().info(f"Order {self.current_order['pancake_id']} cycle complete. Stopped at start station. Total orders completed: {self.pancakes_made_count}.")
                self.current_order = None # Clear completed order
                self.state = RobotState.IDLE # Return to IDLE to check for more orders
             else:
                 self.get_logger().error("Failed to execute stop command before idling! Retrying stop.")
                 # Stay in STOPPING_BEFORE_IDLE state to retry stopping

    # --- Cleanup Methods ---
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

    def shutdown_robot(self):
        """Commands the robot to stop moving."""
        self.get_logger().info("Attempting to stop robot movement...")
        # Check if node is still valid and client is ready before sending stop
        if rclpy.ok() and self.drive_client and self.drive_client.server_is_ready():
             self.stop_moving()
        else:
            self.get_logger().warn("Could not send final stop command (node shutdown or server unavailable).")


# --- Main Execution Function ---
def main(args=None):
    """Initializes ROS, creates the node, spins it, and handles cleanup."""
    rclpy.init(args=args)
    pancake_robot_node = None # Initialize to None for robust cleanup
    try:
        pancake_robot_node = PancakeRobotNode()

        # Check for fatal initialization errors before spinning
        if pancake_robot_node.state not in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.AIRTABLE_ERROR]:
            rclpy.spin(pancake_robot_node) # Keep the node alive and executing callbacks/timer
        else:
            pancake_robot_node.get_logger().error("Node initialization failed. Shutting down without spinning.")

    except KeyboardInterrupt:
        if pancake_robot_node:
            pancake_robot_node.get_logger().info("Keyboard interrupt detected, shutting down.")
    except Exception as e:
         # Log any unexpected errors during spin
         if pancake_robot_node:
             pancake_robot_node.get_logger().fatal(f"An unexpected error occurred during spin: {e}")
             import traceback
             traceback.print_exc() # Print detailed traceback
         else:
             print(f"An unexpected error occurred before node was fully initialized: {e}")
             import traceback
             traceback.print_exc()
    finally:
        # Cleanup sequence
        if pancake_robot_node:
            pancake_robot_node.get_logger().info("Initiating shutdown sequence...")
            # 1. Stop the control loop timer explicitly if it's running
            if hasattr(pancake_robot_node, 'timer') and pancake_robot_node.timer and not pancake_robot_node.timer.is_canceled():
                pancake_robot_node.timer.cancel()
            # 2. Attempt to stop the robot's movement
            pancake_robot_node.shutdown_robot()
            # 3. Stop the camera
            pancake_robot_node.shutdown_camera()
            # 4. Destroy the node
            pancake_robot_node.destroy_node()
            pancake_robot_node.get_logger().info("Pancake Robot Node destroyed.")

        # 5. Shutdown ROS client library
        if rclpy.ok():
             rclpy.shutdown()
        print("ROS2 shutdown complete.")


if __name__ == '__main__':
    main()