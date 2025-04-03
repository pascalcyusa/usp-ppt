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

# iRobot Create 3 specific messages
from irobot_create_msgs.action import DriveDistance, RotateAngle
from irobot_create_msgs.msg import InterfaceButtons, IrIntensityVector # Example sensor msgs
from builtin_interfaces.msg import Duration
from irobot_create_msgs.msg import AudioNoteVector, AudioNote

# --- Configuration Constants ---

# --- Camera Configuration ---
CAMERA_RESOLUTION = (640, 480) # Width, Height
# --- Color Detection Configuration ---
# Define HSV Lower and Upper bounds for each station's target color
# Find these using a color picker tool (e.g., https://alloyui.com/examples/color-picker/hsv.html)
# Format: (Hue, Saturation, Value)
STATION_COLORS_HSV = {
    # Note: Color for STATION 0 is the color TO DETECT when RETURNING
    0: {"name": "Order Station", "process_time": 2.0,
        "hsv_lower": (100, 100, 50), "hsv_upper": (130, 255, 255)}, # Example: Blue (for return)
    1: {"name": "Batter/Cook", "process_time": 5.0,
        "hsv_lower": (0, 100, 100), "hsv_upper": (10, 255, 255)},   # Example: Red
    2: {"name": "Topping 1", "process_time": 3.0,
        "hsv_lower": (20, 100, 100), "hsv_upper": (40, 255, 255)},   # Example: Yellow
    3: {"name": "Topping 2", "process_time": 3.0,
        "hsv_lower": (50, 100, 50), "hsv_upper": (80, 255, 255)},   # Example: Green
    4: {"name": "Topping 3", "process_time": 3.0,
        "hsv_lower": (140, 100, 50), "hsv_upper": (170, 255, 255)},  # Example: Magenta/Pink
}
NUM_STATIONS = len(STATION_COLORS_HSV) # Now 5 (0 to 4)

# Threshold for color detection (minimum number of white pixels in the mask)
COLOR_DETECTION_THRESHOLD = 500 # Adjust based on testing
# Amount to drive forward when searching for color (meters)
DRIVE_INCREMENT = 0.03 # Drive small steps when searching

# Robot Control Parameters
DRIVE_SPEED = 0.1  # m/s (Maybe slower when searching)
ROTATE_SPEED = 0.8  # rad/s
PANCAKES_TO_MAKE = 3 # How many times to run the full cycle

# --- State Machine Definition ---
class RobotState(Enum):
    IDLE = auto()
    FETCHING_ORDER = auto()
    # MOVING_TO_STATION will now include searching
    MOVING_TO_STATION = auto()
    PROCESSING_AT_STATION = auto()
    # RETURNING_TO_START will now include searching
    RETURNING_TO_START = auto()
    STOPPING_BEFORE_PROCESS = auto() # Intermediate state to ensure stop
    STOPPING_BEFORE_IDLE = auto() # Intermediate state to ensure stop
    CYCLE_COMPLETE = auto()
    ERROR = auto()
    CAMERA_ERROR = auto()

# --- Main Robot Control Class ---
class PancakeRobotNode(Node):
    """
    Manages the iCreate3 robot for the pancake making process simulation.
    Follows a state machine, using camera color detection to navigate between stations.
    """
    def __init__(self):
        super().__init__('pancake_robot_node')
        self.get_logger().info("Pancake Robot Node Initializing...")

        # Robot State
        self.state = RobotState.IDLE
        self.current_station_index = 0 # Start at the 'Order Station'
        self.target_station_index = 0 # Station we are currently moving towards
        self.pancakes_made = 0
        self.last_drive_goal_future = None # To track drive goals if needed

        # Action Clients
        self.drive_client = ActionClient(self, DriveDistance, '/drive_distance')
        self.rotate_client = ActionClient(self, RotateAngle, '/rotate_angle')
        self.audio_publisher = self.create_publisher(AudioNoteVector, '/cmd_audio', 10)

        # Sensor Subscriptions (Optional)
        # ... (keep if needed)

        # --- Camera Setup ---
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(main={"size": CAMERA_RESOLUTION})
            self.picam2.configure(config)
            # Enable continuous autofocus
            self.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous, "LensPosition": 0.0})
            self.picam2.start()
            time.sleep(2) # Allow camera to initialize and focus
            self.get_logger().info("Pi Camera initialized successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize Pi Camera: {e}")
            self.state = RobotState.CAMERA_ERROR
            # No point continuing if camera fails
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
        self.timer_period = 0.2  # seconds (Check color more frequently)
        self.timer = self.create_timer(self.timer_period, self.control_loop)

        self.get_logger().info("Pancake Robot Node Initialized.")
        self.play_sound([(440, 200), (550, 300)]) # Play startup sound

    # --- Movement Actions ---
    def drive_distance(self, distance):
        """Sends a DriveDistance goal and waits for completion."""
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR]: return False

        goal_msg = DriveDistance.Goal()
        goal_msg.distance = float(distance)
        # Use a slightly lower speed when inching forward?
        # goal_msg.max_translation_speed = DRIVE_SPEED if distance > 0 else 0.0
        goal_msg.max_translation_speed = DRIVE_SPEED

        self.get_logger().debug(f"Sending Drive goal: distance={distance:.3f}m")

        # Make the call synchronous using an executor
        executor = SingleThreadedExecutor()
        rclpy.spin_once(self, executor=executor, timeout_sec=0.1)

        future = self.drive_client.send_goal_async(goal_msg)

        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, executor=executor, timeout_sec=0.1)

        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error('Drive goal rejected or failed to get handle.')
            # Don't necessarily go to ERROR state, might be recoverable
            # Consider retrying or specific error handling
            return False

        # Wait for the result (this makes it blocking)
        result_future = goal_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
             rclpy.spin_once(self, executor=executor, timeout_sec=0.1)

        result = result_future.result()
        if result and result.status == result.GoalStatus.STATUS_SUCCEEDED:
             self.get_logger().debug(f'Drive completed. Final Pose: {result.result.pose}')
             return True
        else:
             status_str = result.status if result else 'Unknown'
             self.get_logger().warn(f'Drive action did not succeed. Status: {status_str}')
             # Decide if this constitutes an error state
             # self.state = RobotState.ERROR
             return False

    def stop_moving(self):
        """Sends a DriveDistance goal with 0 distance to stop the robot."""
        self.get_logger().info("Sending stop command.")
        # Call drive_distance with 0.0. The internal logging/waiting handles it.
        # The action server should interpret 0 distance as a stop command.
        return self.drive_distance(0.0)

    def rotate_angle(self, angle):
        """Sends a RotateAngle goal and waits for completion."""
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR]: return False

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
            # self.state = RobotState.ERROR # Decide if this is fatal
            return False

        self.get_logger().info('Rotate goal accepted.')

        result_future = goal_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
             rclpy.spin_once(self, executor=executor, timeout_sec=0.1)

        result = result_future.result()
        if result and result.status == result.GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Rotate completed. Final Pose: {result.result.pose}')
            return True
        else:
            status_str = result.status if result else 'Unknown'
            self.get_logger().warn(f'Rotate action did not succeed. Status: {status_str}')
            # self.state = RobotState.ERROR # Decide if this is fatal
            return False

    # --- Color Detection ---
    def detect_target_color(self, target_idx):
        """Checks if the color for the target station index is detected by the camera."""
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR]: return False

        if target_idx not in STATION_COLORS_HSV:
            self.get_logger().error(f"Invalid target index {target_idx} for color detection.")
            return False

        color_info = STATION_COLORS_HSV[target_idx]
        lower_bound = np.array(color_info["hsv_lower"])
        upper_bound = np.array(color_info["hsv_upper"])
        target_color_name = color_info["name"]

        try:
            # Capture image (already BGR format from Picamera2 typically)
            image = self.picam2.capture_array()
            # Ensure it's BGR (cv2 standard) - Picamera2 might give BGRA, RGB, etc.
            # Let's assume capture_array gives something cvtColor can handle.
            # If issues, check format: image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
            # Convert to HSV
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Create mask
            mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

            # Optional: Apply morphological operations to reduce noise
            # kernel = np.ones((5,5),np.uint8)
            # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Count white pixels (detected color)
            detected_pixels = cv2.countNonZero(mask)
            self.get_logger().debug(f"Detecting {target_color_name}: Pixels={detected_pixels} (Threshold={COLOR_DETECTION_THRESHOLD})")

            # --- Debugging: Show the mask ---
            # cv2.imshow("Mask", mask)
            # cv2.waitKey(1) # Important for imshow to update
            # --- End Debugging ---

            return detected_pixels > COLOR_DETECTION_THRESHOLD

        except Exception as e:
            self.get_logger().error(f"Error during color detection: {e}")
            # Potentially set state to CAMERA_ERROR or just log and return False
            # self.state = RobotState.CAMERA_ERROR
            return False

    # --- Station & Process Simulation ---
    def simulate_processing(self, duration):
        """Placeholder to simulate work at a station."""
        self.get_logger().info(f"Processing at {STATION_COLORS_HSV[self.current_station_index]['name']} for {duration:.1f}s...")
        time.sleep(duration) # Use ROS time maybe? rclpy.time.Duration? For simulation, time.sleep is fine.
        self.get_logger().info("Processing complete.")

    # --- External Order System (Placeholder) ---
    def fetch_order_from_airtable(self):
        """Placeholder for fetching order details."""
        self.get_logger().info("Fetching order from Airtable... (Simulated)")
        time.sleep(1.0) # Simulate network delay
        return {"pancake_id": f"P{self.pancakes_made + 1}", "toppings": ["Syrup", "Butter"]}

    def update_order_status(self, order_id, status):
        """Placeholder for updating order status in Airtable."""
        self.get_logger().info(f"Updating Airtable: Order {order_id} status to {status} (Simulated)")
        time.sleep(0.5)

    # --- Sound Utility ---
    def play_sound(self, notes):
        """Publishes a sequence of notes to the audio topic."""
        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR]: return

        audio_msg = AudioNoteVector()
        audio_msg.append = False # Play immediately

        for freq, duration_ms in notes:
            note = AudioNote()
            note.frequency = int(freq)
            note.max_runtime = Duration(sec=0, nanosec=int(duration_ms * 1_000_000))
            audio_msg.notes.append(note)

        self.get_logger().debug(f"Playing sound: {notes}")
        self.audio_publisher.publish(audio_msg)

    # --- Sensor Callbacks (Examples - Keep if used) ---
    # ...

    # --- Main Control Loop ---
    def control_loop(self):
        """The core state machine logic, called periodically by the timer."""

        if self.state in [RobotState.ERROR, RobotState.CAMERA_ERROR, RobotState.CYCLE_COMPLETE]:
            # If in a final or error state, do nothing in the loop
            if self.state == RobotState.ERROR:
                self.get_logger().error("Robot in ERROR state. Halting loop.", throttle_duration_sec=5)
            elif self.state == RobotState.CAMERA_ERROR:
                self.get_logger().error("Robot in CAMERA_ERROR state. Halting loop.", throttle_duration_sec=5)
            # self.timer.cancel() # Optionally stop the timer
            return

        self.get_logger().debug(f"State: {self.state.name}, Current Station: {self.current_station_index}, Target: {self.target_station_index}, Pancakes: {self.pancakes_made}")

        # --- State Transitions ---
        if self.state == RobotState.IDLE:
            if self.pancakes_made < PANCAKES_TO_MAKE:
                self.get_logger().info(f"Starting pancake cycle {self.pancakes_made + 1}/{PANCAKES_TO_MAKE}")
                self.state = RobotState.FETCHING_ORDER
            else:
                self.get_logger().info(f"All {PANCAKES_TO_MAKE} pancakes made!")
                self.play_sound([(600,100), (700,100), (800, 300)])
                self.timer.cancel() # Stop the loop
                self.state = RobotState.CYCLE_COMPLETE # Final state

        elif self.state == RobotState.FETCHING_ORDER:
            order_details = self.fetch_order_from_airtable()
            if order_details:
                self.current_order = order_details
                self.update_order_status(self.current_order["pancake_id"], "Started")
                self.current_station_index = 0 # Explicitly start at 0
                self.target_station_index = 1 # Set target to the first station
                self.get_logger().info(f"Order received. Moving from {STATION_COLORS_HSV[self.current_station_index]['name']} to find {STATION_COLORS_HSV[self.target_station_index]['name']} color.")
                self.state = RobotState.MOVING_TO_STATION
            else:
                self.get_logger().warn("Failed to fetch order, retrying next cycle.")
                # Consider adding a delay or a retry limit
                time.sleep(5) # Wait before trying IDLE state again

        elif self.state == RobotState.MOVING_TO_STATION:
            # Check if we see the color for the target station
            if self.detect_target_color(self.target_station_index):
                self.get_logger().info(f"Target color for Station {self.target_station_index} detected!")
                self.play_sound([(500, 150)]) # Arrival beep
                self.state = RobotState.STOPPING_BEFORE_PROCESS # Go to intermediate stop state
            else:
                # Color not detected, drive forward a small amount
                self.get_logger().debug(f"Color not detected, driving forward {DRIVE_INCREMENT}m...")
                # This call blocks until the small drive is complete
                if not self.drive_distance(DRIVE_INCREMENT):
                    self.get_logger().warn("Failed to drive increment, retrying detection/drive.")
                    # Decide how to handle failure - maybe retry, maybe error out
                    # For now, it will just retry on the next loop iteration
                    pass # Stay in MOVING_TO_STATION state

        elif self.state == RobotState.STOPPING_BEFORE_PROCESS:
            if self.stop_moving():
                self.current_station_index = self.target_station_index # Update current station ONLY after stopping
                self.state = RobotState.PROCESSING_AT_STATION
            else:
                 self.get_logger().error("Failed to stop before processing! Retrying stop.")
                 # Stay in STOPPING_BEFORE_PROCESS to retry stopping

        elif self.state == RobotState.PROCESSING_AT_STATION:
            station_info = STATION_COLORS_HSV[self.current_station_index]
            self.simulate_processing(station_info["process_time"])

            # Decide next state
            if self.current_station_index < (NUM_STATIONS - 1): # If not the last topping station (index 4)
                self.target_station_index = self.current_station_index + 1
                self.get_logger().info(f"Processing done. Moving from {station_info['name']} to find {STATION_COLORS_HSV[self.target_station_index]['name']} color.")
                self.state = RobotState.MOVING_TO_STATION
            else: # Finished the last station (Station 4)
                 self.update_order_status(self.current_order["pancake_id"], "Ready")
                 self.get_logger().info(f"Processing done at last station {station_info['name']}. Returning to start.")
                 self.target_station_index = 0 # Target is now the start station (index 0)
                 self.state = RobotState.RETURNING_TO_START

        elif self.state == RobotState.RETURNING_TO_START:
            # Check if we see the color for the start station (index 0)
            if self.detect_target_color(self.target_station_index): # target_station_index is 0 here
                self.get_logger().info(f"Return color for Station {self.target_station_index} detected!")
                self.play_sound([(800, 100), (700, 100), (600, 200)]) # Completion sound
                self.state = RobotState.STOPPING_BEFORE_IDLE # Go to intermediate stop state
            else:
                # Color not detected, drive forward a small amount
                self.get_logger().debug(f"Return color not detected, driving forward {DRIVE_INCREMENT}m...")
                if not self.drive_distance(DRIVE_INCREMENT):
                    self.get_logger().warn("Failed to drive increment while returning, retrying detection/drive.")
                    # Stay in RETURNING_TO_START state

        elif self.state == RobotState.STOPPING_BEFORE_IDLE:
             if self.stop_moving():
                self.current_station_index = 0 # Officially back at start
                self.pancakes_made += 1
                self.get_logger().info(f"Pancake {self.pancakes_made} cycle complete. Stopped at start. Returning to IDLE.")
                self.state = RobotState.IDLE # Ready for next pancake or finish
             else:
                 self.get_logger().error("Failed to stop before idle! Retrying stop.")
                 # Stay in STOPPING_BEFORE_IDLE to retry stopping


    def shutdown_camera(self):
        """Safely stops the camera."""
        if hasattr(self, 'picam2') and self.picam2:
            try:
                self.get_logger().info("Stopping Pi Camera...")
                self.picam2.stop()
                self.get_logger().info("Pi Camera stopped.")
            except Exception as e:
                self.get_logger().error(f"Error stopping camera: {e}")

def main(args=None):
    rclpy.init(args=args)
    pancake_robot_node = None # Initialize to None
    try:
        pancake_robot_node = PancakeRobotNode()

        # Check for initialization errors (like camera failure)
        if pancake_robot_node.state not in [RobotState.ERROR, RobotState.CAMERA_ERROR]:
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
        # Cleanup
        if pancake_robot_node:
            pancake_robot_node.get_logger().info("Initiating cleanup...")
            # Ensure camera is stopped before destroying node
            pancake_robot_node.shutdown_camera()
            # Stop any potential movement if interrupted abruptly (optional, stop_moving is blocking)
            # pancake_robot_node.stop_moving()
            pancake_robot_node.destroy_node()
            pancake_robot_node.get_logger().info("Node destroyed.")
        if rclpy.ok():
             rclpy.shutdown()
        print("ROS Cleanup complete.")


if __name__ == '__main__':
    main()