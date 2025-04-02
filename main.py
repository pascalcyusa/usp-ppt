#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor # To wait for actions

import time
from enum import Enum, auto
import math

# iRobot Create 3 specific messages
from irobot_create_msgs.action import DriveDistance, RotateAngle
from irobot_create_msgs.msg import InterfaceButtons, IrIntensityVector # Example sensor msgs
from builtin_interfaces.msg import Duration
from irobot_create_msgs.msg import AudioNoteVector, AudioNote

# --- Configuration Constants ---

# Station Definitions (Customize these)
STATIONS = {
    0: {"name": "Order Station", "process_time": 2.0, "distance_from_prev": 0.5}, # Start/End point
    1: {"name": "Batter/Cook", "process_time": 5.0, "distance_from_prev": 1.0},
    2: {"name": "Topping 1", "process_time": 3.0, "distance_from_prev": 0.8},
    3: {"name": "Topping 2", "process_time": 3.0, "distance_from_prev": 0.8},
    4: {"name": "Topping 3", "process_time": 3.0, "distance_from_prev": 0.8},
    # Add distance from Station 4 back to Station 0 (Order Station)
    "RETURN": {"distance_from_prev": 1.5} 
}
NUM_STATIONS = 5 # 0 to 4

# Robot Control Parameters
DRIVE_SPEED = 0.15  # m/s
ROTATE_SPEED = 0.8  # rad/s
PANCAKES_TO_MAKE = 3 # How many times to run the full cycle

# --- State Machine Definition ---
class RobotState(Enum):
    IDLE = auto()
    FETCHING_ORDER = auto()
    MOVING_TO_STATION = auto()
    PROCESSING_AT_STATION = auto()
    RETURNING_TO_START = auto()
    CYCLE_COMPLETE = auto()
    ERROR = auto()

# --- Main Robot Control Class ---
class PancakeRobotNode(Node):
    """
    Manages the iCreate3 robot for the pancake making process simulation.
    Follows a state machine to move between stations and simulate processing.
    """
    def __init__(self):
        super().__init__('pancake_robot_node')
        self.get_logger().info("Pancake Robot Node Initializing...")

        # Robot State
        self.state = RobotState.IDLE
        self.current_station_index = 0 # Start at the 'Order Station'
        self.pancakes_made = 0

        # Action Clients
        self.drive_client = ActionClient(self, DriveDistance, '/drive_distance')
        self.rotate_client = ActionClient(self, RotateAngle, '/rotate_angle')
        self.audio_publisher = self.create_publisher(AudioNoteVector, '/cmd_audio', 10)

        # Sensor Subscriptions (Example - adapt based on actual sensors used)
        # self.ir_subscription = self.create_subscription(
        #     IrIntensityVector,
        #     '/ir_intensity',
        #     self.ir_callback,
        #     10)
        # self.button_subscription = self.create_subscription(
        #     InterfaceButtons,
        #     '/interface_buttons',
        #     self.button_callback,
        #     10)
        
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
        self.timer_period = 0.5  # seconds
        self.timer = self.create_timer(self.timer_period, self.control_loop)

        self.get_logger().info("Pancake Robot Node Initialized.")
        self.play_sound([(440, 200), (550, 300)]) # Play startup sound

    # --- Movement Actions ---
    def drive_distance(self, distance):
        """Sends a DriveDistance goal and waits for completion."""
        if self.state == RobotState.ERROR: return False
        
        goal_msg = DriveDistance.Goal()
        goal_msg.distance = float(distance)
        goal_msg.max_translation_speed = DRIVE_SPEED

        self.get_logger().info(f"Driving distance: {distance:.2f}m")
        
        # Use an executor to make the async call behave synchronously for simplicity here
        executor = SingleThreadedExecutor()
        rclpy.spin_once(self, executor=executor, timeout_sec=0.1) # Process incoming messages once

        future = self.drive_client.send_goal_async(goal_msg)
        
        # Wait for the goal handle
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, executor=executor, timeout_sec=0.1)
            
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Drive goal rejected :(')
            self.state = RobotState.ERROR
            return False

        self.get_logger().info('Drive goal accepted :)')

        # Wait for the result
        result_future = goal_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
             rclpy.spin_once(self, executor=executor, timeout_sec=0.1)

        result = result_future.result().result
        # TODO: Check result status (SUCCEEDED, ABORTED, CANCELED)
        self.get_logger().info(f'Drive completed. Pose: {result.pose}')
        return True # Assume success for now

    def rotate_angle(self, angle):
        """Sends a RotateAngle goal and waits for completion."""
        if self.state == RobotState.ERROR: return False
        
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
        if not goal_handle.accepted:
            self.get_logger().error('Rotate goal rejected :(')
            self.state = RobotState.ERROR
            return False
            
        self.get_logger().info('Rotate goal accepted :)')

        result_future = goal_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
             rclpy.spin_once(self, executor=executor, timeout_sec=0.1)
             
        result = result_future.result().result
        self.get_logger().info(f'Rotate completed. Pose: {result.pose}')
        return True # Assume success for now

    # --- Station & Process Simulation ---
    def simulate_processing(self, duration):
        """Placeholder to simulate work at a station."""
        self.get_logger().info(f"Processing at station {self.current_station_index} for {duration:.1f}s...")
        time.sleep(duration)
        self.get_logger().info("Processing complete.")

    # --- External Order System (Placeholder) ---
    def fetch_order_from_airtable(self):
        """Placeholder for fetching order details."""
        self.get_logger().info("Fetching order from Airtable... (Simulated)")
        # In a real implementation:
        # - Connect to Airtable API
        # - Query for a new order
        # - Parse toppings, quantity etc.
        time.sleep(1.0) # Simulate network delay
        return {"pancake_id": f"P{self.pancakes_made + 1}", "toppings": ["Syrup", "Butter"]}

    def update_order_status(self, order_id, status):
        """Placeholder for updating order status in Airtable."""
        self.get_logger().info(f"Updating Airtable: Order {order_id} status to {status} (Simulated)")
        # In a real implementation:
        # - Connect to Airtable API
        # - Find the order record
        # - Update its status field
        time.sleep(0.5)

    # --- Sound Utility ---
    def play_sound(self, notes):
        """Publishes a sequence of notes to the audio topic."""
        if self.state == RobotState.ERROR: return

        audio_msg = AudioNoteVector()
        audio_msg.append = False # Play immediately, replace previous sounds
        
        for freq, duration_ms in notes:
            note = AudioNote()
            note.frequency = int(freq)
            # Duration expects nanoseconds
            note.max_runtime = Duration(sec=0, nanosec=int(duration_ms * 1_000_000)) 
            audio_msg.notes.append(note)
        
        self.get_logger().debug(f"Playing sound: {notes}")
        self.audio_publisher.publish(audio_msg)

    # --- Sensor Callbacks (Examples) ---
    # def ir_callback(self, msg):
    #     # Process IR sensor data - maybe detect markers at stations
    #     # readings = [reading.value for reading in msg.readings]
    #     # self.get_logger().debug(f"IR Readings: {readings}")
    #     pass

    # def button_callback(self, msg):
    #     # React to button presses (e.g., start/stop)
    #     if msg.button_1.is_pressed:
    #         self.get_logger().info("Button 1 pressed!")
    #         # Example: Maybe reset state or start process
    #         if self.state == RobotState.IDLE or self.state == RobotState.CYCLE_COMPLETE:
    #              self.state = RobotState.FETCHING_ORDER # Start a new cycle
    #     pass

    # --- Main Control Loop ---
    def control_loop(self):
        """The core state machine logic, called periodically by the timer."""
        
        if self.state == RobotState.ERROR:
            # Maybe try to recover or just stay put
            # self.get_logger().error("Robot in ERROR state. Halting.")
             # self.timer.cancel() # Stop the loop
             return

        self.get_logger().debug(f"Current State: {self.state.name}, Station: {self.current_station_index}, Pancakes Made: {self.pancakes_made}")

        # --- State Transitions ---
        if self.state == RobotState.IDLE:
            if self.pancakes_made < PANCAKES_TO_MAKE:
                self.get_logger().info(f"Starting pancake cycle {self.pancakes_made + 1}/{PANCAKES_TO_MAKE}")
                # Transition to fetching the order (even if simulated)
                self.state = RobotState.FETCHING_ORDER 
            else:
                self.get_logger().info("All pancakes made!")
                self.play_sound([(600,100), (700,100), (800, 300)])
                self.timer.cancel() # Stop the loop
                self.state = RobotState.CYCLE_COMPLETE # Final state
        
        elif self.state == RobotState.FETCHING_ORDER:
            order_details = self.fetch_order_from_airtable()
            if order_details:
                self.current_order = order_details # Store for later use if needed
                # Assume order station (0) is the starting point, move to station 1
                self.update_order_status(self.current_order["pancake_id"], "Started")
                self.current_station_index = 0 # Explicitly start at 0
                self.target_station_index = 1 
                self.state = RobotState.MOVING_TO_STATION
            else:
                self.get_logger().warn("Failed to fetch order, retrying next cycle.")
                time.sleep(5) # Wait before retrying

        elif self.state == RobotState.MOVING_TO_STATION:
            distance_to_drive = STATIONS[self.target_station_index]["distance_from_prev"]
            self.get_logger().info(f"Moving from station {self.current_station_index} to {self.target_station_index}")
            
            # --- Optional: Add Rotation Here if Needed ---
            # if self.target_station_index == X: # e.g., turning a corner
            #    if not self.rotate_angle(math.pi / 2): # Turn 90 degrees right
            #        self.state = RobotState.ERROR
            #        return
            # ---------------------------------------------

            if self.drive_distance(distance_to_drive):
                self.current_station_index = self.target_station_index
                self.state = RobotState.PROCESSING_AT_STATION
                self.play_sound([(500, 150)]) # Arrival beep
            else:
                self.get_logger().error("Failed to drive to station.")
                self.state = RobotState.ERROR # Enter error state

        elif self.state == RobotState.PROCESSING_AT_STATION:
            station_info = STATIONS[self.current_station_index]
            self.simulate_processing(station_info["process_time"])
            
            # Decide next state
            if self.current_station_index < (NUM_STATIONS -1) : # If not the last topping station
                self.target_station_index = self.current_station_index + 1
                self.state = RobotState.MOVING_TO_STATION
            else: # Finished the last station (Station 4)
                 self.update_order_status(self.current_order["pancake_id"], "Ready")
                 self.state = RobotState.RETURNING_TO_START

        elif self.state == RobotState.RETURNING_TO_START:
            distance_to_drive = STATIONS["RETURN"]["distance_from_prev"]
            self.get_logger().info(f"Returning from station {self.current_station_index} to start (Station 0)")
            
            # --- Optional: Add Rotation Here if Needed ---
            # if self.target_station_index == X: # e.g., turning a corner
            #    if not self.rotate_angle(math.pi / 2): # Turn 90 degrees right
            #        self.state = RobotState.ERROR
            #        return
            # ---------------------------------------------

            if self.drive_distance(distance_to_drive):
                self.current_station_index = 0 # Back at the start
                self.pancakes_made += 1
                self.get_logger().info(f"Pancake {self.pancakes_made} cycle complete. Returning to IDLE.")
                self.play_sound([(800, 100), (700, 100), (600, 200)]) # Completion sound
                self.state = RobotState.IDLE # Ready for next pancake or finish
            else:
                self.get_logger().error("Failed to return to start.")
                self.state = RobotState.ERROR

        elif self.state == RobotState.CYCLE_COMPLETE:
            # Do nothing, just wait for shutdown
             pass


def main(args=None):
    rclpy.init(args=args)
    pancake_robot_node = PancakeRobotNode()
    
    # Use spin to keep the node alive and allow the timer callback to run
    # Handle potential errors during initialization
    if pancake_robot_node.state != RobotState.ERROR:
        try:
            rclpy.spin(pancake_robot_node)
        except KeyboardInterrupt:
            pancake_robot_node.get_logger().info("Keyboard interrupt detected, shutting down.")
        except Exception as e:
             pancake_robot_node.get_logger().error(f"An unexpected error occurred: {e}")
             import traceback
             traceback.print_exc()
        finally:
            # Cleanup
            pancake_robot_node.get_logger().info("Destroying node...")
            pancake_robot_node.destroy_node()
            rclpy.shutdown()
            pancake_robot_node.get_logger().info("Shutdown complete.")
    else:
        pancake_robot_node.get_logger().error("Initialization failed. Shutting down.")
        pancake_robot_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()