#!/usr/bin/env python3

import rclpy
from rclpy.node import Node # Still need a Node for clients/logging
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor

import time
import sys # For exiting on error

# iRobot Create 3 specific messages
from irobot_create_msgs.action import DriveDistance
from builtin_interfaces.msg import Duration # Although not used for sound, DriveDistance might use it internally

# --- Configuration Constants ---

# Station Definitions (Customize distances!)
# Format: {station_index: (name, process_time_sec, distance_from_previous_m)}
STATIONS = {
    # 0: ("Start/Order", 2.0, 0.0), # Start, handled separately
    1: ("Batter/Cook", 5.0, 1.0),   # Drive 1.0m to station 1, wait 5s
    2: ("Topping 1",   3.0, 0.8),   # Drive 0.8m to station 2, wait 3s
    3: ("Topping 2",   3.0, 0.8),   # Drive 0.8m to station 3, wait 3s
    4: ("Topping 3",   3.0, 0.8),   # Drive 0.8m to station 4, wait 3s
}
NUM_STATIONS = len(STATIONS) # Number of processing stations (1 to 4)
INITIAL_WAIT = 2.0 # Wait time at the very start (simulating order taking)
RETURN_DISTANCE = 1.5 # Distance from last station (4) back to start (0) - Customize!

# Robot Control Parameters
DRIVE_SPEED = 0.15  # m/s
PANCAKES_TO_MAKE = 2 # How many times to run the full cycle

# --- Helper Function for Driving and Waiting ---

def drive_and_wait(node, drive_client, distance, wait_time):
    """
    Sends a DriveDistance goal, waits for completion, then waits some more.

    Args:
        node: The rclpy Node instance.
        drive_client: The ActionClient for DriveDistance.
        distance: The distance to drive in meters.
        wait_time: The time to wait *after* driving completes, in seconds.

    Returns:
        True if the drive was successful (or distance was 0), False otherwise.
    """
    logger = node.get_logger()

    if distance <= 0:
        if wait_time > 0:
            logger.info(f"Waiting for {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        return True # No driving needed, considered success

    # --- Send Goal and Wait Logic ---
    goal_msg = DriveDistance.Goal()
    goal_msg.distance = float(distance)
    goal_msg.max_translation_speed = DRIVE_SPEED

    logger.info(f"Driving {distance:.2f}m...")

    # We need an executor to spin the node and wait for the action result
    executor = SingleThreadedExecutor()
    # Add the node to the executor *temporarily* for this action call
    executor.add_node(node)

    try:
        # Send the goal
        future = drive_client.send_goal_async(goal_msg)

        # Spin until the goal handle is available
        while rclpy.ok() and not future.done():
            executor.spin_once(timeout_sec=0.1)
            if not rclpy.ok():
                 logger.warn("ROS shutdown requested during goal sending.")
                 return False

        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            logger.error('Drive goal rejected.')
            return False
        logger.info('Drive goal accepted.')

        # Spin until the result is available
        result_future = goal_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
            executor.spin_once(timeout_sec=0.1)
            if not rclpy.ok():
                 logger.warn("ROS shutdown requested during goal execution.")
                 # Optionally try to cancel goal: goal_handle.cancel_goal_async()
                 return False

        # Check the result status
        result = result_future.result()
        if result and result.status == result.status.SUCCEEDED:
            logger.info("Drive completed successfully.")
            # --- Wait after driving ---
            if wait_time > 0:
                logger.info(f"Waiting at station for {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            return True
        else:
            status_text = f"{result.status if result else 'Unknown'}"
            logger.error(f"Drive failed or was cancelled. Status: {status_text}")
            return False
    finally:
        # Important: Remove the node from the executor after use
        # if it's going to be used elsewhere or spun differently later.
        # In this simple script, it might not be strictly necessary,
        # but it's good practice if the executor were reused.
        executor.remove_node(node)
    # --- End Send Goal and Wait Logic ---


# --- Main Execution ---

if __name__ == '__main__':
    rclpy.init()
    node = None # Initialize node to None for finally block safety
    drive_client = None

    try:
        # Create the Node
        node = rclpy.create_node('minimal_pancake_robot_functions')
        logger = node.get_logger()
        logger.info("Minimal Pancake Robot Initializing...")

        # Create Action Client
        drive_client = ActionClient(node, DriveDistance, '/drive_distance')

        # Wait for Server
        logger.info("Waiting for DriveDistance Action Server...")
        if not drive_client.wait_for_server(timeout_sec=10.0):
            logger.error('DriveDistance action server not available! Exiting.')
            sys.exit(1) # Exit script if server isn't found
        logger.info("Action server found. Starting cycles...")

        # --- Pancake Cycles Loop ---
        for cycle in range(PANCAKES_TO_MAKE):
            if not rclpy.ok():
                logger.warn("ROS shutdown requested before starting cycle.")
                break

            logger.info(f"\n--- Starting Pancake Cycle {cycle + 1} / {PANCAKES_TO_MAKE} ---")

            # 1. Initial Wait at Start
            logger.info(f"Waiting at Start/Order station for {INITIAL_WAIT:.1f}s")
            time.sleep(INITIAL_WAIT)

            # 2. Loop Through Stations 1 to N
            cycle_successful = True
            for station_index in range(1, NUM_STATIONS + 1):
                if not rclpy.ok():
                    cycle_successful = False
                    logger.warn("ROS shutdown requested during station processing.")
                    break # Exit station loop

                station_name, process_time, distance = STATIONS[station_index]
                logger.info(f"-> Processing Station {station_index} ({station_name})")

                # Drive to the station and wait
                success = drive_and_wait(node, drive_client, distance, process_time)

                if not success:
                    logger.error(f"Failed action for station {station_index}. Aborting cycle.")
                    cycle_successful = False
                    break # Exit station loop

            if not cycle_successful:
                logger.warn(f"Cycle {cycle + 1} aborted due to failure or shutdown.")
                # Decide whether to break outer loop or try next cycle (here we continue)
                continue # Go to the next pancake cycle attempt

            if not rclpy.ok():
                logger.warn("ROS shutdown requested before returning to start.")
                break # Exit cycle loop

            # 3. Return to Start
            logger.info(f"-> Returning to Start (Station 0)")
            success = drive_and_wait(node, drive_client, RETURN_DISTANCE, 0) # Drive back, no wait needed

            if not success:
                logger.error("Failed to return to start. Stopping all cycles.")
                break # Exit the main pancake loop

            logger.info(f"--- Pancake Cycle {cycle + 1} Complete ---\n")
            time.sleep(1.0) # Small pause between cycles

        logger.info("All planned pancake cycles finished or process aborted.")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected, shutting down.")
    except Exception as e:
         # Use logger if available, otherwise print
         log_func = logger.error if node else print
         log_func(f"An unexpected error occurred: {e}")
         import traceback
         traceback.print_exc()
    finally:
        # Cleanup
        logger.info("Shutting down...")
        if node:
            node.destroy_node()
        rclpy.shutdown()
        logger.info("Shutdown complete.")