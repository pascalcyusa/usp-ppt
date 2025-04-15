#!/usr/bin/env python3

import os
from dotenv import load_dotenv
import time
import requests
import json

# Load environment variables from .env file
load_dotenv()

# --- Airtable Configuration ---
AIRTABLE_API_TOKEN = os.getenv('AIRTABLE_API_TOKEN')
AIRTABLE_BASE_ID = os.getenv('AIRTABLE_BASE_ID')
AIRTABLE_TABLE_NAME = os.getenv('AIRTABLE_TABLE_NAME')

if not all([AIRTABLE_API_TOKEN, AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME]):
    raise EnvironmentError(
        "Missing required Airtable environment variables. Please check your .env file.")

# --- Construct Airtable URL and Headers ---
AIRTABLE_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
AIRTABLE_HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_API_TOKEN}",
    "Content-Type": "application/json",
}

# --- Field names in Airtable base (MUST match exactly, case-sensitive) ---
AIRTABLE_ORDER_NAME_COLUMN = "Order Name"       # Column for the order identifier

# Station Status Fields (Numeric)
AIRTABLE_COOKING_1_STATUS_FIELD = "Cooking 1 Status"
AIRTABLE_ROBOT2_WAIT_STATUS_FIELD = "Cooking 2 Status"
AIRTABLE_WHIPPED_CREAM_STATUS_FIELD = "Whipped Cream Status"
AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD = "Choco Chips Status"
AIRTABLE_SPRINKLES_STATUS_FIELD = "Sprinkles Status"
AIRTABLE_PICKUP_STATUS_FIELD = "Pickup Status"

# --- Airtable Status Codes (Numeric) ---
STATUS_WAITING = 0
STATUS_ARRIVED = 1
STATUS_DONE = 99

def fetch_all_orders():
    """Fetches all orders and their status from Airtable."""
    try:
        response = requests.get(url=AIRTABLE_URL, headers=AIRTABLE_HEADERS)
        response.raise_for_status()
        data = response.json()
        return data.get('records', [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching orders: {e}")
        return []

def update_station_status(record_id, station_field_name, new_status_code):
    """Updates a specific station's status for an order."""
    if not record_id or not station_field_name:
        print("Cannot update Airtable: record_id or station_field_name missing.")
        return False

    update_data = {
        "fields": {
            station_field_name: new_status_code
        }
    }

    try:
        response = requests.patch(
            url=f"{AIRTABLE_URL}/{record_id}",
            headers=AIRTABLE_HEADERS,
            json=update_data
        )
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error updating station status: {e}")
        return False

def get_order_status(record_id):
    """Gets the current status of a specific order."""
    try:
        response = requests.get(
            url=f"{AIRTABLE_URL}/{record_id}",
            headers=AIRTABLE_HEADERS
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting order status: {e}")
        return None

def wait_for_status(record_id, station_field_name, target_status, timeout=120):
    """Waits for a station to reach a specific status."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        order_data = get_order_status(record_id)
        if order_data and order_data.get('fields', {}).get(station_field_name) == target_status:
            return True
        time.sleep(2)  # Poll every 2 seconds
    return False

def main():
    # Fetch and display all orders
    print("\nFetching all orders...")
    orders = fetch_all_orders()
    for order in orders:
        print(f"\nOrder: {order['fields'].get(AIRTABLE_ORDER_NAME_COLUMN, 'Unknown')}")
        for field, value in order['fields'].items():
            print(f"{field}: {value}")

    # Get the first order that needs processing
    if not orders:
        print("No orders found!")
        return

    test_order = orders[0]
    record_id = test_order['id']
    order_name = test_order['fields'].get(AIRTABLE_ORDER_NAME_COLUMN, 'Unknown')
    print(f"\nTesting with order: {order_name} (ID: {record_id})")

    # Test sequence of station updates
    stations_to_test = [
        AIRTABLE_COOKING_1_STATUS_FIELD,
        AIRTABLE_ROBOT2_WAIT_STATUS_FIELD,
        AIRTABLE_CHOCOLATE_CHIPS_STATUS_FIELD,
        AIRTABLE_WHIPPED_CREAM_STATUS_FIELD,
        AIRTABLE_SPRINKLES_STATUS_FIELD,
        AIRTABLE_PICKUP_STATUS_FIELD
    ]

    for station in stations_to_test:
        print(f"\nTesting {station}")
        print("Waiting 5 seconds before updating status...")
        time.sleep(5)

        # Update status to 1 (arrived)
        print(f"Updating {station} to STATUS_ARRIVED (1)")
        if update_station_status(record_id, station, STATUS_ARRIVED):
            print("Update successful, waiting for STATUS_DONE (99)...")
            
            # Wait for status to become 99
            if wait_for_status(record_id, station, STATUS_DONE):
                print(f"{station} completed successfully!")
                print("Moving to next station...")
            else:
                print(f"Timeout waiting for {station} to complete!")
                break
        else:
            print(f"Failed to update {station}!")
            break

    print("\nAirtable testing complete!")

if __name__ == "__main__":
    main()