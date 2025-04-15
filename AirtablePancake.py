# functions for accessing and changing data in the Pancake System Airtable
# Dylan Chen
import requests

class at:

    # INITIALIZER FUNCTION
    #   Initializer for the airtable class. Defines the Airtable API Key, Base ID, Table Name, url, and headers. 
    def __init__(self):
        # Airtable API setup
        self.AIRTABLE_API_KEY = "pat9eVlOP9knFawW5.cfe50b9999314cff7122fc2ec4373338acb96d860b0d43aab09b619733fc1a23"
        self.AIRTABLE_BASE_ID = "app4CfINDWGkqlxrN"
        self.AIRTABLE_TABLE_NAME = "Orders"
        self.url = f"https://api.airtable.com/v0/{self.AIRTABLE_BASE_ID}/{self.AIRTABLE_TABLE_NAME}"

        self.headers = {
            "Authorization": f"Bearer {self.AIRTABLE_API_KEY}",
            "Content-Type": "application/json"
        }

    # CHECK VALUE FUNCTION
    #   Inputs: 
    #       station: input a string of the name of the station you would like to check the value of.
    #           - Note: the spelling, spacing and punctuation must be exactly as written in airtable. 
    #   Returns: the value of the cell of the requested station for the current order. 
    def checkValue(self, station):

        params = {
            'sort[0][field]': 'Created',
            'sort[0][direction]': 'asc',
        }

        response = requests.get(self.url, headers=self.headers, params=params)
        if response.status_code == 200 or response.status_code == 201:
            data = response.json()
            # Extract and print the desired field from each record
            for record in data['records']:
                #print(f"Checking: {record['fields'].get("Order Name")}, Pickup Status is: {record['fields'].get("Pickup Status")}")
                if record['fields'].get("Pickup Status") != 99:
                    print(f"The value at the {station} cell of the current order is: {record['fields'].get(station)}")
                    return record['fields'].get(station)
        else:
            print("Failed to upload order. Status code:", response.status_code)
            print("Response:", response.json())

    # CHANGE VALUE FUNCTION
    #   Inputs: 
    #       station: input a string of the name of the station you would like to change the value of.
    #           - Note: the spelling, spacing and punctuation must be exactly as written in airtable. 
    #       value: the value you would like to change the station status to. 
    #   Returns: N/A 
    def changeValue(self, station, value):

        current_order = 0

        params = {
            'sort[0][field]': 'Created',
            'sort[0][direction]': 'asc', 
        }

        response = requests.get(self.url, headers=self.headers, params=params)
        if response.status_code == 200 or response.status_code == 201:
            data = response.json()
            # Find the current order by checking for a pickup status of 0
            for record in data['records']:
                if record['fields'].get("Pickup Status") != 99:
                    print(f"Current Order is: {record.get("id")}, Order Name: {record['fields'].get("Order Name")}")
                    current_order = record.get("id")
                    break
        else:
            print("Failed Status code:", response.status_code)
            print("Response:", response.json())

        patch_url = f'{self.url}/{current_order}'
        data = {
            "fields": {
                station : value
            }
        }

        # Send to Airtable
        response = requests.patch(patch_url, headers=self.headers, json=data)
        # Check response
        if response.status_code == 200 or response.status_code == 201:
            print(f"Value succesfully updated to {value}")
        else:
            print("Failed to upload order. Status code:", response.status_code)
            print("Response:", response.json())

