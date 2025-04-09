#!/usr/bin/env python3

import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls
import time

class ColorDetector:
    def __init__(self):
        # Camera setup
        self.camera = Picamera2()
        config = self.camera.create_preview_configuration(main={"size": (640, 480)})
        self.camera.configure(config)
        self.camera.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        self.camera.start()
        time.sleep(2)  # Allow camera to initialize

        # Color ranges for different stations (HSV format)
        self.station_colors = {
            "Order Station": {"lower": (100, 100, 50), "upper": (130, 255, 255)},  # Blue
            "Batter Station": {"lower": (0, 100, 100), "upper": (10, 255, 255)},   # Red
            "Topping 1": {"lower": (20, 100, 100), "upper": (40, 255, 255)},       # Yellow
            "Topping 2": {"lower": (50, 100, 50), "upper": (80, 255, 255)},        # Green
            "Topping 3": {"lower": (140, 100, 50), "upper": (170, 255, 255)}       # Pink
        }

    def detect_colors(self):
        try:
            while True:
                # Capture frame
                frame = self.camera.capture_array()
                
                # Convert to HSV
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # Create a blank image for showing all detected colors
                color_mask = np.zeros_like(frame)
                
                # Detect each station's color
                for station_name, color_range in self.station_colors.items():
                    lower = np.array(color_range["lower"])
                    upper = np.array(color_range["upper"])
                    
                    # Create mask for this color
                    mask = cv2.inRange(hsv, lower, upper)
                    
                    # Count pixels of this color
                    detected_pixels = cv2.countNonZero(mask)
                    
                    # Add text showing pixel count
                    text = f"{station_name}: {detected_pixels}"
                    y_position = 30 * (list(self.station_colors.keys()).index(station_name) + 1)
                    cv2.putText(frame, text, (10, y_position), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Highlight detected areas
                    color_detected = cv2.bitwise_and(frame, frame, mask=mask)
                    color_mask = cv2.add(color_mask, color_detected)

                # Show original frame with text
                cv2.imshow("Camera Feed", frame)
                
                # Show detected colors
                cv2.imshow("Detected Colors", color_mask)
                
                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Stopping camera feed...")
        finally:
            self.cleanup()

    def cleanup(self):
        self.camera.stop()
        cv2.destroyAllWindows()

def main():
    detector = ColorDetector()
    detector.detect_colors()

if __name__ == '__main__':
    main()