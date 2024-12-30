import os
import cv2
import pandas as pd
from datetime import datetime
import numpy as np
import torch
from time import time
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors

class ObjectCounter(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.in_count = 0
        self.out_count = 0
        self.counted_ids = []
        self.saved_ids = []
        self.classwise_counts = {}
        self.region_initialized = False
        self.track_history = {}  # Store tracking history for each ID
        self.tracked_speeds = {}  # Store calculated speeds
        
        # Speed tracking parameters
        self.speed_tracking = {
            'positions': {},     # Last known positions
            'timestamps': {},    # Last timestamp for each ID
            'smoothed_speeds': {} # Smoothed speed values
        }
        
        # Configuration parameters
        self.pixels_per_meter = 0.1  # Calibration factor (adjust based on camera view)
        self.speed_smoothing_factor = 0.3  # For exponential smoothing
        self.min_speed_distance = 50  # Minimum pixels to calculate speed
        self.min_speed_time = 0.1     # Minimum seconds between measurements
        
        self.show_in = self.CFG.get("show_in", True)
        self.show_out = self.CFG.get("show_out", True)

        # Initialize CSV data storage
        self.csv_filename = self.get_daily_filename()
        self.create_csv()

    def get_daily_filename(self):
        """Generate a filename based on the current date."""
        current_date = datetime.now().strftime("%Y-%m-%d")
        filename = f"vehicle_count_data_{current_date}.csv"
        return filename

    def create_csv(self):
        """Create the CSV file with proper headers if it doesn't exist."""
        if not os.path.exists(self.csv_filename):
            header = ["Track ID", "Label", "Action", "Speed (km/h)", "Class", "Date", "Time"]
            df = pd.DataFrame(columns=header)
            df.to_csv(self.csv_filename, index=False)
            print(f"CSV file created: {self.csv_filename} with headers.")

    def save_label_to_file(self, track_id, label, action, speed, class_name):
        """Save detection data to CSV file."""
        # Convert speed to scalar if needed
        if isinstance(speed, (torch.Tensor, np.ndarray)):
            speed = float(speed)
        
        # Round speed to integer
        speed = int(round(float(speed)))

        current_time = datetime.now()
        data = {
            "Track ID": track_id,
            "Label": label,
            "Action": action,
            "Speed (km/h)": speed,
            "Class": class_name,
            "Date": current_time.date(),
            "Time": current_time.strftime("%H:%M:%S")
        }

        df = pd.DataFrame([data])
        df.to_csv(self.csv_filename, mode='a', header=False, index=False)

    def calculate_speed(self, track_id, current_position, current_time):
        """
        Calculate speed for a tracked object using time and position differences.
        Returns speed in km/h.
        """
        tracking = self.speed_tracking
        
        # Initialize tracking for new objects
        if track_id not in tracking['positions']:
            tracking['positions'][track_id] = current_position
            tracking['timestamps'][track_id] = current_time
            tracking['smoothed_speeds'][track_id] = 0
            return 0

        # Get time difference
        time_diff = current_time - tracking['timestamps'][track_id]
        if time_diff < self.min_speed_time:
            return tracking['smoothed_speeds'][track_id]

        # Calculate distance in pixels
        prev_pos = tracking['positions'][track_id]
        distance_px = np.sqrt(
            (current_position[0] - prev_pos[0]) ** 2 +
            (current_position[1] - prev_pos[1]) ** 2
        )

        if distance_px < self.min_speed_distance:
            return tracking['smoothed_speeds'][track_id]

        # Convert to real-world units and calculate speed
        distance_m = distance_px * self.pixels_per_meter
        speed_mps = distance_m / time_diff
        speed_kmh = speed_mps * 3.6  # Convert m/s to km/h

        # Apply exponential smoothing
        if track_id in tracking['smoothed_speeds']:
            prev_speed = tracking['smoothed_speeds'][track_id]
            speed_kmh = (prev_speed * (1 - self.speed_smoothing_factor) +
                        speed_kmh * self.speed_smoothing_factor)

        # Update tracking data
        tracking['positions'][track_id] = current_position
        tracking['timestamps'][track_id] = current_time
        tracking['smoothed_speeds'][track_id] = speed_kmh

        return speed_kmh

    def count_objects(self, current_centroid, track_id, prev_position, cls):
        """Count objects and update file based on centroid movements."""
        if prev_position is None or track_id in self.counted_ids:
            return

        action = None
        current_speed = self.calculate_speed(track_id, current_centroid, time())

        # Handle linear region counting
        if len(self.region) == 2:
            line = self.LineString(self.region)
            if line.intersects(self.LineString([prev_position, current_centroid])):
                # Determine direction based on dominant axis
                if abs(self.region[0][0] - self.region[1][0]) < abs(self.region[0][1] - self.region[1][1]):
                    # Vertical line case
                    action = "IN" if current_centroid[0] > prev_position[0] else "OUT"
                else:
                    # Horizontal line case
                    action = "IN" if current_centroid[1] > prev_position[1] else "OUT"
                    
                # Update counts
                if action == "IN":
                    self.in_count += 1
                    self.classwise_counts[self.names[cls]]["IN"] += 1
                else:
                    self.out_count += 1
                    self.classwise_counts[self.names[cls]]["OUT"] += 1
                    
                self.counted_ids.append(track_id)

        # Handle polygonal region counting
        elif len(self.region) > 2:
            polygon = self.Polygon(self.region)
            if polygon.contains(self.Point(current_centroid)):
                action = "IN" if current_centroid[0] > prev_position[0] else "OUT"
                
                if action == "IN":
                    self.in_count += 1
                    self.classwise_counts[self.names[cls]]["IN"] += 1
                else:
                    self.out_count += 1
                    self.classwise_counts[self.names[cls]]["OUT"] += 1
                    
                self.counted_ids.append(track_id)

        # Save data if action was determined
        if action:
            label = f"{self.names[cls]} ID: {track_id}"
            self.save_label_to_file(track_id, label, action, current_speed, self.names[cls])
            self.tracked_speeds[track_id] = current_speed

    def store_classwise_counts(self, cls):
        """Initialize count dictionary for a given class."""
        if self.names[cls] not in self.classwise_counts:
            self.classwise_counts[self.names[cls]] = {"IN": 0, "OUT": 0}

    def display_counts(self, im0):
        """Display counts and speed information on the image."""
        # Display class-wise counts
        labels_dict = {
            str.capitalize(key): f"{'IN ' + str(value['IN']) if self.show_in else ''} "
                                f"{'OUT ' + str(value['OUT']) if self.show_out else ''}".strip()
            for key, value in self.classwise_counts.items()
            if value["IN"] != 0 or value["OUT"] != 0
        }

        if labels_dict:
            self.annotator.display_analytics(im0, labels_dict, (104, 31, 17), (255, 255, 255), 10)

        # Display object boxes with speed
        for track_id in self.track_ids:
            track_index = self.track_ids.index(track_id)
            cls = self.clss[track_index]
            box = self.boxes[track_index]
            
            # Get speed for display
            speed = self.tracked_speeds.get(track_id, 0)
            speed_label = f"{int(speed)} km/h" if speed > 0 else self.names[int(cls)]
            
            # Combine label information
            label = f"{speed_label}, ID: {track_id}"
            self.annotator.box_label(box, label=label, color=colors(int(cls), True))

    
    def count(self, im0):
        """Main counting function to track objects and update counts."""
        if not self.region_initialized:
            self.initialize_region()
            self.region_initialized = True

        self.annotator = Annotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)
        
        # Draw counting region
        self.annotator.draw_region(reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2)

        # Process each tracked object
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            # Initialize tracking history if needed
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            
            # Get current centroid
            current_centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            self.track_history[track_id].append(current_centroid)
            
            # Keep track history at a reasonable length
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)
            
            self.store_classwise_counts(cls)
            
            # Get previous position for direction calculation
            prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None
            
            # Process counting and speed calculation
            self.count_objects(current_centroid, track_id, prev_position, cls)
            
            # Draw tracking visualization
            if len(self.track_history[track_id]) > 1:
                self.annotator.draw_centroid_and_tracks(
                    self.track_history[track_id],
                    color=colors(int(track_id), True),
                    track_thickness=self.line_width
                )

        # Display results
        self.display_counts(im0)
        return im0