import os
import cv2
import pandas as pd
from datetime import datetime
import numpy as np
import torch
from time import time
from collections import defaultdict, deque
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
        
        # Initialize perspective transform (using ROI points for transform calculation)
        self.source_points = np.array([
            [300, 450],  # Bottom left
            [210, 270],  # Top left
            [645, 244],  # Top right
            [930, 375]   # Bottom right
        ], dtype=np.float32)
        
        # Define target points for bird's eye view (adjusted for typical road dimensions)
        self.target_width = 15  # meters (typical road width)
        self.target_height = 30
          # meters (visible road length)
        self.target_points = np.array([
            [0, self.target_height],
            [0, 0],
            [self.target_width, 0],
            [self.target_width, self.target_height]
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        self.transform_matrix = cv2.getPerspectiveTransform(self.source_points, self.target_points)
        
        # Speed tracking parameters
        self.coordinates = defaultdict(lambda: deque(maxlen=15))  # Reduced buffer for more responsive updates
        self.speed_tracking = {}  # Store current speeds
        self.speed_scale_factor = 0.7  # Adjustment factor for speed calibration
        
        self.show_in = self.CFG.get("show_in", True)
        self.show_out = self.CFG.get("show_out", True)

        # Initialize CSV data storage
        self.csv_filename = self.get_daily_filename()
        self.create_csv()

    def transform_point(self, point):
        """Transform a point using perspective transform matrix."""
        if isinstance(point, (tuple, list)):
            point = np.array(point)
        pts = point.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(pts, self.transform_matrix)
        return transformed.reshape(-1, 2)

    def get_daily_filename(self):
        """Generate a filename based on the current date."""
        current_date = datetime.now().strftime("%Y-%m-%d")
        filename = f"vehicle_count_data_{current_date}.csv"
        return filename

    def create_csv(self):
        """Create the CSV file with proper headers if it doesn't exist."""
        if not os.path.exists(self.csv_filename):
            directory = "C:\\Users\\pcexp\\Desktop\\yolo11_TI\\Data"
            self.csv_filename = os.path.join(directory, "output.csv") 
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

    def calculate_speed(self, track_id, current_position):
        """
        Calculate speed using perspective transformed coordinates.
        Returns speed in km/h.
        """
        # Store the current position
        self.coordinates[track_id].append(current_position)
        
        # Need at least 2 points to calculate speed
        if len(self.coordinates[track_id]) < 2:
            return 0
            
        # Transform points to bird's eye view
        points = np.array(list(self.coordinates[track_id]))
        transformed_points = self.transform_point(points)
        
        # Calculate speed using the last two points
        if len(transformed_points) >= 2:
            # Get the last two points
            p1 = transformed_points[-2]
            p2 = transformed_points[-1]
            
            # Calculate distance in meters
            distance = np.sqrt(((p2 - p1) ** 2).sum())
            
            # Calculate time (assuming 30 fps)
            time = 1/30.0  # seconds between frames
            
            # Calculate speed with calibration factor (m/s to km/h)
            speed_mps = distance / time
            
            # Apply perspective correction factor (speed appears faster from side view)
            perspective_factor = 0.3  # Reduce speed estimation for side view
            speed_kmh = speed_mps * 3.6 * self.speed_scale_factor * perspective_factor
            
            # Apply threshold to handle unrealistic speeds
            speed_kmh = min(speed_kmh, 50)  # Cap at 50 km/h for urban roads
            
            # Update speed tracking with smoothing
            if track_id in self.speed_tracking:
                prev_speed = self.speed_tracking[track_id]
                speed_kmh = 0.85 * prev_speed + 0.15 * speed_kmh  # Smooth transitions
            
            self.speed_tracking[track_id] = speed_kmh
            
            return speed_kmh
            
        return self.speed_tracking.get(track_id, 0)

    def count_objects(self, current_centroid, track_id, prev_position, cls):
        """Count objects and update file based on centroid movements."""
        if prev_position is None:
            return

        # Calculate speed for every frame
        current_speed = self.calculate_speed(track_id, current_centroid)
        # Always update speed for display
        self.tracked_speeds[track_id] = current_speed

        # Process counting logic
        if track_id in self.counted_ids:
            return

        action = None

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

        # Display object boxes with speed for all tracked objects
        for track_id in self.track_ids:
            track_index = self.track_ids.index(track_id)
            cls = self.clss[track_index]
            box = self.boxes[track_index]
            
            # Always show current speed
            speed = self.tracked_speeds.get(track_id, 0)
            speed_label = f"{int(speed)} km/h"
            
            # Combine label information
            label = f"{self.names[int(cls)]} {speed_label}"
            self.annotator.box_label(box, label=label, color=colors(int(cls), True))
    
    def count(self, im0):
        """Main counting function to track objects and update counts."""
        if not self.region_initialized:
            self.initialize_region()
            self.region_initialized = True

        self.annotator = Annotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)

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
