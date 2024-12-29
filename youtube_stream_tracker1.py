import cv2
from tracker3 import ObjectCounter
from pathlib import Path
import logging
import yt_dlp

class YouTubeStreamCounter:
    def __init__(self, youtube_url, region_points_1=None,region_points_2=None ,model_path="yolo11n.pt", classes=None, show_counts=True):
        """
        Initialize the YouTube stream counter system.
        
        Args:
            youtube_url (str): URL of the YouTube live stream
            region_points (list): List of points defining the counting region
            model_path (str): Path to the YOLO model file
            classes (list): List of class IDs to detect
            show_counts (bool): Whether to show both in and out counts
        """
        self.youtube_url = youtube_url
        self.region_points_1 = region_points_1 or [(135, 208), (676, 203)]
        self.region_points_2 = region_points_2 or [(65, 33), (76, 20)]
        self.classes = classes or [2, 5, 7]  # Default classes if none provided
        
        # Initialize YouTube stream
        self.cap = self._initialize_stream()
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open YouTube stream: {youtube_url}")
            
        # Initialize the object counter
        self.counter = ObjectCounter(
            region_1=self.region_points_1,
            region_2=self.region_points_2,
            model=model_path,
            classes=self.classes,
            show_in=show_counts,
            show_out=show_counts,
            line_width=2
        )
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Mouse callback state
        self.mouse_position = None

    def _initialize_stream(self):
        """
        Initialize the YouTube stream using yt_dlp.
        
        Returns:
            cv2.VideoCapture: Initialized video capture object
        """
        try:
            ydl_opts = {
                'format': 'best[ext=mp4]',
                'quiet': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.youtube_url, download=False)
                stream_url = info['url']
                return cv2.VideoCapture(stream_url)
                
        except Exception as e:
            self.logger.error(f"Failed to initialize YouTube stream: {str(e)}")
            raise

    def _reconnect_stream(self):
        """Attempt to reconnect to the stream if connection is lost."""
        self.logger.info("Attempting to reconnect to stream...")
        try:
            self.cap.release()
            self.cap = self._initialize_stream()
            return self.cap.isOpened()
        except Exception as e:
            self.logger.error(f"Failed to reconnect: {str(e)}")
            return False

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for debugging and region selection."""
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_position = (x, y)
            self.logger.debug(f"Mouse position: {self.mouse_position}")

    def process_frame(self, frame):
        """
        Process a single frame through the counter.
        
        Args:
            frame (numpy.ndarray): Input frame to process
            
        Returns:
            numpy.ndarray: Processed frame with annotations
        """
        if frame is None:
            return None
            
        # Resize frame for consistent processing
        frame = cv2.resize(frame, (1020, 500))
        
        # Process the frame with the counter
        processed_frame = self.counter.count(frame)
        
        return processed_frame

    def run(self, display=True, frame_skip=2, max_reconnect_attempts=3):
        """
        Run the vehicle counter system on the YouTube stream.
        
        Args:
            display (bool): Whether to display the output window
            frame_skip (int): Number of frames to skip between processing
            max_reconnect_attempts (int): Maximum number of reconnection attempts
        """
        if display:
            window_name = "YouTube Stream Vehicle Counter"
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, self.mouse_callback)

        frame_count = 0
        reconnect_attempts = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame from stream")
                    if reconnect_attempts < max_reconnect_attempts:
                        reconnect_attempts += 1
                        if self._reconnect_stream():
                            self.logger.info("Successfully reconnected to stream")
                            continue
                    self.logger.error("Stream connection lost permanently")
                    break

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                processed_frame = self.process_frame(frame)
                
                if display and processed_frame is not None:
                    cv2.imshow(window_name, processed_frame)
                    
                    # Handle key events
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.logger.info("User requested quit")
                        break
                    elif key == ord('p'):
                        self.logger.info("Playback paused - press any key to continue")
                        cv2.waitKey(0)

                # Reset reconnect attempts on successful frame processing
                reconnect_attempts = 0

        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")
        except Exception as e:
            self.logger.error(f"Error during processing: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        self.logger.info("Resources cleaned up")

    def get_counts(self):
        """
        Get the current counting statistics.
        
        Returns:
            dict: Dictionary containing in/out counts and class-wise statistics
        """
        return {
            'in_count': self.counter.in_count,
            'out_count': self.counter.out_count,
            'class_counts': self.counter.classwise_counts
        }

    @property
    def is_running(self):
        """Check if the stream capture is still active."""
        return self.cap is not None and self.cap.isOpened()

def main():
    """Example usage of the YouTubeStreamCounter class."""
    # Example YouTube live stream URL (replace with actual traffic camera stream)
    youtube_url = "https://www.youtube.com/watch?v=ByED80IKdIU"
    
    try:
        # Initialize and run the counter system
        counter_system = YouTubeStreamCounter(
            youtube_url=youtube_url,
            region_points_1=[(135, 208), (676, 203)], 
            region_points_2=[(235, 135), (376, 50)], 
            classes=[2, 5, 7],
            show_counts=True
        )
        
        # Run the system
        counter_system.run(display=True, frame_skip=2)
        
        # Get final counts
        final_counts = counter_system.get_counts()
        print("\nFinal Counts:")
        print(f"Total IN: {final_counts['in_count']}")
        print(f"Total OUT: {final_counts['out_count']}")
        print("\nClass-wise counts:")
        for class_name, counts in final_counts['class_counts'].items():
            print(f"{class_name}: IN={counts['IN']}, OUT={counts['OUT']}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()