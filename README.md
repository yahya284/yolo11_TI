# Traffic Analysis and Accident Detection System

## Overview

This advanced traffic monitoring system combines real-time vehicle tracking, speed estimation, and accident detection using computer vision and deep learning techniques. The system processes video feeds from various sources, including YouTube live streams, to analyze traffic patterns, measure vehicle speeds, and automatically detect accidents using a custom-trained YOLOv11 model.

## Key Features

- Real-time vehicle detection and tracking
- Speed estimation using perspective transform
- Accident detection with fine-tuned YOLOv11
- Traffic flow analysis with heatmap visualization
- Automated data logging and statistics
- Support for multiple video sources
- Interactive visualization controls

## Project Structure

```
traffic-analysis/
│
├── src/
│   ├── tracker3.py           # Core tracking module
│   ├── youtube_stream_tracker1.py  # Stream processing
|   ├──test_it      #image_test_accident
|   ├──test_skuld_vid       #video_test_accident
│   └── heatmap.py           # Heat visualization
│
├── models/
│   └── yolo11n.pt          # Fine-tuned YOLOv11 model
│
├── data/
│   └── ...csv     # CSV output directory
│
│
├── requirements.txt       # Project dependencies
│
└── README.md             # This file
```

## Technical Details

### Core Components

1. **Vehicle Tracker (`tracker3.py`)**
   - Multi-object tracking system
   - Real-world speed calculation using perspective transform
   - Automated CSV logging of vehicle data
   - Customizable detection regions
   - Support for multiple vehicle classes

2. **Stream Processor (`youtube_stream_tracker1.py`)**
   - YouTube stream integration
   - Real-time visualization
   - Automatic reconnection handling
   - Interactive controls
   - Frame rate optimization

3. **Heatmap Generator (`heatmap.py`)**
   - Dynamic traffic density visualization
   - Customizable visualization parameters
   - Class-specific filtering options
   - Smooth Gaussian rendering

4. **Accident Detection Model**
   - Fine-tuned YOLOv11 architecture
   - Specialized accident detection capabilities
   - Real-time inference processing
   - High accuracy in various conditions
### Testing and Validation Tools

5. **Image Testing (`test_it.py`)**
   This utility allows you to test the accident detection model on individual images:
   ```python
   python test_it.py
   ```
   Key features:
   - Single image processing
   - Visualization of detection results
   - Confidence score display
   - Color-coded class identification
   
   Configuration steps:
   1. Replace `"Your API Key Here"` with your actual Roboflow API key
   2. Update `IMAGE_PATH` to point to your test image
   3. Verify the `MODEL_ID` matches your deployed model version

6. **Video Testing (`test_skuld_vid.py`)**
   This tool processes video files for accident detection:
   ```python
   python test_skuld_vid.py
   ```
   Features:
   - Full video processing capability
   - Real-time visualization
   - Output video generation
   - Frame-by-frame analysis
   
   Setup requirements:
   1. Insert your Roboflow API key
   2. Set appropriate video input/output paths
   3. Confirm model ID and version



## Installation Guide

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- FFmpeg installed on your system

### Setup Steps


1. **Clone the Repository**
   ```bash
   git clone https://github.com/yahya284/yolo11_TI.git
   cd traffic-analysis
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Model**
   ```bash
   # Create models directory and download fine-tuned model
   mkdir -p models
   # Add instructions for downloading your model
   ```



## Usage Guide

### Basic Implementation

```python
from youtube_stream_tracker1 import YouTubeStreamCounter

# Initialize system
counter = YouTubeStreamCounter(
    youtube_url="YOUR_STREAM_URL",
    model_path="models/yolo11n.pt",
    classes=[2, 5, 7]  # Vehicle classes to track
)

# Start processing
counter.run(display=True)
```

### Interactive Controls

- `q`: Exit application
- `p`: Pause/resume processing
- `h`: Toggle heatmap overlay
- `s`: Save current frame
- `d`: Toggle debug information

### Configuration Options

```python
# Advanced configuration example
counter = YouTubeStreamCounter(
    youtube_url="YOUR_STREAM_URL",
    region_points=[(300, 450), (210, 270), (645, 244), (930, 375)],
    model_path="models/yolo11n.pt",
    classes=[2, 5, 7],
    show_counts=True,
    heatmap_opacity=0.5,
    detection_radius=25
)
```

## Data Output

### CSV Format
The system generates dated CSV files with the following structure:
```
Track ID, Label, Action, Speed (km/h), Class, Date, Time
1, Car 1, IN, 45, car, 2025-01-05, 14:30:22
2, Truck 1, OUT, 35, truck, 2025-01-05, 14:30:23
```

### Analysis Tools
Included utilities for processing output data:
- Speed distribution analysis
- Traffic flow patterns
- Accident statistics
- Peak hour identification

## Performance Optimization

- Adjust `frame_skip` for processing speed
- Configure resolution scaling
- Optimize detection frequency
- Manage memory usage

## Accident Detection Capabilities

The fine-tuned YOLOv11 model provides:
- Real-time accident detection
- Severity classification

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## Troubleshooting

Common issues and solutions:
- Stream connection errors: Check network stability
- GPU memory issues: Adjust batch size
- CSV writing errors: Check file permissions
- Model loading fails: Verify CUDA compatibility








