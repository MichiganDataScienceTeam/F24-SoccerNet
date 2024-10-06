import numpy as np
import supervision as sv
from roboflow import Roboflow
import pickle
import cv2

# Paths to files
PICKLE_FILE_PATH = "video_detections.pkl"  # Path to the saved detections
SOURCE_VIDEO_PATH = "source_video.mov"      # Path to the source video file
TARGET_VIDEO_PATH = "video_out.mov"         # Path to the output video file

# Initialize Roboflow and model
rf = Roboflow(api_key="2c8BbR867fBO7Rmyg9VI")
project = rf.workspace().project("football-players-detection-3zvbc")
model = project.version(9).model

# Create BYTETracker instance
byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)

# Create instance of BoxAnnotator and TraceAnnotator
box_annotator = sv.BoxAnnotator(thickness=4)
trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)

# Function to load detections from a pickle file
def load_detections_from_pickle(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        detections = pickle.load(f)
    print(f"Loaded detections from {pickle_file_path}")
    return detections

# Load detections from pickle file
all_detections = load_detections_from_pickle(PICKLE_FILE_PATH)

# Open the source video
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)

# Get video information
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'MJPG', etc.
out = cv2.VideoWriter(TARGET_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

# Process each detection and corresponding frame
for index, detections in enumerate(all_detections):
    ret, frame = cap.read()  # Read a frame from the source video
    if not ret:
        break  # Exit if the video ends

    # Prepare labels for box annotations
    labels = [
        f"{detections.data['class_name'][i]} {detections.confidence[i]:0.2f}"
        for i in range(len(detections.confidence))
    ]

    # Annotate frame with traces and boxes
    annotated_frame = trace_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )

    out.write(annotated_frame)

# Release the VideoCapture and VideoWriter objects
cap.release()
out.release()
print(f"Processed video saved to {TARGET_VIDEO_PATH}")


# Shiva's favorite animal is the red panda