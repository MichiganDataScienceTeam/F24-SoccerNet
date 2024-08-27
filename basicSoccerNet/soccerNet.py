import numpy as np
import supervision as sv
from roboflow import Roboflow
import pickle
import cv2

# Paths to video files
PICKLE_FILE_PATH = "video_frames.pkl"
TARGET_VIDEO_PATH = "video_out.mov"

# Initialize Roboflow and model
rf = Roboflow(api_key="")
project = rf.workspace().project("football-players-detection-3zvbc")
model = project.version(9).model

# Create BYTETracker instance
byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)

# Create instance of BoxAnnotator and TraceAnnotator
box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)

# Function to load video frames from a pickle file
def load_frames_from_pickle(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        frames = pickle.load(f)
    print(f"Loaded {len(frames)} frames from {pickle_file_path}")
    return frames

# Define callback function for processing frames
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    # Model prediction on single frame and conversion to supervision Detections
    results = model.predict(frame).json()
    detections = sv.Detections.from_roboflow(results)

    # Tracking detections
    detections = byte_tracker.update_with_detections(detections)

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
    
    # Return annotated frame
    return annotated_frame

# Load frames from pickle file
frames = load_frames_from_pickle(PICKLE_FILE_PATH)

# Get video information (assuming all frames have the same dimensions)
frame_height, frame_width = frames[0][1].shape[:2]
fps = 30  # Assuming 30 frames per second; adjust if different

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'MJPG', etc.
out = cv2.VideoWriter(TARGET_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

# Process each frame and write to the output video
for index, frame in frames:
    annotated_frame = callback(frame, index)
    out.write(annotated_frame)

# Release the VideoWriter object
out.release()
print(f"Processed video saved to {TARGET_VIDEO_PATH}")
