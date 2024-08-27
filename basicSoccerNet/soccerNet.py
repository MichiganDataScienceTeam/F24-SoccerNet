import numpy as np
import supervision as sv
from roboflow import Roboflow
import pickle

# Paths to video files
SOURCE_VIDEO_PATH = "Soccergame.mov"
TARGET_VIDEO_PATH = "video_out.mov"
PICKLE_FILE_PATH = "video_frames.pkl"

# Initialize Roboflow and model
rf = Roboflow(api_key="")
project = rf.workspace().project("football-players-detection-3zvbc")
model = project.version(9).model

# Create BYTETracker instance
byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)

# Create VideoInfo instance
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

# Create frame generator
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# Create instance of BoxAnnotator and TraceAnnotator
box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)

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

# Function to save video frames to a pickle file
def save_frames_to_pickle(generator, pickle_file_path):
    frames = []
    for index, frame in enumerate(generator):
        frames.append((index, frame))
    
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(frames, f)
    print(f"Saved {len(frames)} frames to {pickle_file_path}")

# Function to load video frames from a pickle file
def load_frames_from_pickle(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        frames = pickle.load(f)
    print(f"Loaded {len(frames)} frames from {pickle_file_path}")
    return frames

# Option to save frames to pickle for later use
save_frames_to_pickle(generator, PICKLE_FILE_PATH)

# Option to load frames from pickle file
frames = load_frames_from_pickle(PICKLE_FILE_PATH)

sv.process_video(
     source_path=SOURCE_VIDEO_PATH,
     target_path=TARGET_VIDEO_PATH,
     callback=callback
)
