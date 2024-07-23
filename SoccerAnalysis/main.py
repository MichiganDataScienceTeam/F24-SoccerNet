import numpy as np
from roboflow import Roboflow
from tracker import Tracker
import videoUtils

SOURCE_VIDEO_PATH = "shortTestVid.mov"
TARGET_VIDEO_PATH = "new_vid.mp4"
STUB_PATH = "stub_tracks.pkl"
PLAYER_PROJECT_NAME = "football-players-detection-3zvbc"
VERSION_NUMBER = 1
PlAYER_API_KEY = ""


# Initialize the Plyer Roboflow model and Tracker
player_rf = Roboflow(api_key=PlAYER_API_KEY)
player_project = player_rf.workspace().project(PLAYER_PROJECT_NAME)
player_model = player_project.version(VERSION_NUMBER).model
print("Roboflow models and trackers initialized")

# Create Tracker instance
tracker = Tracker(PlAYER_API_KEY, PLAYER_PROJECT_NAME, VERSION_NUMBER)

frames = videoUtils.read_video(SOURCE_VIDEO_PATH)
print("Reading frames")

# Check if frames are a numpy array
if isinstance(frames, np.ndarray):
    
    # Get object tracks
    tracks = tracker.get_object_tracks(frames, read_from_stub=True, stub_path=STUB_PATH)

    # Draw annotations
    annotated_frames = tracker.draw_annotations(frames, tracks)
    print("Drawing annotations")
    
    videoUtils.save_video(annotated_frames, TARGET_VIDEO_PATH)
    print("Video saved successfully")
else:
        print("Error: Invalid frames data")
