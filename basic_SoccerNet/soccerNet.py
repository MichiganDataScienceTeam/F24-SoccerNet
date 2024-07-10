import numpy as np
import supervision as sv
from roboflow import Roboflow

SOURCE_VIDEO_PATH = "Soccergame.mov"
TARGET_VIDEO_PATH = "video_out.mov"

rf = Roboflow(api_key="2c8BbR867fBO7Rmyg9VI")
project = rf.workspace().project("football-players-detection-3zvbc")
model = project.version(9).model

# Create BYTETracker instance
byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)

# Create VideoInfo instance
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

# Create frame generator
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# Create instance of BoxAnnotator
box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

# Create instance of TraceAnnotator
trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)

# Define callback function to be used in video processing
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    # Model prediction on single frame and conversion to supervision Detections
    results = model.predict(frame).json()
    detections = sv.Detections.from_roboflow(results)

    # Show detections in real time
    print(detections)

    # Tracking detections
    detections = byte_tracker.update_with_detections(detections)
    print("Updated Detections:", detections)

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

# Process the whole video
sv.process_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path=TARGET_VIDEO_PATH,
    callback=callback
)
