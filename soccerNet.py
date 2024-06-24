import numpy as np
import supervision as sv
from roboflow import Roboflow

SOURCE_VIDEO_PATH = "Soccergame.mov"
TARGET_VIDEO_PATH = "video_out.mov"

# use https://roboflow.github.io/polygonzone/ to get the points for your line
LINE_START = sv.Point(0, 300)
LINE_END = sv.Point(800, 300)

rf = Roboflow(api_key="2c8BbR867fBO7Rmyg9VI")
project = rf.workspace().project("football-players-detection-3zvbc")
model = project.version(9).model

# create BYTETracker instance
byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)

# create VideoInfo instance
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

# create frame generator
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# create LineZone instance, it is previously called LineCounter class
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

# create instance of BoxAnnotator
box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

# create instance of TraceAnnotator
trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)
line_zone_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)

# define call back function to be used in video processing
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    # Model prediction on single frame and conversion to supervision Detections
    results = model.predict(frame).json()
    detections = sv.Detections.from_roboflow(results)

    # Show detections in real time
    print(detections)

    # Tracking detections
    detections = byte_tracker.update_with_detections(detections)
    print("Updated Detections:", detections)

    # Adjust unpacking according to the structure of detections
    labels = [
        f"{detections.data['class_name'][i]} {detections.confidence[i]:0.2f}"
        for i in range(len(detections.confidence))
    ]

    annotated_frame = trace_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )

    # Update line counter
    line_zone.trigger(detections)
    
    # Return frame with box and line annotated result
    return line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

# Process the whole video
sv.process_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path=TARGET_VIDEO_PATH,
    callback=callback
)