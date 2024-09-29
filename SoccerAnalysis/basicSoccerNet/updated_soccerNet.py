import numpy as np
import supervision as sv
from roboflow import Roboflow
from inference import get_model


SOURCE_VIDEO_PATH = "shortTestVid.mov"
TARGET_VIDEO_PATH = "video_out.mp4"

rf = Roboflow(api_key="")
#project = rf.workspace().project("football-players-detection-3zvbc")
model = get_model(model_id="football-players-detection-3zvbc/9")
#model = project.version(9).model
#print("model is 1111", model)

# Create BYTETracker instance
byte_tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30, minimum_matching_threshold=0.8, frame_rate=1)

# Create VideoInfo instance
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

# Create frame generator
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# Create instance of BoxAnnotator
box_annotator = sv.BoxAnnotator(thickness=4)

# Create instance of TraceAnnotator
trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)

# Define callback function to be used in video processing
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    # Model prediction on single frame and conversion to supervision Detections
    print('callback')
    print("frame is ", frame)
    print("model is ", model)
    results = model.infer(frame)
    detections = sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))

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
    )
    
    # Return annotated frame
    return annotated_frame

# Process the whole video
sv.process_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path=TARGET_VIDEO_PATH,
    callback=callback
)
