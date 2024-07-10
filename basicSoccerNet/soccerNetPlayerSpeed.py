import numpy as np
import supervision as sv
from roboflow import Roboflow
import collections

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

# Create instance of BoxAnnotator with smaller text
box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1.5)

# Create instance of TraceAnnotator
trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50)

# We will be using a dictionary to store the last known positions of the trackers
last_positions = {}
speed_buffers = collections.defaultdict(lambda: collections.deque(maxlen=5))  # Buffer to store recent speeds
frame_counts = collections.defaultdict(int)  # To keep track of frames for each tracker
last_speeds = collections.defaultdict(lambda: "Calculating...")  # Store the last computed speeds

def get_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

# Define the callback function to be used in video processing
def callback(frame: np.ndarray, index: int, video_fps: float) -> np.ndarray:
    # Model prediction on single frame and conversion to supervision Detections
    results = model.predict(frame).json()
    detections = sv.Detections.from_inference(results)

    # Display detections in real time
    print(detections)

    # Update tracking detections
    detections = byte_tracker.update_with_detections(detections=detections)
    print("Updated Detections:", detections)

    # Prepare labels for box annotations
    labels = []
    for i, bbox in enumerate(detections.xyxy):
        tracker_id = detections.tracker_id[i]

        centroid = get_centroid(bbox)
        frame_counts[tracker_id] += 1  # Increment the frame count for this tracker

        if tracker_id in last_positions:
            # Calculate pixel distance between previous and current centroids
            displacement = np.linalg.norm(centroid - last_positions[tracker_id])
            # Estimate speed in units of 50 pixels per second
            speed_units = (displacement * video_fps) / 50
            # Add the current speed to the buffer
            speed_buffers[tracker_id].append(speed_units)

            # Update speed display every fourth second
            if frame_counts[tracker_id] >= video_fps / 4:
                # Calculate the average speed from the buffer
                avg_speed = np.mean(speed_buffers[tracker_id])
                last_speeds[tracker_id] = f"Speed: {int(avg_speed)} fiftypx/sec"
                frame_counts[tracker_id] = 0  # Reset the frame count
            labels.append(f"{detections.data['class_name'][i]} - {last_speeds[tracker_id]}")
        else:
            labels.append(f"{detections.data['class_name'][i]} {detections.confidence[i]:.2f}")

        # Update the last known position
        last_positions[tracker_id] = centroid

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

# Process the whole video, assuming there is a way to get fps from video_info or similar
video_fps = video_info.fps  # Make sure to replace with the actual FPS from your video_info object

sv.process_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path=TARGET_VIDEO_PATH,
    callback=lambda frame, index: callback(frame, index, video_fps)
)