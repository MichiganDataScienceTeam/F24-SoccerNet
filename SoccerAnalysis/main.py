import numpy as np
import cv2
import pickle
from roboflow import Roboflow
from tracker import Tracker
from line import WhiteLineDetector
from interpolator import Interpolator
from assign_team import TeamAssigner
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import videoUtils

SOURCE_VIDEO_PATH = "shortTestVid.mov"
TARGET_VIDEO_PATH = "middle_vid.mp4"
PlAYER_API_KEY = ""
PLAYER_STUB_PATH = "stub_tracks_mid.pkl"
PLAYER_PROJECT_NAME = "football-players-detection-3zvbc"
VERSION_NUMBER = 1
BOXES_API_KEY = ""
BOXES_PROJECT_NAME = "football0detections"


# Initialize the Plyer Roboflow model and Tracker
player_rf = Roboflow(api_key=PlAYER_API_KEY)
boxes_rf = Roboflow(api_key=BOXES_API_KEY)


player_project = player_rf.workspace().project(PLAYER_PROJECT_NAME)
boxes_project = boxes_rf.workspace().project(BOXES_PROJECT_NAME)

player_model = player_project.version(VERSION_NUMBER).model
boxes_model = boxes_project.version(VERSION_NUMBER).model

print("Roboflow models initialized")


# Create instances
tracker = Tracker(PlAYER_API_KEY, PLAYER_PROJECT_NAME, VERSION_NUMBER, BOXES_API_KEY, BOXES_PROJECT_NAME, VERSION_NUMBER, VERSION_NUMBER)
print("Tracker Initialized")

# Read frames from the video
frames = videoUtils.read_video(SOURCE_VIDEO_PATH)
print("Reading frames")


# Check if frames are a numpy array
if isinstance(frames, np.ndarray):
    
    # Get object tracks
    tracks = tracker.get_object_tracks(frames, read_from_stub=True, stub_path="stub_path.pkl")
    tracker.add_position_to_tracks(tracks)
    print("Adding position to tracks")

    # Estimate Speed and Distance
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams 
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0], tracks['players'][0])
    for frame_num, player_track in enumerate(tracks['players']):
          for track_id, track in player_track.items():
                team = team_assigner.get_player_team(frames[frame_num], track['bbox'], track_id)
                tracks['players'][frame_num][track_id]['team'] = team
                tracks['players'][frame_num][track_id]['team_color'] = team_assigner.team_colors[team]

    

    print("Getting field intersection points")
    white_line_detector = WhiteLineDetector()
    tracks = white_line_detector.get_intersections(frames, tracks, 'intersections.pkl', 'intersections.pkl')

    #THIS DOESNT WORK I GOTTA WORK ON THIS 
    #interpolator = Interpolator()
    #tracks = interpolator.interpolate_missing_bboxes(tracks)
    #tracks = interpolator.clear_and_fix_unwanted_intersections(tracks)
    #tracks = interpolator.interpolate_intersections(tracks)
    #tracks = interpolator.smooth_bboxes(tracks)
    
    print("drawing field intersection points")
    frames = white_line_detector.draw_frame_intersections(frames, tracks)

    # Initialize camera movement estimator
    #camera_movement_estimator = CameraMovementEstimator(frame=frames[0])
    #camera_movement = camera_movement_estimator.get_camera_movement(frames, tracks, read_from_stub=True, stub_path="camera_movement.pkl")

    #print("Camera movement estimator initialized")

    # Adjust positions based on camera movement
    #camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement)
    
    # Transform positions to real-world coordinates
    #view_transformer = ViewTransformer()
    #view_transformer.add_transformed_position_to_tracks(tracks)
    #print("Transforming positions and adjustments to real-world coordinates")
    print("Drawing Annotations")
    annotated_frames = tracker.draw_annotations(frames, tracks)

    # Draw Speed and Distance
    annotated_frames = speed_and_distance_estimator.draw_speed_and_distance(annotated_frames, tracks)

    # Draw camera movement annotations
    #print("Drawing camera movement annotations")
    #final_annotated_frames = camera_movement_estimator.draw_camera_movement(annotated_frames, camera_movement)
    # Save the output video


    videoUtils.save_video(annotated_frames, TARGET_VIDEO_PATH) #SWITCH TO FINAL_ANNOTATED_FRAMES
    print("Video saved successfully")
else:
        print("Error: Invalid frames data. Unable to process video.")
