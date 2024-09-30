import numpy as np
import os
import cv2
import pickle
from roboflow import Roboflow
from assign_team import TeamAssigner
from tracker import Tracker
from line import WhiteLineDetector
from interpolator import Interpolator
from camera_movement import CameraMovementEstimator
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from intersection_finder import SuperAlgorithm
from transformer import ViewTransformer
import videoUtils

SOURCE_VIDEO_PATH = "20SecGoodVid.mov"
TARGET_VIDEO_PATH = "interpolation_test4.mp4"
PlAYER_API_KEY = ""
PLAYER_STUB_PATH = "stub_tracks_mid.pkl"
PLAYER_PROJECT_NAME = "football-players-detection-3zvbc"
VERSION_NUMBER = 1
BOXES_API_KEY = ""
BOXES_PROJECT_NAME = "football0detections"
STUB_FRAMES_PATH = "frames_stub.pkl"


# Initialize the Plyer Roboflow model and Tracker
player_rf = Roboflow(api_key=PlAYER_API_KEY)
boxes_rf = Roboflow(api_key=BOXES_API_KEY)

player_project = player_rf.workspace().project(PLAYER_PROJECT_NAME)
boxes_project = boxes_rf.workspace().project(BOXES_PROJECT_NAME)

player_model = player_project.version(VERSION_NUMBER).model
boxes_model = boxes_project.version(VERSION_NUMBER).model

print("Roboflow models initialized")

tracker = Tracker(PlAYER_API_KEY, PLAYER_PROJECT_NAME, VERSION_NUMBER, BOXES_API_KEY, BOXES_PROJECT_NAME, VERSION_NUMBER)
print("Tracker Initialized")

if os.path.exists(STUB_FRAMES_PATH):
        try:
            with open(STUB_FRAMES_PATH, 'rb') as f:
                frames = pickle.load(f)
            print("Frames loaded successfully from stub")
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Error loading frames from stub: {e}")
            frames = None
else:
        print("Reading frames from video...")
        frames = videoUtils.read_video(SOURCE_VIDEO_PATH)
        if isinstance(frames, np.ndarray):
            with open(STUB_FRAMES_PATH, 'wb') as f:
                pickle.dump(frames, f)
            print(f"Frames stored in {STUB_FRAMES_PATH}")
        else:
            print("Error: Invalid frames data. Unable to process video.")
            frames = None
    
    
tracks = tracker.get_object_tracks(frames, read_from_stub=True, stub_path="stub_path_mid.pkl")

tracker.add_position_to_tracks(tracks)
print("Adding Positions To Tracks")
    
interpolator = Interpolator()
tracks = interpolator.interpolate_missing_bboxes(tracks)
    
print("Getting Field Intersection Points")
white_line_detector = WhiteLineDetector()
tracks = white_line_detector.get_intersections(frames, tracks, 'intersections_mid.pkl', 'intersections_mid.pkl')

print("Finalizing Stationary Points")
super_algorithm = SuperAlgorithm()
tracks = super_algorithm.find_middle_intersections(tracks)
tracks = super_algorithm.find_outer_intersections(tracks)
tracks = super_algorithm.find_right_field_intersections(tracks)
tracks = super_algorithm.find_left_field_intersections(tracks)
#tracks = super_algorithms.find_left_field_intersections(tracks)

print("Initializing Camera Movement")
camera_movement_estimator = CameraMovementEstimator(frames[0])
camera_movement_per_frame = camera_movement_estimator.get_camera_movement(tracks, frames, read_from_stub=True, stub_path='camera_movement_stub.pkl')

print("Adding Camera Movement To Tracks")
camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


print("Computing Homography")
view_transformer = ViewTransformer(camera_movement_per_frame, field_length=105, field_width=68)
view_transformer.apply_homography(frames, tracks)

print("Adding Transformed Position To Tracks")
view_transformer.add_transformed_position_to_tracks(tracks)


speed_and_distance_estimator = SpeedAndDistance_Estimator()
speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

team_assigner = TeamAssigner()
team_assigner.assign_team_color(frames[0], tracks['players'][0])
for frame_num, player_track in enumerate(tracks['players']):
        for track_id, track in player_track.items():
            team = team_assigner.get_player_team(frames[frame_num], track['bbox'], track_id)
            tracks['players'][frame_num][track_id]['team'] = team
            tracks['players'][frame_num][track_id]['team_color'] = team_assigner.team_colors[team]


print("Drawing Speed and Distances")
frames = speed_and_distance_estimator.draw_speed_and_distance(frames, tracks)

print("Drawing Camera Movement")
frames = camera_movement_estimator.draw_camera_movement(frames,camera_movement_per_frame)


print("Drawing Transformed Positions")
frames = view_transformer.draw_transformed_position(frames, tracks)

print("Drawing Field Intersection Points")
frames = white_line_detector.draw_frame_intersections(frames, tracks)

print("Drawing Annotations")
annotated_frames = tracker.draw_annotations(frames, tracks)

videoUtils.save_video(annotated_frames, TARGET_VIDEO_PATH) #SWITCH TO FINAL_ANNOTATED_FRAMES
print("Video saved successfully")
print("My name is Eshwar and this is a test")
