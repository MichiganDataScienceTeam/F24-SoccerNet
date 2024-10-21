import numpy as np
import os
import cv2
import pickle
from roboflow import Roboflow
from .assign_team import TeamAssigner
from .tracker import Tracker
from .line import WhiteLineDetector
from .interpolator import Interpolator
from .camera_movement import CameraMovementEstimator
from .speed_and_distance_estimator import SpeedAndDistance_Estimator
from .intersection_finder import SuperAlgorithm
from .transformer import ViewTransformer
from . import videoUtils
from . import bboxUtils



'''
SOURCE_VIDEO_PATH = "20SecGoodVid.mov"
TARGET_VIDEO_PATH = "interpolation_test4.mp4"
PlAYER_API_KEY = ""
PLAYER_STUB_PATH = "stub_tracks_mid.pkl"
PLAYER_PROJECT_NAME = "football-players-detection-3zvbc"
VERSION_NUMBER = 1
BOXES_API_KEY = ""
BOXES_PROJECT_NAME = "football0detections"
STUB_FRAMES_PATH = "frames_stub.pkl"
'''

def process_video(source_video_path, target_video_path, player_api_key, player_stub_path,
                  player_project_name, version_number, boxes_api_key, boxes_project_name, stub_frames_path,
                  field_length=105, field_width=68):
    """
    Process a video, track player movement, and return annotated frames.

    Parameters:
    source_video_path (str): Path to the source video.
    target_video_path (str): Path to save the processed video (optional).
    player_api_key (str): API key for player detection Roboflow model.
    player_stub_path (str): Path to player stub file.
    player_project_name (str): Roboflow project name for player detection.
    version_number (int): Version number for the models.
    boxes_api_key (str): API key for boxes detection Roboflow model.
    boxes_project_name (str): Roboflow project name for box detection.
    stub_frames_path (str): Path to frames stub file.
    field_length (int): Length of the soccer field in meters.
    field_width (int): Width of the soccer field in meters.

    Returns:
    frames (list): Annotated video frames with object tracks.
    """

    # Initialize the Plyer Roboflow model and Tracker
    player_rf = Roboflow(api_key=player_api_key)
    boxes_rf = Roboflow(api_key=boxes_api_key)

    player_project = player_rf.workspace().project(player_project_name)
    boxes_project = boxes_rf.workspace().project(boxes_project_name)

    player_model = player_project.version(version_number).model
    boxes_model = boxes_project.version(version_number).model

    print("Roboflow models initialized")

    tracker = Tracker(player_api_key, player_project_name, version_number, boxes_api_key, boxes_project_name, version_number)
    print("Tracker Initialized")

    frames = videoUtils.read_video(source_video_path)
    
    tracks = tracker.get_object_tracks(frames, read_from_stub=True, stub_path=player_stub_path)

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

    print("Initializing Camera Movement")
    camera_movement_estimator = CameraMovementEstimator(frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(tracks, frames, read_from_stub=True, stub_path=stub_frames_path)

    print("Adding Camera Movement To Tracks")
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    print("Computing Homography")
    view_transformer = ViewTransformer(camera_movement_per_frame, field_length=field_length, field_width=field_width)
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
    frames = camera_movement_estimator.draw_camera_movement(frames, camera_movement_per_frame)

    print("Drawing Transformed Positions")
    frames = view_transformer.draw_transformed_position(frames, tracks)

    print("Drawing Field Intersection Points")
    frames = white_line_detector.draw_frame_intersections(frames, tracks)

    print("Drawing Annotations")
    annotated_frames = tracker.draw_annotations(frames, tracks)

    # Optionally save video to the target path
    if target_video_path:
        videoUtils.save_video(annotated_frames, target_video_path)
        print("Video saved successfully")

    return annotated_frames
