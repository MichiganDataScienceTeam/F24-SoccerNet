import pickle
import cv2
import numpy as np
import os
import sys
sys.path.append('../')
from bboxUtils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self, frame):
        self.minimum_distance = 5
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    if object in ['ball', 'referees', 'players']:
                        position = track_info['position']
                        position = np.array(position)
                        camera_movement = np.array(camera_movement_per_frame[frame_num])
                        position_adjusted = position - camera_movement

                        if position_adjusted is not None:
                            position_adjusted = tuple(position_adjusted.squeeze().tolist())
                        
                        tracks[object][frame_num][track_id]['position adjusted'] = position_adjusted
            '''
            for key_point in tracks["Key Points"][frame_num]:
                if key_point != "points":  # Skip the "points" key
                    point = tracks["Key Points"][frame_num][key_point]
                    point = np.array(point)
                    camera_movement = np.array(camera_movement_per_frame[frame_num])
                    point_adjusted = point - camera_movement

                    if point_adjusted is not None:
                        point_adjusted = tuple(point_adjusted.squeeze().tolist())

                    tracks["Key Points"][frame_num][key_point + ' Adjusted'] = point_adjusted
            '''
    def get_camera_movement(self, tracks, frames, read_from_stub=False, stub_path=None):
        # Check if we should read from a pickle stub
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                camera_movement_per_frame = pickle.load(f)
            return camera_movement_per_frame

        # Initialize a list to hold the camera movement for each frame
        camera_movement_per_frame = []

        # Iterate through the frames
        for frame_num in range(1, len(frames)):
            movements = []

            # Get the key points from the current frame and the previous frame
            key_points_current_frame = tracks["Key Points"][frame_num]
            key_points_previous_frame = tracks["Key Points"][frame_num - 1]

            # Iterate through all key points in the current frame
            for key, point_current in key_points_current_frame.items():
                if key == "points":
                    continue
            # Check if the same key point exists in the previous frame
                if key in key_points_previous_frame:
                    point_previous = key_points_previous_frame[key]

                    # Ensure both points are valid and have the correct format
                    if point_current is not None and point_previous is not None and len(point_current) == 2 and len(point_previous) == 2:
                        # Calculate the movement (difference) between the consecutive frames
                        print(point_current[0])
                        print(point_previous[0])
                        movement_x = point_current[0] - point_previous[0]
                        movement_y = point_current[1] - point_previous[1]

                        # Append the movement to the list
                        if movement_x < self.minimum_distance:
                            movement_x = 0
                        if movement_y < self.minimum_distance:
                            movement_y = 0
                        movements.append((movement_x, movement_y))

            # Calculate the average movement for the frame
            if movements:
                avg_movement_x = sum([m[0] for m in movements]) / len(movements)
                avg_movement_y = sum([m[1] for m in movements]) / len(movements)
                camera_movement_per_frame.append((avg_movement_x, avg_movement_y))
            else:
                # If no movement could be calculated, assume no movement
                camera_movement_per_frame.append((0, 0))

        # Prepend a (0, 0) movement for the first frame since there's no previous frame to compare with
        camera_movement_per_frame.insert(0, (0, 0))

        # Save the camera movement data to a pickle file if a stub path is provided
        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement_per_frame, f)

        return camera_movement_per_frame


    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            frame = cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            output_frames.append(frame)

        return output_frames
