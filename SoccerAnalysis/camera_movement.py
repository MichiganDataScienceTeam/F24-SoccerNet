import pickle
import cv2
import numpy as np
import os
import sys
sys.path.append('../')


#added new stuff


class CameraMovementEstimator():
    def __init__(self, frame):
        # Minimum distance to to ignore for camera movement (too small)
        self.minimum_distance = 5

    # Loops through and adds adjusted positions with camera movements for 
    # for players, the ball, and referees
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

    # Gets the camera movements for each frame in x and y direction
    def get_camera_movement(self, tracks, frames, read_from_stub=False, stub_path=None):
        # Check if we should read from a pickle stub
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                camera_movement_per_frame = pickle.load(f)
            return camera_movement_per_frame

        camera_movement_per_frame = []

        for frame_num in range(1, len(frames)):
            movements = []

            key_points_current_frame = tracks["Key Points"][frame_num]
            key_points_previous_frame = tracks["Key Points"][frame_num - 1]

            for key, point_current in key_points_current_frame.items():
                if key == "points":
                    continue
                if key in key_points_previous_frame:
                    point_previous = key_points_previous_frame[key]

                    # Ensure both points are valid and have the correct format
                    if point_current is not None and point_previous is not None and len(point_current) == 2 and len(point_previous) == 2:
                        # Calculate the movement (difference) between the consecutive frames and append them to camera movement
                        movement_x = point_current[0] - point_previous[0]
                        movement_y = point_current[1] - point_previous[1]

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

        camera_movement_per_frame.insert(0, (0, 0))

        # Save the camera movement data to a pickle file if a stub path is provided
        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement_per_frame, f)

        return camera_movement_per_frame

    # Annotate camera movement to the video
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
