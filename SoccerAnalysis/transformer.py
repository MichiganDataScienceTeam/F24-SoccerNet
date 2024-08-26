import numpy as np 
import cv2


class ViewTransformer():
    def __init__(self, camera_movement, field_length, field_width):
        self.key_points = ["Right Top 18Yard Point", "Right Top 18Yard Circle Point", "Right Bottom 18Yard Circle Point", 
                  "Right Top 5Yard Point", "Right Bottom 18Yard Point", "Top Circle Point", "Bottom Circle Point", 
                  "Left Circle Point", "Right Circle Point", "Center Circle Point", "Top Center Point", "Bottom Center Point"]
        
        self.field_coords = {
            "Top Left Corner": (0, field_width),
            "Bottom Left Corner": (0, 0),
            "Top Right Corner": (field_length, field_width),
            "Bottom Right Corner": (field_length, 0),
            "Center Circle Point": (field_length/2, field_width/2),
            "Left Circle Point": (field_length/2 - 9.14, field_width/2),
            "Right Circle Point": (field_length/2 + 9.14, field_width/2), 
            "Top Circle Point": (field_length/2, field_width/2 + 9.14),
            "Bottom Circle Point": (field_length/2, field_width/2 - 9.14),
            "Right Top 18Yard Point": (field_length - 16.459, field_width/2 + 20.168), 
            "Right Top 18Yard Circle Point": (field_length - 16.459, field_width/2 + 7.343), 
            "Right Top 5Yard Point" :(field_length - 4.572, field_width/2 + 9.144) , 
            "Right Bottom 18Yard Point": (field_length - 16.459, field_width/2 - 20.168),
            "Right Bottom 18Yard Circle Point": (field_length - 16.459, field_width/2 - 7.343), 
            "Top Center Point": (field_length/2 , field_width),
            "Bottom Center Point": (field_length/2, 0)
        }
        self.homographies = {}
        self.camera_movement = camera_movement

    # Extracts the key points from the tracks 
    def extract_points_from_tracks(self, tracks, frame_num):
        seen_points = []
        field_points = []
        for key_point in self.key_points:
            if key_point in tracks["Key Points"][frame_num] and key_point in self.field_coords:
                point = tracks["Key Points"][frame_num][key_point]
                point_in_field = self.field_coords[key_point]
                if point is not None:
                    seen_points.append([point[0], point[1]])
                    field_points.append([point_in_field[0], point_in_field[1]])
        return (np.array(seen_points, dtype=np.float32), np.array(field_points, dtype=np.float32))

    # applies a homography throughout all the frames
    def apply_homography(self, frames, tracks, read_from_stub=False, stub_path=None):
        found_center = False
        for frame_num in range(len(frames)):
            seen_points, field_points = self.extract_points_from_tracks(tracks, frame_num)
            if seen_points.size == 0:
                found_center = False
                continue 


            if ("Left Circle Point" in tracks["Key Points"][frame_num] and 
            "Right Circle Point" in tracks["Key Points"][frame_num] and 
            tracks["Key Points"][frame_num]["Left Circle Point"] is not None and 
            tracks["Key Points"][frame_num]["Right Circle Point"] is not None):
                if field_points.size == 0 or seen_points.size == 0:
                    continue

                
                if len(seen_points) >= 4 and len(field_points) >= 4:
                    M, mask = cv2.findHomography(seen_points, field_points)
                    self.homographies[frame_num] = M

                found_center = True


            else:
                if found_center == True:
                    player_positions = []

                    if field_points.size == 0 or seen_points.size == 0:
                        continue
                    
                    if frame_num > 0:  # Ensure there is a previous frame
                        prev_frame_num = frame_num - 1
                        
                        # Extract player positions from the previous frame
                        for track_id, track_info in tracks["players"][prev_frame_num].items():
                            position = track_info.get('position')
                            if position:
                                player_positions.append(np.array(position))

                        #get the camera movement for this frame
                        current_camera_movement = np.array(self.camera_movement[frame_num])

                        #add the position from the camera movement to simulate those position to this frame
                        adjusted_positions_to_current_frame = [position + current_camera_movement for position in player_positions]

                    
                        homography_player_positions = []
                        for position in adjusted_positions_to_current_frame:
                            new_point = self.transform_point(position, frame_num-1)
                            homography_player_positions.append(new_point)


                        if homography_player_positions and seen_points.size != 0:
                            seen_points = np.vstack((seen_points, np.array(adjusted_positions_to_current_frame)))
                            homography_player_positions_array = np.array(homography_player_positions)
    
                            # Ensure it has the correct shape
                            if homography_player_positions_array.ndim == 1:
                                homography_player_positions_array = homography_player_positions_array.reshape(-1, 2)
                            field_points = np.vstack((field_points, np.array(homography_player_positions)))


                    #append all the player positions to seen points and all the homographies to field points
                    if len(seen_points) >= 4 and len(field_points) >= 4:
                        M, mask = cv2.findHomography(seen_points, field_points)
                        self.homographies[frame_num] = M
                
                #use those to do a homography on this frame. then move on

    # Use homography matrix to transform a point
    def transform_point(self, point, frame_num):
        if frame_num in self.homographies:
            M = self.homographies[frame_num]
            if M is not None:
                point_homogeneous = np.append(point, 1)  # Convert to homogeneous coordinates
                transformed_point = np.dot(M, point_homogeneous)
                transformed_point /= transformed_point[2]  # Normalize to make the last component 1
                return transformed_point[:2]
        return None
        
    # Add all transformed positions to tracks 
    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                        if object == 'ball' or object == 'referees' or object == 'players':
                            position = track_info['position']
                            position = np.array(position)
                            position_transformed = self.transform_point(position, frame_num)
                            if position_transformed is not None:
                                position_transformed = tuple(position_transformed.squeeze().tolist())
                            tracks[object][frame_num][track_id]['position transformed'] = position_transformed
                
                for key_point in self.key_points:
                    if key_point in tracks["Key Points"][frame_num]:
                        point = tracks["Key Points"][frame_num][key_point]
                        point = np.array(point)
                        point_transformed = self.transform_point(point, frame_num)
                        if point_transformed is not None:
                            point_transformed = tuple(point_transformed.squeeze().tolist())

                        tracks["Key Points"][frame_num][key_point + ' Transformed'] = point_transformed

    # draw transformed positions to video
    def draw_transformed_position(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            
            # Draw transformed positions of the object's points
            for object, object_tracks in tracks.items():
                if object == "Key Points":
                    continue
                if frame_num >= len(object_tracks):
                    continue
                for track_id, track_info in object_tracks[frame_num].items():
                    if 'position transformed' in track_info:
                        position = track_info['position']
                        position_transformed = track_info['position transformed']
                        if position_transformed is not None:
                            text = f'{position_transformed[0]:.2f}, {position_transformed[1]:.2f}'
                            text_position = (int(position[0]), int(position[1] + 50))
                            cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

                # Draw transformed positions of the key points
                for key_point in self.key_points:
                    transformed_key = key_point + ' Transformed'
                    if transformed_key in tracks["Key Points"][frame_num]:
                        point = tracks["Key Points"][frame_num][key_point]
                        point_transformed = tracks["Key Points"][frame_num][transformed_key]
                        if point_transformed is not None:
                            text = f'{point_transformed[0]:.2f}, {point_transformed[1]:.2f}'
                            text_position = (int(point[0]), int(point[1] + 50))
                            cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
                            
            output_frames.append(frame)

        return output_frames
