import pickle
import os
import numpy as np
import pandas as pd
import cv2
import base64
import supervision as sv
from inference_sdk import InferenceHTTPClient
import bboxUtils

class Tracker:
    def __init__(self, api_key, project_name, version_number):
        self.client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=api_key)
        self.model_id = f"{project_name}/{version_number}"
        self.tracker = sv.ByteTrack()

    def __init__(self, player_api_key, player_project_name, player_version_number, boxes_api_key, boxes_project_name, boxes_version_number, points_version_number):
        self.player_client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=player_api_key)
        self.player_model_id = f"{player_project_name}/{player_version_number}"

        self.boxes_client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=boxes_api_key)
        self.boxes_model_id = f"{boxes_project_name}/{boxes_version_number}"

        self.tracker = sv.ByteTrack()
    
    
    def preprocess_image(self, image, target_size=(640, 640)):
    # Resize the image to the target size
        self.original_size = (image.shape[1], image.shape[0])
        
        image_resized = cv2.resize(image, target_size)
        return image_resized
    
    def resize_back_to_original(self, image):
        # Resize the image back to the original size
        return cv2.resize(image, self.original_size)
    
    def add_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= bboxUtils.get_center_of_bbox(bbox)
                        tracks[object][frame_num][track_id]['position'] = position
                    elif object == 'player' or object == 'referee':
                        position = bboxUtils.get_foot_position(bbox)
                        tracks[object][frame_num][track_id]['position'] = position
        
    
    
    def detect_frames(self, frames):
        detections = []
        for frame in frames:
            _, encoded_image = cv2.imencode('.jpg', frame)
            base64_image = base64.b64encode(encoded_image).decode('utf-8')
            player_response = self.player_client.infer(base64_image, model_id=self.player_model_id)

            boxes_response = self.boxes_client.infer(base64_image, model_id=self.boxes_model_id)

            
            combined_detection = {
            'frame': frame,
            'player_detection': player_response,
            'boxes_detection': boxes_response,
            }

            detections.append(combined_detection)
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            print("loading tracks from stub...")
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            print("Loaded tracks from stub:", stub_path)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": [],
            "18Yard": [],
            "18Yard Circle": [],
            "5Yard": [],
            "First Half Central Circle": [],
            "First Half Field": [],
            "Second Half Central Circle": [], 
            "Second Half Field": [], 
            "Key Points": [],
            
        }
        class_name_to_id = {
                "ball": 0,
                "player": 1,
                "referee": 2,
                "1": 3,
                "0": 4,
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
                "6": 9,
            }
        print("reading detections...")
        for frame_num, detection in enumerate(detections):
            
            
            player_detections = detection.get('player_detection', {}).get('predictions', [])

            boxes_detections = detection.get('boxes_detection', {}).get('predictions', [])

            

            combined_detections = player_detections + boxes_detections 


            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            tracks["18Yard"].append({})
            tracks["18Yard Circle"].append({})
            tracks["5Yard"].append({})
            tracks["First Half Central Circle"].append({})
            tracks["First Half Field"].append({})
            tracks["Second Half Central Circle"].append({})
            tracks["Second Half Field"].append({})
            tracks["Key Points"].append({})

            # Prepare for conversion to supervision format
            bboxes = []
            confidences = []
            class_ids = []

            for pred in combined_detections:
                if pred['confidence'] < .5 and pred['class'] != "ball":
                    continue  # Skip low-confidence detections

                class_name = pred['class']
                confidence = pred['confidence']

                # Log the class name for debugging
                print(f"Detected class: {class_name}")

                # Check if the class name exists in the mapping
                if class_name not in class_name_to_id:
                    print(f"Warning: Unrecognized class '{class_name}'. Skipping detection.")
                    continue
                
                x_center = pred['x']
                y_center = pred['y']
                width = pred['width']
                height = pred['height']
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                bboxes.append([x1, y1, x2, y2])
                confidences.append(confidence)
                class_ids.append(class_name_to_id[class_name])

            # Handle bounding box detections
            if bboxes:
                detection_supervision = sv.Detections(
                    xyxy=np.array(bboxes),
                    confidence=np.array(confidences),
                    class_id=np.array(class_ids)
                )

                cls_names_inv = {v: k for k, v in class_name_to_id.items()}

                detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

                for frame_detection in detection_with_tracks:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]
                    track_id = frame_detection[4]

                    track_info = {"bbox": bbox}

                    if cls_names_inv[cls_id] == 'player':
                        tracks["players"][frame_num][track_id] = track_info

                    if cls_names_inv[cls_id] == 'referee':
                        tracks["referees"][frame_num][track_id] = track_info

                    if cls_names_inv[cls_id] == 'ball':
                        tracks["ball"][frame_num][1] = track_info

                    if cls_names_inv[cls_id] == '1':
                        tracks["18Yard"][frame_num][cls_id] = track_info

                    if cls_names_inv[cls_id] == '0':
                        tracks["18Yard Circle"][frame_num][cls_id] = track_info

                    if cls_names_inv[cls_id] == '2':
                        tracks["5Yard"][frame_num][cls_id] = track_info

                    if cls_names_inv[cls_id] == '3':
                        tracks["First Half Central Circle"][frame_num][cls_id] = track_info

                    if cls_names_inv[cls_id] == '4':
                        tracks["First Half Field"][frame_num][cls_id] = track_info

                    if cls_names_inv[cls_id] == '5':
                        tracks["Second Half Central Circle"][frame_num][cls_id] = track_info

                    if cls_names_inv[cls_id] == '6':
                        tracks["Second Half Field"][frame_num][cls_id] = track_info


        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
            print("Saved tracks to stub:", stub_path)

        return tracks
    def draw_rectangle(self, frame, bbox, color=(255, 0, 0), thickness=2, label=None):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        if label:
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = bboxUtils.get_center_of_bbox(bbox)
        width = bboxUtils.get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(frame, f"{track_id}", (int(x1_text), int(y1_rect + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame
    
        

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = bboxUtils.get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame
    '''
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame
    
    def scale_points(self, points):
            scale_x = self.original_size[0] / 640 #self.target_size[0]
            scale_y = self.original_size[1] / 640 #self.target_size[1]
            return [{'x': int(point['x'] * scale_x), 'y': int(point['y'] * scale_y)} for point in points]
    '''
    def scale_points(tracker, points, original_size, target_size, x_scale, y_scale):
        #print(f"tracker: {tracker}")
        #print(f"points: {points}")
        #print(f"Original size: {original_size}")
        #print(f"Target size: {target_size}")
        if isinstance(original_size, tuple) and len(original_size) == 2:
            original_width, original_height = original_size
        else:
            raise ValueError("original_size should be a tuple with two elements (width, height)")

        if isinstance(target_size, tuple) and len(target_size) == 2:
            target_width, target_height = target_size
        else:
            raise ValueError("target_size should be a tuple with two elements (width, height)")
 
        scale_x = original_width / x_scale  #890
        scale_y = original_height / y_scale #1320

        

        scaled_points = []
        for point in points:
            scaled_x = int(point['x'] * scale_x)
            scaled_y = int(point['y'] * scale_y)
            scaled_points.append({'x': scaled_x, 'y': scaled_y})

        return scaled_points

        

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            original_size = (frame.shape[1], frame.shape[0])

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            eighteen_yard_dict = tracks["18Yard"][frame_num]
            five_yard_dict = tracks["5Yard"][frame_num]
            first_half_central_circle_dict = tracks["First Half Central Circle"][frame_num]
            first_half_field_dict = tracks["First Half Field"][frame_num]
            second_half_central_circle_dict = tracks["Second Half Central Circle"][frame_num]
            second_half_field_dict = tracks["Second Half Field"][frame_num]
            eighteen_yard_circle_dict = tracks["18Yard Circle"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                #if player.get('has_ball', False):
                    #frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))
            
            
            for track_id, field_box in eighteen_yard_dict.items():
                frame = self.draw_rectangle(frame, field_box["bbox"], (255, 0, 0), 2, "eighteen_yard")

            for track_id, field_box in five_yard_dict.items():
                frame = self.draw_rectangle(frame, field_box["bbox"], (255, 0, 0), 2, "five_yard")

            for track_id, field_box in first_half_central_circle_dict.items():
                frame = self.draw_rectangle(frame, field_box["bbox"], (0, 0, 255), 2, "first_half_central_circle")

            for track_id, field_box in first_half_field_dict.items():
                frame = self.draw_rectangle(frame, field_box["bbox"], (0, 0, 255), 2, "first_half_field")

            for track_id, field_box in second_half_central_circle_dict.items():
                frame = self.draw_rectangle(frame, field_box["bbox"], (0, 0, 255), 2, "second_central_circle")
            
            for track_id, field_box in second_half_field_dict.items():
                frame = self.draw_rectangle(frame, field_box["bbox"], (0, 0, 255), 2, "second_half_field")

            for track_id, circle in eighteen_yard_circle_dict.items():
                frame = self.draw_rectangle(frame, circle["bbox"], (255, 0, 0), 2, "eighteen_yard_circle")
            

            # Draw field boxes and circles
            
            # Draw Team Ball Control
            #frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            
            output_video_frames.append(frame)

        return output_video_frames




# Usage example:
# tracker = Tracker(api_key='your_api_key', project_name='your_project_name', version_number='your_version_number')
# tracks = tracker.get_object_tracks(frames, read_from_stub=True, stub_path=STUB_PATH)
# annotated_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
