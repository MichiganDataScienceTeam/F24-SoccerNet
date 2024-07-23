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
        self.player_client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=api_key)
        self.player_model_id = f"{project_name}/{version_number}"
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position
    
    
    def detect_frames(self, frames):
        detections = []
        for frame in frames:
            _, encoded_image = cv2.imencode('.jpg', frame)
            base64_image = base64.b64encode(encoded_image).decode('utf-8')
            player_response = self.player_client.infer(base64_image, model_id=self.player_model_id)
            
            detections.append(player_response)
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            print("Loaded tracks from stub:", stub_path)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        #add the classes from the field lines
        }
        class_name_to_id = {
                "ball": 0,
                "player": 1,
                "referee": 2,
            #the the key values from the field lines here
            }

        for frame_num, detection in enumerate(detections):
            player_detections = detection.get('player_detection', {}).get('predictions', [])

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # Prepare for conversion to supervision format
            bboxes = []
            confidences = []
            class_ids = []
            #maybe something like this to get the points
            points = []
            points_confidences = []
            points_class_ids = []

            for pred in player_detections:
                if pred['confidence'] < .5:  # Skip low-confidence detections
                    continue

                class_name = pred['class']
                confidence = pred['confidence']

                # Log the class name for debugging
                print(f"Detected class: {class_name}")

                # Check if the class name exists in the mapping
                if class_name not in class_name_to_id:
                    print(f"Warning: Unrecognized class '{class_name}'. Skipping detection.")
                    continue

                if 'points' in pred:
                    #im thinking something kind of like this where you would grab the points or something 
                    print()
                else:
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
                    print("not points" + str(class_name_to_id[class_name]))

            # For the bounding boxes you could probably say elif points or something like that
            if bboxes:
                print("running bboxes")
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
                        print(str(frame_num))

                    if cls_names_inv[cls_id] == 'referee':
                        tracks["referees"][frame_num][track_id] = track_info

                    if cls_names_inv[cls_id] == 'ball':
                        tracks["ball"][frame_num][1] = track_info

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
            print("Saved tracks to stub:", stub_path)

        return tracks


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

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            #probably put field lines here but it could be done differently

            # Draw Players
            for track_id, player in player_dict.items():
                
                #debug statement
                print("printing players")
                frame = self.draw_ellipse(frame, player["bbox"], (0, 0, 255), track_id)

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))
            
            output_video_frames.append(frame)

        return output_video_frames

