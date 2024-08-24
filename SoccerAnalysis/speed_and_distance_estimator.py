import cv2
import sys 
import numpy as np
sys.path.append('../')
from bboxUtils import measure_distance ,get_foot_position

class SpeedAndDistance_Estimator():
    def __init__(self):
        self.frame_window=5
        self.frame_rate=30
    
    def add_speed_and_distance_to_tracks(self,tracks):
        total_distance= {}

        for object, object_tracks in tracks.items():
            if object != "players":
                continue 
            number_of_frames = len(object_tracks)
            for frame_num in range(0,number_of_frames, self.frame_window):
                last_frame = min(frame_num+self.frame_window,number_of_frames-1 )

                for track_id,_ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id]['position transformed']
                    end_position = object_tracks[last_frame][track_id]['position transformed']

                    if start_position is None or end_position is None:
                        continue
                    
                    distance_covered = measure_distance(start_position,end_position)
                    time_elapsed = (last_frame-frame_num)/self.frame_rate
                    speed_meteres_per_second = distance_covered/time_elapsed
                    speed_km_per_hour = speed_meteres_per_second*3.6
                    speed_mph = speed_km_per_hour/1.609

                    if object not in total_distance:
                        total_distance[object]= {}
                    
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    
                    total_distance[object][track_id] += distance_covered

                    for frame_num_batch in range(frame_num,last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_mph
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]
    
    def draw_speed_and_distance(self,frames,tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object != "players":
                    continue 
                for _, track_info in object_tracks[frame_num].items():
                    # Check if track_info is a dictionary
                    if isinstance(track_info, dict):
                        has_speed = "speed" in track_info
                    # Check if track_info is a structured NumPy array
                    elif isinstance(track_info, np.ndarray) and track_info.dtype.names:
                        has_speed = 'speed' in track_info.dtype.names
                    else:
                        continue

                    if has_speed:
                        if isinstance(track_info, dict):
                            speed = track_info.get('speed', None)
                            distance = track_info.get('distance', None)
                            bbox = track_info.get('bbox', None)
                        elif isinstance(track_info, np.ndarray):
                            speed = track_info['speed'] if 'speed' in track_info.dtype.names else None
                            distance = track_info['distance'] if 'distance' in track_info.dtype.names else None
                            bbox = track_info['bbox'] if 'bbox' in track_info.dtype.names else None
                        else:
                            speed, distance, bbox = None, None, None

                        if speed is None or distance is None or bbox is None:
                            continue
                #    if "speed" in track_info:
                #        speed = track_info.get('speed',None)
                #        distance = track_info.get('distance',None)
                #        if speed is None or distance is None:
                #            continue
                       
                        bbox = track_info['bbox']
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1]+=100

                        position = tuple(map(int,position))
                        cv2.putText(frame, f"{speed:.2f} mph",position,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
                        cv2.putText(frame, f"{distance:.2f} m",(position[0],position[1]+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
            output_frames.append(frame)
        
        return output_frames
