import numpy as np
import pandas as pd
import bboxUtils

class Interpolator:
    def __init__(self):
        self.first_half_series = None
        self.second_half_series = None
        self.checked_half_1 = False
        self.checked_half_2 = False

    # Interpolate necessary intersection points (main driver)
    def interpolate_missing_bboxes(self, tracks, max_missing_frames=15, size_change_threshold=0.5):
        classes_to_exclude = {"players", "referees", "ball", "Key Points"}
        zero_frames = self.find_zero_frames(tracks)

        self.check_18yard_circle(tracks)

        for object_type, object_tracks in tracks.items():
            if object_type in classes_to_exclude:
                continue

            for track_id in self.get_all_track_ids(object_tracks):
                self.process_tracks(object_tracks, track_id, size_change_threshold, max_missing_frames, zero_frames, tracks)
        
        return tracks
    

    # This is the driver for this class and will interpolate detections accordingly 
    def process_tracks(self, object_tracks, track_id, size_change_threshold, max_missing_frames, zero_frames, tracks):
        
        # Collect bounding boxes for this track ID across all frames
        bbox_series = [frame_data.get(track_id, {}).get('bbox', None) for frame_data in object_tracks]

        # Filter out None values for conversion to DataFrame
        bbox_series_filtered = [bbox if bbox is not None else [np.nan, np.nan, np.nan, np.nan] for bbox in bbox_series]

        # Convert bbox_series to a DataFrame
        df_bbox_series = pd.DataFrame(bbox_series_filtered, columns=['x1', 'y1', 'x2', 'y2'])

        
        # Handle dramatic size changes
        df_bbox_series = self.handle_dramatic_size_changes(df_bbox_series, size_change_threshold)
        
            # Interpolate missing bounding boxes (up to max_missing_frames)
        df_bbox_series = self.interpolate_within_limits(df_bbox_series, max_missing_frames)

        
        #Debug
        #print(track_id)

        if track_id == 7:
            self.first_half_series = df_bbox_series
            self.checked_half_1 = True

        if track_id == 9:
            self.second_half_series = df_bbox_series
            self.checked_half_2 = True
        
        if self.first_half_series is not None and self.second_half_series is not None and self.checked_half_1 and self.checked_half_2:
            self.interpolate_half_field(object_tracks, track_id, self.first_half_series, self.second_half_series, zero_frames)
            return

        # Convert DataFrame back to list of bounding boxes
        bbox_series = df_bbox_series.to_numpy().tolist()
        
        # Debug: Print interpolated bounding boxes
        # print(f"Track ID {track_id} interpolated bounding boxes: {bbox_series}")
        # Update the original tracks with the interpolated bounding boxes
        for frame_idx, bbox in enumerate(bbox_series):
            if not all(np.isnan(bbox)):
                if track_id in object_tracks[frame_idx]:
                    object_tracks[frame_idx][track_id]['bbox'] = bbox
                else:
                    object_tracks[frame_idx][track_id] = {'bbox': bbox}

    def interpolate_half_field(self, object_tracks, track_id, df_first_half, df_second_half, zero_frames):

        # Interpolate normally where both are present
        df_first_half, df_second_half = self.fill_missing_half_fields(df_first_half, df_second_half, zero_frames)

        # Convert DataFrame back to list of bounding boxes
        first_half_series = df_first_half.to_numpy().tolist()
        second_half_series = df_second_half.to_numpy().tolist()

        for frame_idx, bbox in enumerate(first_half_series):
            if not all(np.isnan(bbox)):
                if track_id in object_tracks[frame_idx]:
                        object_tracks[frame_idx][track_id]['bbox'] = bbox
                else:
                        object_tracks[frame_idx][track_id] = {'bbox': bbox}

        for frame_idx, bbox in enumerate(second_half_series):
            if not all(np.isnan(bbox)):
                if track_id in object_tracks[frame_idx]:
                        object_tracks[frame_idx][track_id]['bbox'] = bbox
                else:
                        object_tracks[frame_idx][track_id] = {'bbox': bbox}


    #Fill in all gaps for half field since we need them to always show
    def fill_missing_half_fields(self, df_first_half, df_second_half, zero_frames):

        # Find gaps in the data for both half fields
        nan_groups_first = df_first_half.isna().all(axis=1).astype(int).groupby(df_first_half.notna().all(axis=1).cumsum()).cumsum()
        nan_groups_second = df_second_half.isna().all(axis=1).astype(int).groupby(df_second_half.notna().all(axis=1).cumsum()).cumsum()


        # Find gaps in both dataframes
        first_gaps = self.find_nan_gaps_no_lim(nan_groups_first, zero_frames)
        second_gaps = self.find_nan_gaps_no_lim(nan_groups_second, zero_frames)

        first_idx, second_idx = 0, 0

        while first_idx < len(first_gaps) and second_idx < len(second_gaps):
            start_first, end_first = first_gaps[first_idx]
            start_second, end_second = second_gaps[second_idx]

            if end_first <= start_second:
                # First gap ends before the second gap starts
                self.interpolate_forward(df_first_half, start_first, end_first)
                first_idx += 1
            elif end_second <= start_first:
                # Second gap ends before the first gap starts
                self.interpolate_forward(df_second_half, start_second, end_second)
                second_idx += 1
            else:
                # Overlapping gaps
                if (end_first - start_first) <= (end_second - start_second):

                    self.interpolate_forward(df_first_half, start_first, end_first)
                    first_idx += 1
                else:                   
                    self.interpolate_forward(df_second_half, start_second, end_second)
                    second_idx += 1

        while first_idx < len(first_gaps):
            start_first, end_first = first_gaps[first_idx]
            self.interpolate_forward(df_first_half, start_first, end_first)
            first_idx += 1

        while second_idx < len(second_gaps):
            start_second, end_second = second_gaps[second_idx]
            self.interpolate_forward(df_second_half, start_second, end_second)
            second_idx += 1

        return df_first_half, df_second_half


    # Check where there are no detections in a frame (like when the camera pans somewhere else)
    def find_zero_frames(self, tracks):
        classes_to_exclude = {"players", "referees", "ball", "Key Points"}

        zero_detection_frames = []
        total_frames = len(next(iter(tracks.values())))

        start_frame = None
        for frame_idx in range(total_frames):
            has_detection = False
            for object_type, object_tracks in tracks.items():
                if object_type in classes_to_exclude:
                    continue
                
                # Check if any bbox in the current frame for the object type is valid
                if any(self.is_valid_bbox(track['bbox']) for track in object_tracks[frame_idx].values()):
                    has_detection = True
                    break
            
            if not has_detection:
                if start_frame is None:
                    start_frame = frame_idx
            else:
                if start_frame is not None:
                    zero_detection_frames.append((start_frame, frame_idx - 1))
                    start_frame = None

        if start_frame is not None:
            zero_detection_frames.append((start_frame, total_frames - 1))
        
        return zero_detection_frames

    # Check if bbox exists and contains valid coordinates (not nan)
    def is_valid_bbox(self, bbox):
        return bbox is not None and not any(np.isnan(coord) for coord in bbox)

    # interpolate ahead until you reach the end_idx
    def interpolate_forward(self, df, start_idx, end_idx):
    
        if start_idx > 0 and end_idx < len(df):
            interpolated_values = df.iloc[start_idx - 1:end_idx + 1].interpolate()
            df.iloc[start_idx:end_idx] = interpolated_values.iloc[1:end_idx - start_idx + 1].values
        elif start_idx == 0 and end_idx < len(df):
            df.iloc[start_idx:end_idx] = df.iloc[end_idx]
        elif start_idx > 0 and end_idx == len(df):
            df.iloc[start_idx:end_idx] = df.iloc[start_idx - 1]

    #find all the nan_groups using where there are zero detections (don't interpolate)
    def find_nan_gaps_no_lim(self, nan_groups, zero_frames):
        gaps = []
        current_start = None
        for idx, val in nan_groups.items():
            if val == 1 and current_start is None:
                current_start = idx
            elif val == 0 and current_start is not None:
                gap = (current_start, idx)
                if not self.is_in_zero_frames(gap, zero_frames):
                    gaps.append(gap)
                current_start = None
        if current_start is not None:
            gap = (current_start, len(nan_groups))
            if not self.is_in_zero_frames(gap, zero_frames):
                gaps.append(gap)
        return gaps      

    # Checks if a gap is in a zero detections frame
    def is_in_zero_frames(self, gap, zero_frames):
        start, end = gap
        for zero_start, zero_end in zero_frames:
            if start+1 >= zero_start and end-1 <= zero_end:
                return True
        return False

    # Returns all track_ids
    def get_all_track_ids(self, object_tracks):
        track_ids = set()
        for frame_data in object_tracks:
            track_ids.update(frame_data.keys())
        return track_ids

    '''
    def preprocess_zero_detection_frames(self, tracks):
        zero_detection_frames = set()
        total_frames = len(next(iter(tracks.values())))

        for frame_idx in range(total_frames):
            if not any(
                self.frame_has_detections(object_tracks, frame_idx)
                for object_tracks in tracks.values()
            ):
                zero_detection_frames.add(frame_idx)
        return zero_detection_frames
    '''
    # Interpolates within max_missing_frames limit
    def interpolate_within_limits(self, df, max_missing_frames):
        # Find gaps in the data
        nan_groups = df.isna().all(axis=1).astype(int).groupby(df.notna().all(axis=1).cumsum()).cumsum()

        # Only interpolate gaps within the specified limit
        df_interpolated = df.copy()
        for start, end in self.find_nan_gaps(nan_groups, max_missing_frames):
            df_interpolated.iloc[start-1:end+1] = df.iloc[start-1:end+1].interpolate()
        return df_interpolated
    
    # Find frames where there are nans in the dataframe 
    def find_nan_gaps(self, nan_groups, max_missing_frames):
        gaps = []
        current_start = None
        for idx, val in nan_groups.items():
            if val == 1 and current_start is None:
                current_start = idx
            elif val == 0 and current_start is not None:
                if idx - current_start <= max_missing_frames:
                    gaps.append((current_start, idx))
                current_start = None
        return gaps

    # Sometimes detections will go haywire and change sizes for no reason.
    # This will find those changes and set them to nan
    def handle_dramatic_size_changes(self, df, threshold):
        for i in range(6, len(df) - 6):
            minus6 = df.iloc[i - 6]
            plus1 = df.iloc[i + 1]
            minus1 = df.iloc[i - 1]
            normal = df.iloc[i]
            plus6 = df.iloc[i + 6]
            minus2 = df.iloc[i - 2]
            plus2 = df.iloc[i+2]

            if self.is_consecutive_size_change(minus2, minus1, normal, plus1, plus2, threshold):
                df.iloc[i] = [np.nan, np.nan, np.nan, np.nan]
                df.iloc[i-1] = [np.nan, np.nan, np.nan, np.nan]
                df.iloc[i-2] = [np.nan, np.nan, np.nan, np.nan]
                df.iloc[i+1] = [np.nan, np.nan, np.nan, np.nan]
                df.iloc[i+2] = [np.nan, np.nan, np.nan, np.nan]


        return df

    '''
    def is_dramatic_difference(self, bbox1, bbox5, bbox6, bbox7, bbox12, threshold):
        size1 = (bbox1['x2'] - bbox1['x1']) * (bbox1['y2'] - bbox1['y1'])
        size5 = (bbox5['x2'] - bbox5['x1']) * (bbox5['y2'] - bbox5['y1'])
        size6 = (bbox6['x2'] - bbox6['x1']) * (bbox6['y2'] - bbox6['y1'])
        size7 = (bbox7['x2'] - bbox7['x1']) * (bbox7['y2'] - bbox7['y1'])
        size12 = (bbox12['x2'] - bbox12['x1']) * (bbox12['y2'] - bbox12['y1'])

        size_change1_to_5 = abs(size5 - size1) / size1 if size1 > 0 else 0
        size_change1_to_6 = abs(size6 - size1) / size1 if size1 > 0 else 0
        size_change1_to_7 = abs(size7 - size1) / size1 if size1 > 0 else 0
        size_change5_to_12 = abs(size12 - size5) / size5 if size5 > 0 else 0
        size_change6_to_12 = abs(size12 - size6) / size6 if size6 > 0 else 0
        size_change7_to_12 = abs(size12 - size7) / size7 if size7 > 0 else 0
        size_change1_to_12 = abs(size12 - size1) / size1 if size1 > 0 else 0

        return (size_change1_to_5 > threshold or size_change1_to_6 > threshold or size_change1_to_7 > threshold or
            size_change5_to_12 > threshold or size_change6_to_12 > threshold or size_change7_to_12 > threshold) and size_change1_to_12 < threshold
    '''

    # Checks for consecutive size changes 
    def is_consecutive_size_change(self, prev2_bbox, prev_bbox, curr_bbox, next_bbox, next2_bbox, threshold):
        size1 = (prev2_bbox['x2'] - prev2_bbox['x1']) * (prev2_bbox['y2'] - prev2_bbox['y1'])
        size2 = (prev_bbox['x2'] - prev_bbox['x1']) * (prev_bbox['y2'] - prev_bbox['y1'])
        size3 = (curr_bbox['x2'] - curr_bbox['x1']) * (curr_bbox['y2'] - curr_bbox['y1'])
        size4 = (next_bbox['x2'] - next_bbox['x1']) * (next_bbox['y2'] - next_bbox['y1'])
        size5 = (next2_bbox['x2'] - next2_bbox['x1']) * (next2_bbox['y2'] - next2_bbox['y1'])

        size_change1 = abs(size2 - size1) / size1 if size1 > 0 else 0
        size_change2 = abs(size3 - size2) / size2 if size2 > 0 else 0
        size_change3 = abs(size4 - size3) / size3 if size3 > 0 else 0
        size_change4 = abs(size5 - size4) / size4 if size4 > 0 else 0

        return size_change1 > threshold or size_change2 > threshold or size_change3 > threshold or size_change4 > threshold
    
    # 18 yard circle for our model specifically goes haywire so we interpolate that 
    def check_18yard_circle(self, tracks):
        for frame_idx, frame_data in enumerate(tracks["18Yard Circle"]):
            for track_id, bbox in frame_data.items():
                if bbox and self.is_circle_too_large(bbox["bbox"], tracks["18Yard"][frame_idx]):
                    tracks["18Yard Circle"][frame_idx][track_id]['bbox'] = None

    # checks if 18yard circle is too large 
    def is_circle_too_large(self, circle_bbox, yard_bbox_list):
        if not bboxUtils.is_bbox_not_none(circle_bbox):
            return False

        circle_size = bboxUtils.calculate_bbox_size(circle_bbox)
        for yard_bbox in yard_bbox_list.values():
            if bboxUtils.is_bbox_not_none(yard_bbox["bbox"]):
                yard_size = bboxUtils.calculate_bbox_size(yard_bbox["bbox"])
                if circle_size >= 0.75 * yard_size:
                    return True
        return False
    
