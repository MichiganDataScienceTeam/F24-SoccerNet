import numpy as np

class Interpolator:
    def __init__(self):
        pass

    def interpolate_missing_bboxes(self, tracks, max_missing_frames=15):
        for obj_class in tracks.keys():
            if obj_class in ["players", "referees", "ball"]:
                continue
            frames = tracks[obj_class]
            for track_id in self.get_all_track_ids(frames):
                self.interpolate_track(tracks, track_id, obj_class, max_missing_frames)
        return tracks

    def get_all_track_ids(self, frames):
        track_ids = set()
        for frame_data in frames:
            for track_id in frame_data.keys():
                track_ids.add(track_id)
        return track_ids

    def interpolate_track(self, tracks, track_id, obj_class, max_missing_frames):
        frames = tracks[obj_class]
        
        for frame_num in range(len(frames)):
            if track_id in frames[frame_num] and 'bbox' in frames[frame_num][track_id]:
                continue

            missing_count = 0
            last_valid_bbox = None

            # Special handling for field bounding boxes
            if obj_class in ["First Half Field", "Second Half Field"]:
                other_half_class = "Second Half Field" if obj_class == "First Half Field" else "First Half Field"

                # If the other half is detected, interpolate normally
                if any('bbox' in frames[frame_num] for frames in tracks[other_half_class]):
                    self.interpolate_bbox_in_half(frames, track_id, frame_num, max_missing_frames)
                else:
                    # If only one half is detected, fill in the missing frames until the next valid detection
                    for next_frame_num in range(frame_num + 1, len(frames)):
                        if track_id in frames[next_frame_num] and 'bbox' in frames[next_frame_num][track_id]:
                            next_valid_bbox = frames[next_frame_num][track_id]['bbox']
                            for fill_frame_num in range(frame_num, next_frame_num):
                                if track_id not in frames[fill_frame_num]:
                                    frames[fill_frame_num][track_id] = {}
                                frames[fill_frame_num][track_id]['bbox'] = next_valid_bbox
                            break
            else:
                # Regular interpolation for other bounding boxes
                for prev_frame_num in range(frame_num - 1, -1, -1):
                    if track_id in frames[prev_frame_num] and 'bbox' in frames[prev_frame_num][track_id]:
                        last_valid_bbox = frames[prev_frame_num][track_id]['bbox']
                        break

                if last_valid_bbox is None:
                    continue

                next_valid_bbox = None
                for next_frame_num in range(frame_num + 1, len(frames)):
                    if track_id in frames[next_frame_num] and 'bbox' in frames[next_frame_num][track_id]:
                        next_valid_bbox = frames[next_frame_num][track_id]['bbox']
                        missing_count = next_frame_num - frame_num - 1
                        break

                if next_valid_bbox is not None and missing_count > 0 and missing_count <= max_missing_frames:
                    interpolated_bboxes = self.interpolate_bbox(last_valid_bbox, next_valid_bbox, missing_count)
                    for i in range(1, missing_count + 1):
                        interpolated_frame_num = frame_num + i
                        if track_id not in frames[interpolated_frame_num]:
                            frames[interpolated_frame_num][track_id] = {}
                        frames[interpolated_frame_num][track_id]['bbox'] = interpolated_bboxes[i - 1]

    def interpolate_bbox_in_half(self, frames, track_id, start_frame, max_missing_frames):
        last_valid_bbox = None

        for prev_frame_num in range(start_frame - 1, -1, -1):
            if track_id in frames[prev_frame_num] and 'bbox' in frames[prev_frame_num][track_id]:
                last_valid_bbox = frames[prev_frame_num][track_id]['bbox']
                break

        if last_valid_bbox is None:
            return

        next_valid_bbox = None
        for next_frame_num in range(start_frame + 1, len(frames)):
            if track_id in frames[next_frame_num] and 'bbox' in frames[next_frame_num][track_id]:
                next_valid_bbox = frames[next_frame_num][track_id]['bbox']
                missing_count = next_frame_num - start_frame - 1
                break

        if next_valid_bbox is not None:
            interpolated_bboxes = self.interpolate_bbox(last_valid_bbox, next_valid_bbox, missing_count)
            for i in range(1, missing_count + 1):
                interpolated_frame_num = start_frame + i
                if track_id not in frames[interpolated_frame_num]:
                    frames[interpolated_frame_num][track_id] = {}
                frames[interpolated_frame_num][track_id]['bbox'] = interpolated_bboxes[i - 1]

    def interpolate_bbox(self, bbox1, bbox2, num_interpolations):
        bboxes = []
        for i in range(1, num_interpolations + 1):
            alpha = i / (num_interpolations + 1)
            interpolated_bbox = [
                bbox1[j] * (1 - alpha) + bbox2[j] * alpha for j in range(4)
            ]
            bboxes.append(interpolated_bbox)
        return bboxes

    def clear_and_fix_unwanted_intersections(self, tracks):
        for frame_num in range(len(tracks["Key Points"])):
            has_bbox = False
            for obj_class, frames in tracks.items():
                if obj_class in ["Key Points"]:
                    continue
                if any("bbox" in track_info for track_info in frames[frame_num].values()):
                    has_bbox = True
                    break

            if not has_bbox:
                tracks["Key Points"][frame_num].clear()

        return tracks

    def interpolate_intersections(self, tracks, max_missing_frames=15):
        for frame_num in range(len(tracks["Key Points"])):
            self.interpolate_frame_intersections(tracks, frame_num, max_missing_frames)
        self.ensure_intersections_within_field_boxes(tracks)
        return tracks

    def interpolate_frame_intersections(self, tracks, frame_num, max_missing_frames):
        current_intersections = tracks["Key Points"][frame_num]["points"]
        
        for point_idx, point in enumerate(current_intersections):
            if not point:
                missing_count = 0
                last_valid_point = None

                # Find the previous valid point
                for prev_frame_num in range(frame_num - 1, -1, -1):
                    if tracks["Key Points"][prev_frame_num]["points"]:
                        last_valid_point = tracks["Key Points"][prev_frame_num]["points"][point_idx]
                        break

                # If there is no previous valid point, continue to the next point
                if last_valid_point is None:
                    continue

                # Find the next valid point
                next_valid_point = None
                for next_frame_num in range(frame_num + 1, len(tracks["Key Points"])):
                    if tracks["Key Points"][next_frame_num]["points"]:
                        next_valid_point = tracks["Key Points"][next_frame_num]["points"][point_idx]
                        missing_count = next_frame_num - frame_num - 1
                        break

                # Interpolate if missing frames are within the threshold
                if next_valid_point is not None and missing_count > 0 and missing_count <= max_missing_frames:
                    interpolated_points = self.interpolate_point(last_valid_point, next_valid_point, missing_count)
                    for i in range(1, missing_count + 1):
                        interpolated_frame_num = frame_num + i
                        if "points" not in tracks["Key Points"][interpolated_frame_num]:
                            tracks["Key Points"][interpolated_frame_num]["points"] = []
                        tracks["Key Points"][interpolated_frame_num]["points"].append(interpolated_points[i - 1])

    def interpolate_point(self, point1, point2, num_interpolations):
        points = []
        for i in range(1, num_interpolations + 1):
            alpha = i / (num_interpolations + 1)
            interpolated_point = [
                int(point1[j] * (1 - alpha) + point2[j] * alpha) for j in range(2)
            ]
            points.append(interpolated_point)
        return points

    def ensure_intersections_within_field_boxes(self, tracks):
        for frame_num in range(len(tracks["Key Points"])):
            intersections = tracks["Key Points"][frame_num]["points"]
            first_half_boxes = tracks["First Half Field"][frame_num]
            second_half_boxes = tracks["Second Half Field"][frame_num]

            # Get all bounding boxes for the current frame
            field_boxes = list(first_half_boxes.values()) + list(second_half_boxes.values())

            # Filter intersections to ensure they are within any field box
            filtered_intersections = []
            for point in intersections:
                if any(self.is_point_in_bbox(point, bbox["bbox"]) for bbox in field_boxes):
                    filtered_intersections.append(point)

            tracks["Key Points"][frame_num]["points"] = filtered_intersections

    def is_point_in_bbox(self, point, bbox):
        x, y = point
        x1, y1, x2, y2 = map(int, bbox)
        return x1 <= x <= x2 and y1 <= y <= y2

    def smooth_bboxes(self, tracks, threshold=0.5, max_missing_frames=15):
        for obj_class in tracks.keys():
            if obj_class in ["players", "referees", "ball"]:
                continue
            frames = tracks[obj_class]
            for track_id in self.get_all_track_ids(frames):
                self.interpolate_anomalous_bboxes(frames, track_id, threshold, max_missing_frames)
        return tracks

    def interpolate_anomalous_bboxes(self, frames, track_id, threshold, max_missing_frames):
        bbox_areas = self.calculate_bbox_areas(frames, track_id)
        
        for frame_num in range(1, len(frames) - 1):
            if track_id not in frames[frame_num] or 'bbox' not in frames[frame_num][track_id]:
                continue
            
            prev_area = bbox_areas[frame_num - 1]
            curr_area = bbox_areas[frame_num]
            next_area = bbox_areas[frame_num + 1]
            
            if self.is_anomalous(prev_area, curr_area, next_area, threshold):
                self.interpolate_track(frames, track_id, max_missing_frames)
                break

    def calculate_bbox_areas(self, frames, track_id):
        areas = []
        for frame_data in frames:
            if track_id in frame_data and 'bbox' in frame_data[track_id]:
                bbox = frame_data[track_id]['bbox']
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                areas.append(area)
            else:
                areas.append(0)
        return areas

    def is_anomalous(self, prev_area, curr_area, next_area, threshold):
        return (abs(curr_area - prev_area) / prev_area > threshold and
                abs(curr_area - next_area) / next_area > threshold)

