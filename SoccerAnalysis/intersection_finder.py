import bboxUtils
import pandas as pd
import numpy as np

class SuperAlgorithm:
    def __init__(self):
        pass


    def find_right_field_intersections(self, tracks):
        first_half = tracks.get("First Half Field", [])
        second_half = tracks.get("Second Half Field", [])
        eighteen_yard_circle = tracks.get("18Yard Circle", [])
        eighteen_yard = tracks.get("18Yard", [])
        five_yard = tracks.get("5Yard", [])

        for frame_num, (first_half_frame, second_half_frame) in enumerate(zip(first_half, second_half)):
            if not first_half_frame and second_half_frame or (first_half_frame and second_half_frame and self.is_bigger_bbox(second_half_frame, first_half_frame)):
                for second_id, second_info in second_half_frame.items():
                    eighteen_yard_frame = eighteen_yard[frame_num]
                    eighteen_yard_circle_frame = eighteen_yard_circle[frame_num]
                    five_yard_frame = five_yard[frame_num]
                    
                    for eighteen_yard_id, eighteen_yard_info in eighteen_yard_frame.items():
                        eighteen_yard_bbox = eighteen_yard_info['bbox']
                        intersections = tracks["Key Points"][frame_num]["points"]
                        top_18yard_point = bboxUtils.get_top_left(eighteen_yard_bbox)
                        top_18yard_point = (top_18yard_point[0], top_18yard_point[1] + 40)
                        if self.is_intersection_present(intersections, top_18yard_point, 100) is not False:
                                tracks["Key Points"][frame_num]["Right Top 18Yard Point"] = self.is_intersection_present(intersections, top_18yard_point, 100)

                    for five_yard_id, five_yard_info in five_yard_frame.items():
                        five_yard_bbox = five_yard_info['bbox']
                        intersections = tracks["Key Points"][frame_num]["points"]
                        top_right_five_yard_point = bboxUtils.get_top_left(five_yard_bbox)
                        top_right_five_yard_point = (top_right_five_yard_point[0], top_right_five_yard_point[1] + 30)
                        if self.is_intersection_present(intersections, top_right_five_yard_point, 100) is not False:
                            tracks["Key Points"][frame_num]["Right Top 5Yard Point"] = self.is_intersection_present(intersections, top_right_five_yard_point, 100)
                        else:
                            tracks["Key Points"][frame_num]["Right Top 5Yard Point"] = top_right_five_yard_point

                    for eighteen_yard_id, eighteen_yard_info in eighteen_yard_frame.items():
                        eighteen_yard_bbox = eighteen_yard_info['bbox']
                        for eighteen_yard_circle_id,eighteen_yard_circle_info in eighteen_yard_circle_frame.items():
                            intersections = tracks["Key Points"][frame_num]["points"]
                            eighteen_yard_circle_bbox = eighteen_yard_circle_info['bbox']
                            if bboxUtils.is_bbox_inside(eighteen_yard_bbox, eighteen_yard_circle_bbox):
                                bottom_18_yard_circle_point = bboxUtils.get_bottom_right(eighteen_yard_circle_bbox)
                                tracks["Key Points"][frame_num]["Right Bottom 18Yard Circle Point"] = bottom_18_yard_circle_point

                    for eighteen_yard_circle_id,eighteen_yard_circle_info in eighteen_yard_circle_frame.items():
                            intersections = tracks["Key Points"][frame_num]["points"]
                            eighteen_yard_circle_bbox = eighteen_yard_circle_info['bbox']
                            top_18yard_circle_point = bboxUtils.get_top_left(eighteen_yard_circle_bbox)
                            top_18yard_circle_point = (top_18yard_circle_point[0] + 40, top_18yard_circle_point[1])
                            if self.is_intersection_present(intersections, top_18yard_circle_point, 150) is not False:
                                tracks["Key Points"][frame_num]["Right Top 18Yard Circle Point"] = self.is_intersection_present(intersections, top_18yard_circle_point, 150)
                            else:
                                top_left_point = bboxUtils.get_top_left(eighteen_yard_circle_bbox)
                                top_right_point = bboxUtils.get_top_right(eighteen_yard_circle_bbox)
                                right_top_18yard_point = tracks["Key Points"][frame_num].get("Right Top 18Yard Point")
                                right_bottom_18yard_circle_point = tracks["Key Points"][frame_num].get("Right Bottom 18Yard Circle Point")
                                if right_top_18yard_point and right_bottom_18yard_circle_point:
                                    # Calculate the intersection point of the two lines: bottom-left to bottom-right and right_top to right_bottom_circle
                                    x1, y1 = top_left_point
                                    x2, y2 = top_right_point
                                    x3, y3 = right_top_18yard_point
                                    x4, y4 = right_bottom_18yard_circle_point

                                    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                                    if denom != 0:
                                        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
                                        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
                                        intersection_point = (int(px), int(py))
                                        tracks["Key Points"][frame_num]["Right Top 18Yard Circle Point"] = intersection_point

                    
                    for eighteen_yard_id, eighteen_yard_info in eighteen_yard_frame.items():
                        eighteen_yard_bbox = eighteen_yard_info['bbox']
                        intersections = tracks["Key Points"][frame_num]["points"]

                        # Get the bottom-left and bottom-right points of the 18-yard bounding box
                        bottom_left_point = bboxUtils.get_bottom_left(eighteen_yard_bbox)
                        bottom_right_point = bboxUtils.get_bottom_right(eighteen_yard_bbox)

                        # Get the "Right Top 18Yard Point" and "Right Bottom 18Yard Circle Point" points
                        right_top_18yard_point = tracks["Key Points"][frame_num].get("Right Top 18Yard Point")
                        right_bottom_18yard_circle_point = tracks["Key Points"][frame_num].get("Right Bottom 18Yard Circle Point")

                        if right_top_18yard_point and right_bottom_18yard_circle_point:
                            # Calculate the intersection point of the two lines
                            x1, y1 = bottom_left_point
                            x2, y2 = bottom_right_point
                            x3, y3 = right_top_18yard_point
                            x4, y4 = right_bottom_18yard_circle_point

                            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                            if denom != 0:
                                px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
                                py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
                                intersection_point = (px, py)

                                # Check if intersection point is within the bounding box formed by the lines
                                x_min = min(x1, x2, x3, x4)
                                x_max = max(x1, x2, x3, x4)
                                y_min = min(y1, y2, y3, y4)
                                y_max = max(y1, y2, y3, y4)

                                if x_min <= px <= x_max and y_min <= py <= y_max:
                                    # Check if intersection point is present in intersections
                                    if self.is_intersection_present(intersections, intersection_point, 150) is not False:
                                        tracks["Key Points"][frame_num]["Right Bottom 18Yard Point"] = self.is_intersection_present(intersections, intersection_point, 150)

        list_of_points = ["Right Top 18Yard Point","Right Top 18Yard Circle Point","Right Bottom 18Yard Circle Point", "Right Top 5Yard Point","Right Bottom 18Yard Point"]
        max_missing_frames = 60
        self.interpolate_points(tracks, max_missing_frames, list_of_points)
        return tracks

    def find_outer_intersections(self,tracks):
        first_half = tracks.get("First Half Field", [])
        second_half = tracks.get("Second Half Field", [])
        for frame_num, (first_half_frame, second_half_frame) in enumerate(zip(first_half, second_half)):
            for first_id, first_info in first_half_frame.items():
                for second_id, second_info in second_half_frame.items():
                    intersections = tracks["Key Points"][frame_num]["points"]
                    if 'bbox' in first_info and 'bbox' in second_info:
                        first_bbox = first_info['bbox']
                        second_bbox = second_info['bbox']
                        first_top_point = bboxUtils.get_top_right(first_bbox)
                        second_top_point = bboxUtils.get_top_left(second_bbox)
                        

                        if self.is_intersection_present(intersections, first_top_point, 70) is not False:
                            tracks["Key Points"][frame_num]["Top Center Point"] = self.is_intersection_present(intersections, first_top_point, 70)
                        else:
                            if self.is_intersection_present(intersections, second_top_point, 70) is not False:
                                tracks["Key Points"][frame_num]["Top Center Point"] = self.is_intersection_present(intersections, second_top_point, 70)
                    
                        first_bottom_point = bboxUtils.get_bottom_right(first_bbox)
                        second_bottom_point = bboxUtils.get_bottom_left(second_bbox)

                        if self.is_intersection_present(intersections, first_bottom_point, 70) is not False:
                            tracks["Key Points"][frame_num]["Bottom Center Point"] = self.is_intersection_present(intersections, first_bottom_point, 70)
                        else:
                            if self.is_intersection_present(intersections, second_bottom_point, 70) is not False:
                                tracks["Key Points"][frame_num]["Bottom Center Point"] = self.is_intersection_present(intersections, second_top_point, 70)
        list_of_points = ["Top Center Point", "Bottom Center Point"]
        max_missing_frames = 60
        self.interpolate_points(tracks, max_missing_frames, list_of_points)

        return tracks
    
    def find_middle_intersections(self, tracks):
        first_half_central = tracks.get("First Half Central Circle", [])
        second_half_central = tracks.get("Second Half Central Circle", [])
        once_or_twice = False

        for frame_num, (first_half_frame, second_half_frame) in enumerate(zip(first_half_central, second_half_central)):
            if not first_half_frame and second_half_frame:
                for second_id, second_info in second_half_frame.items():
                    second_bbox = second_info['bbox']
                    right_circle_point = bboxUtils.get_right_middle(second_bbox)
                    tracks["Key Points"][frame_num]["Right Circle Point"] = tuple(map(int, right_circle_point))
                continue

            if not second_half_frame and first_half_frame:
                for first_id, first_info in first_half_frame.items():
                    first_bbox = first_info['bbox']
                    left_circle_point = bboxUtils.get_left_middle(first_bbox)
                    tracks["Key Points"][frame_num]["Left Circle Point"] = tuple(map(int, left_circle_point))
                continue

            for first_id, first_info in first_half_frame.items():
                for second_id, second_info in second_half_frame.items():
                    if 'bbox' in first_info and 'bbox' in second_info:
                        first_bbox = first_info['bbox']
                        second_bbox = second_info['bbox']

                        #Calculate intersections at the specified points
                        self.add_top_bottom_middle_intersections(tracks, frame_num, first_bbox, second_bbox, once_or_twice)

                        #check widths if they are similar add both intersections
                        if self.are_boxes_similar_width(first_bbox, second_bbox, 30):
                            left_circle_point = bboxUtils.get_left_middle(first_bbox)
                            right_circle_point = bboxUtils.get_right_middle(second_bbox)
                            tracks["Key Points"][frame_num]["Left Circle Point"] = tuple(map(int, left_circle_point))
                            tracks["Key Points"][frame_num]["Right Circle Point"] = tuple(map(int, right_circle_point))
                        else:
                            if bboxUtils.get_bbox_width(first_bbox) > bboxUtils.get_bbox_width(second_bbox):
                                left_circle_point = bboxUtils.get_left_middle(first_bbox)
                                tracks["Key Points"][frame_num]["Left Circle Point"] = tuple(map(int, left_circle_point))
                            else:
                                right_circle_point = bboxUtils.get_right_middle(second_bbox)
                                tracks["Key Points"][frame_num]["Right Circle Point"] = tuple(map(int, right_circle_point))
                        
        #interpolate missing circle points
        once_or_twice = True
        list_of_points = ["Top Circle Point", "Bottom Circle Point", "Left Circle Point", "Right Circle Point"]
        max_missing_frames = 30
        self.interpolate_points(tracks, max_missing_frames, list_of_points) 

        
        for frame_num, (first_half_frame, second_half_frame) in enumerate(zip(first_half_central, second_half_central)):
            if not first_half_frame or not second_half_frame:
                continue

            for first_id, first_info in first_half_frame.items():
                for second_id, second_info in second_half_frame.items():
                    if 'bbox' in first_info and 'bbox' in second_info:
                        first_bbox = first_info['bbox']
                        second_bbox = second_info['bbox']

                        self.add_top_bottom_middle_intersections(tracks, frame_num, first_bbox, second_bbox, once_or_twice)

                        self.add_center_intersection(tracks, frame_num)

        
        return tracks

    def add_center_intersection(self, tracks, frame_num):
        top_circle_point = tracks["Key Points"][frame_num]["Top Circle Point"]
        bottom_circle_point = tracks["Key Points"][frame_num]["Bottom Circle Point"]
        middle_point = bboxUtils.get_midpoint(top_circle_point, bottom_circle_point)
        middle_point = (middle_point[0], middle_point[1] - 15)
        tracks["Key Points"][frame_num]["Center Circle Point"] = middle_point
    
    def add_top_bottom_middle_intersections(self, tracks, frame_num, first_bbox, second_bbox, once_or_twice):
        # Define the top and bottom areas of interest
        top_circle_point = bboxUtils.get_midpoint(bboxUtils.get_top_right(first_bbox), bboxUtils.get_top_left(second_bbox))
        bottom_circle_point = bboxUtils.get_midpoint(bboxUtils.get_bottom_right(first_bbox), bboxUtils.get_bottom_left(second_bbox))
        confirmed_top_circle_point = None
        confirmed_bottom_circle_point = None

        if "Key Points" not in tracks:
            tracks["Key Points"] = []

        intersections = tracks["Key Points"][frame_num]["points"]

        top_val = self.is_intersection_present(intersections, top_circle_point, 50)

        if top_val is False:
            if once_or_twice is True:
                tracks["Key Points"][frame_num]["Top Circle Point"] = tuple(map(int, top_circle_point))
        else:
            if once_or_twice is False:
                confirmed_top_circle_point = top_val
      
        if once_or_twice is False:
            if confirmed_top_circle_point is not None:
                tracks["Key Points"][frame_num]["Top Circle Point"] = tuple(map(int,confirmed_top_circle_point))
            else:
                tracks["Key Points"][frame_num]["Top Circle Point"] = None
        
        bottom_val = self.is_intersection_present(intersections, bottom_circle_point, 50)

        if bottom_val is False:
            if once_or_twice is True:
                tracks["Key Points"][frame_num]["Bottom Circle Point"] = tuple(map(int, bottom_circle_point))
        else:
            if once_or_twice is False:
                confirmed_bottom_circle_point = bottom_val
      
        if once_or_twice is False:
            if confirmed_bottom_circle_point is not None:
                tracks["Key Points"][frame_num]["Bottom Circle Point"] = tuple(map(int,confirmed_bottom_circle_point))
            else:
                tracks["Key Points"][frame_num]["Bottom Circle Point"] = None

    
    def is_intersection_present(self, intersections, point, threshold):
        for intersection in intersections:
            if bboxUtils.measure_distance(intersection, point) <= threshold:
                return intersection
        return False
    
    def interpolate_points(self, tracks, max_missing_frames, list_of_points):
        key_points = tracks.get("Key Points", [])

        for point_name in list_of_points:
            point_series = [frame.get(point_name, None) for frame in key_points]

            point_series_filtered = [
                point if point is not None else [np.nan, np.nan] for point in point_series
            ]

            df_point_series = pd.DataFrame(point_series_filtered, columns=['x', 'y'])
            df_point_series = self.interpolate_within_limits(df_point_series, max_missing_frames)
            interpolated_points = df_point_series.to_numpy().tolist()

            for frame_idx, point in enumerate(interpolated_points):
                if not all(np.isnan(point)):
                    if point_name in key_points[frame_idx]:
                        key_points[frame_idx][point_name] = tuple(map(int, point))
                    else:
                        key_points[frame_idx][point_name] = tuple(map(int, point))

    def interpolate_within_limits(self, df, max_missing_frames):
        nan_groups = df.isna().all(axis=1).astype(int).groupby(df.notna().all(axis=1).cumsum()).cumsum()

        df_interpolated = df.copy()
        for start, end in self.find_nan_gaps(nan_groups, max_missing_frames):
            df_interpolated.iloc[start-1:end+1] = df.iloc[start-1:end+1].interpolate()
        return df_interpolated

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

    def are_boxes_similar_width(self, bbox1, bbox2, amount):
        first_width = bboxUtils.get_bbox_width(bbox1)
        second_width = bboxUtils.get_bbox_width(bbox2)
        return abs(first_width - second_width) <= amount

    def is_bigger_bbox(self, second_half_frame, first_half_frame):
        for second_id, second_info in second_half_frame.items():
            second_bbox = second_info['bbox']
            second_area = self.calculate_bbox_area(second_bbox)
            
            for first_id, first_info in first_half_frame.items():
                first_bbox = first_info['bbox']
                first_area = self.calculate_bbox_area(first_bbox)
                
                if second_area > first_area:
                    return True
        return False

    def calculate_bbox_area(self, bbox):
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width * height
    def find_left_field_intersections(self, tracks):
            # Implementation for future use
            pass

