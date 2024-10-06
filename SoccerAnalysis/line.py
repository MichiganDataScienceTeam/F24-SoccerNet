import cv2
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.linear_model import LinearRegression
import pickle
import os 
class WhiteLineDetector:
    def __init__(self):
        self.intersections = []

    # This is the driver and collects all intersections points 
    def get_intersections(self, frames, tracks, save_path=None, load_path=None):
        if load_path and os.path.exists(load_path):
            with open(load_path, 'rb') as f:
                loaded_tracks = pickle.load(f)
            print(f"Loaded tracks from {load_path}")
            return loaded_tracks

        all_intersections = []
        for frame_num, frame in enumerate(frames):
            green_colors = self.get_dominant_green_colors(frame)

            edges = self.detect_edges(frame)
            lines = self.detect_lines(edges)
            filtered_lines = self.filter_lines(lines, frame, green_colors)
            intersections = self.find_intersections(filtered_lines)
            combined_intersections = self.combine_close_intersections(intersections)
            
            # Filter out intersections inside player bounding boxes
            first_half_dict = tracks["First Half Field"][frame_num]
            second_half_dict = tracks["Second Half Field"][frame_num]
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            filtered_intersections = self.filter_intersections(combined_intersections, player_dict, referee_dict, first_half_dict, second_half_dict)
            all_intersections.append(filtered_intersections)
                
            tracks["Key Points"][frame_num]["points"] = filtered_intersections
            tracks["Key Points"][frame_num]["lines"] = filtered_lines
            tracks["Key Points"][frame_num]["edges"] = edges

        self.intersections = all_intersections

        # Save tracks to a file if specified
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(tracks, f)
            print(f"Saved tracks to {save_path}")

        return tracks

    # Finds a large list of intersection points
    def find_intersections(self, lines):
        intersections = []
        if lines is None:
            return intersections
        
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                x1, y1, x2, y2 = lines[i][0]
                x3, y3, x4, y4 = lines[j][0]
                
                # Calculate the intersection point
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if denom == 0:
                    continue
                
                intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
                intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
                
                if (min(x1, x2) <= intersect_x <= max(x1, x2) and
                    min(y1, y2) <= intersect_y <= max(y1, y2) and
                    min(x3, x4) <= intersect_x <= max(x3, x4) and
                    min(y3, y4) <= intersect_y <= max(y3, y4)):
                    
                    angle = self.angle_between_lines(lines[i], lines[j])
                    if 20 <= angle <= 135:  # Only keep intersections close to 90 degrees
                        point = (int(intersect_x), int(intersect_y))
                        intersections.append(point)
        
        self.intersections.extend(intersections)
        return intersections


    def detect_edges(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Apply morphological operations to clean up the edges
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges

    # detects hough lines
    def detect_lines(self, edges):
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=20)
        return lines

    # will ge tthe dominant green colors on the field
    def get_dominant_green_colors(self, image, k=2):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3)

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(pixels)

        dominant_colors = kmeans.cluster_centers_
        dominant_colors = np.array(dominant_colors, dtype=np.uint8)
        
        # Filter out non-green colors
        green_colors = []
        for color in dominant_colors:
            h, s, v = color
            if 30 <= h <= 90 and s > 40 and v > 40:  # Roughly the HSV range for green
                green_colors.append(color)

        return green_colors

    # Returns true if a point is near green colors 
    def is_near_green(self, image, x1, y1, x2, y2, green_colors):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        masks = [cv2.inRange(hsv, color - np.array([10, 50, 50]), color + np.array([10, 255, 255])) for color in green_colors]
        
        if len(masks) == 0:
            return False
        elif len(masks) == 1:
            mask = masks[0]
        else:
            mask = cv2.bitwise_or(masks[0], masks[1])
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        
        line_image = np.zeros_like(mask)
        cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
        
        masked_line = cv2.bitwise_and(mask, mask, mask=line_image)
        green_pixels = cv2.countNonZero(masked_line)
        return green_pixels > 0

    # Returns true if a point is close to an edge
    def is_close_to_edge(self, image, x1, y1, x2, y2, edge_threshold=5):
        height, width = image.shape[:2]
        return (x1 < edge_threshold or x2 < edge_threshold or
                y1 < edge_threshold or y2 < edge_threshold or
                x1 > width - edge_threshold or x2 > width - edge_threshold or
                y1 > height - edge_threshold or y2 > height - edge_threshold)

    # filters all detected lines that are no near green or close to edges
    def filter_lines(self, lines, image, green_colors):
        filtered_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if (self.is_near_green(image, x1, y1, x2, y2, green_colors) and
                    not self.is_close_to_edge(image, x1, y1, x2, y2)):
                    filtered_lines.append(line)
        return filtered_lines

    # calculate angles between lines using vectors 
    def angle_between_lines(self, line1, line2):
        x1, y1, x2, y2 = line1[0]
        x3, y3, x4, y4 = line2[0]
        
        vec1 = np.array([x2 - x1, y2 - y1])
        vec2 = np.array([x4 - x3, y4 - y3])
        
        unit_vec1 = vec1 / np.linalg.norm(vec1)
        unit_vec2 = vec2 / np.linalg.norm(vec2)
        
        # Calculate the dot product and the angle
        dot_product = np.dot(unit_vec1, unit_vec2)
        angle = np.arccos(dot_product) * (180 / np.pi)
        
        return angle

    # Combines overlapping intersections 
    def combine_close_intersections(self, intersections, distance_threshold=80):
        if len(intersections) == 0:
            return intersections

        dbscan = DBSCAN(eps=distance_threshold, min_samples=1)
        intersections = np.array(intersections)
        labels = dbscan.fit_predict(intersections)

        combined_intersections = []
        for label in set(labels):
            cluster_points = intersections[labels == label]
            cluster_center = cluster_points.mean(axis=0)
            combined_intersections.append(tuple(cluster_center.astype(int)))

        return combined_intersections

    # Draws intersection points on images 
    def draw_intersections(self, image, intersections):
        for point in intersections:
            cv2.circle(image, point, 5, (0, 0, 255), -1)
    
    # Returns true if a point is in a bbox 
    def is_point_in_bbox(self, point, bbox, margin=8):
        x, y = point
        x1, y1, x2, y2 = map(int, bbox)
        return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)

    # Can be used to get a single image 
    # Only runs on this class locally
    def process_image(self, frame):
        all_intersections = []
        green_colors = self.get_dominant_green_colors(frame)

        edges = self.detect_edges(frame)
        lines = self.detect_lines(edges)
        filtered_lines = self.filter_lines(lines, frame, green_colors)
        intersections = self.find_intersections(filtered_lines)
        combined_intersections = self.combine_close_intersections(intersections)
            
        # Filter out intersections inside player bounding boxes
        all_intersections.append(combined_intersections)

        self.intersections = all_intersections

        # Draw the intersections as large purple circles
        for intersection in combined_intersections:
            cv2.circle(frame, (int(intersection[0]), int(intersection[1])), 10, (255, 0, 255), -1) # 10 is the radius, (255, 0, 255) is the color purple in BGR, -1 is the fill thickness

        return frame
    
    # filters intersections to be in halfs and outside players and refs 
    def filter_intersections(self, intersections, player_dict, referee_dict, first_half_dict, second_half_dict):
        filtered_intersections = []
        for point in intersections:
            inside_half = any(
                self.is_point_in_bbox(point, obj["bbox"])
                for half_dict in [first_half_dict, second_half_dict]
                for obj in half_dict.values()
            )

            if not inside_half:
                continue

            inside_bbox = any(
                self.is_point_in_bbox(point, obj["bbox"])
                for bbox_dict in [player_dict, referee_dict]
                for obj in bbox_dict.values()
            )

            if not inside_bbox:
                filtered_intersections.append(point)

        return filtered_intersections

    
    # Draws all collected and relevant intersections on the tracks 
    def draw_frame_intersections(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            intersection_points = tracks["Key Points"][frame_num]["points"]

            #These are all intersections and will show up in purple
            #for point in intersection_points:
            #    cv2.circle(frame, tuple(point), 20, (255, 0, 255), -1)

            if "Top Circle Point" in tracks["Key Points"][frame_num]:
                top_circle_point = tracks["Key Points"][frame_num]["Top Circle Point"]
                if top_circle_point is not None:
                    top_circle_point = tuple(map(int,top_circle_point))
                    cv2.circle(frame, top_circle_point, 20, (0, 255, 0), -1)
                    cv2.putText(frame, "Top Circle Point", (top_circle_point[0], top_circle_point[1] + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            if "Bottom Circle Point" in tracks["Key Points"][frame_num]:
                bottom_circle_point = tracks["Key Points"][frame_num]["Bottom Circle Point"]
                if bottom_circle_point is not None:
                    bottom_circle_point = tuple(map(int, bottom_circle_point))
                    cv2.circle(frame, bottom_circle_point, 20, (0, 255, 0), -1)
                    cv2.putText(frame, "Bottom Circle Point", (bottom_circle_point[0], bottom_circle_point[1] + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
            if "Left Circle Point" in tracks["Key Points"][frame_num]:
                left_circle_point = tracks["Key Points"][frame_num]["Left Circle Point"]
                if left_circle_point is not None:
                    left_circle_point = tuple(map(int, left_circle_point))
                    cv2.circle(frame, left_circle_point, 20, (0, 165, 255), -1)
                    cv2.putText(frame, "Left Circle Point", (left_circle_point[0], left_circle_point[1] + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2, cv2.LINE_AA)
                    
            if "Right Circle Point" in tracks["Key Points"][frame_num]:
                right_circle_point = tracks["Key Points"][frame_num]["Right Circle Point"]
                if right_circle_point is not None:
                    right_circle_point = tuple(map(int, right_circle_point))
                    cv2.circle(frame, right_circle_point, 20, (0, 165, 255), -1)
                    cv2.putText(frame, "Right Circle Point", (right_circle_point[0], right_circle_point[1] + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2, cv2.LINE_AA)
            
            if "Center Circle Point" in tracks["Key Points"][frame_num]:
                center_circle_point = tracks["Key Points"][frame_num]["Center Circle Point"]
                if center_circle_point is not None:
                    center_circle_point = tuple(map(int, center_circle_point))
                    cv2.circle(frame, center_circle_point, 20, (0, 255, 255), -1)
                    cv2.putText(frame, "Center Circle Point", (center_circle_point[0], center_circle_point[1] + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                    
            if "Top Center Point" in tracks["Key Points"][frame_num]:
                top_center_point = tracks["Key Points"][frame_num]["Top Center Point"]
                if top_center_point is not None:
                    top_center_point = tuple(map(int, top_center_point))
                    cv2.circle(frame, top_center_point, 20, (0, 255, 0), -1)
                    cv2.putText(frame, "Top Center Point", (top_center_point[0], top_center_point[1] + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
            if "Bottom Center Point" in tracks["Key Points"][frame_num]:
                bottom_center_point = tracks["Key Points"][frame_num]["Bottom Center Point"]
                if bottom_center_point is not None:
                    bottom_center_point = tuple(map(int, bottom_center_point))
                    cv2.circle(frame, bottom_center_point, 20, (0, 255, 0), -1)
                    cv2.putText(frame, "Bottom Center Point", (top_center_point[0], bottom_center_point[1] + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
            if "Right Bottom 18Yard Circle Point" in tracks["Key Points"][frame_num]:
                bottom_18_yard_circle_point = tracks["Key Points"][frame_num]["Right Bottom 18Yard Circle Point"]
                if bottom_18_yard_circle_point is not None:
                    bottom_18_yard_circle_point = tuple(map(int,bottom_18_yard_circle_point))
                    cv2.circle(frame, bottom_18_yard_circle_point, 20, (0, 255, 0), -1)
                    cv2.putText(frame, "Right Bottom 18Yard Circle Point", (bottom_18_yard_circle_point[0], bottom_18_yard_circle_point[1] + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
            if "Right Top 18Yard Circle Point" in tracks["Key Points"][frame_num]:
                top_18_yard_circle_point = tracks["Key Points"][frame_num]["Right Top 18Yard Circle Point"]
                if top_18_yard_circle_point is not None:
                    top_18_yard_circle_point = tuple(map(int,top_18_yard_circle_point))
                    cv2.circle(frame, top_18_yard_circle_point, 20, (0, 255, 0), -1)
                    cv2.putText(frame, "Right Top 18Yard Circle Point", (top_18_yard_circle_point[0], top_18_yard_circle_point[1] + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
            if "Right Top 18Yard Point" in tracks["Key Points"][frame_num]:
                top_18_yard_point = tracks["Key Points"][frame_num]["Right Top 18Yard Point"]
                if top_18_yard_point is not None:
                    top_18_yard_point = tuple(map(int,top_18_yard_point))
                    cv2.circle(frame, top_18_yard_point, 20, (0, 255, 0), -1)
                    cv2.putText(frame, "Right Top 18Yard Point", (top_18_yard_point[0], top_18_yard_point[1] + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                 
            if "Right Top 5Yard Point" in tracks["Key Points"][frame_num]:
                top_5_yard_point = tracks["Key Points"][frame_num]["Right Top 5Yard Point"]
                if top_5_yard_point is not None:
                    top_5_yard_point = tuple(map(int,top_5_yard_point))
                    cv2.circle(frame, top_5_yard_point, 20, (0, 255, 0), -1)
                    cv2.putText(frame, "Right Top 5Yard Point", (top_5_yard_point[0], top_5_yard_point[1] + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
            if "Right Bottom 18Yard Point" in tracks["Key Points"][frame_num]:
                right_bottom_18yard_point = tracks["Key Points"][frame_num]["Right Bottom 18Yard Point"]
                if right_bottom_18yard_point is not None:
                    right_bottom_18yard_point = tuple(map(int, right_bottom_18yard_point))
                    cv2.circle(frame, right_bottom_18yard_point, 20, (0, 255, 0), -1)
                    cv2.putText(frame, "Right Bottom 18Yard Point", (right_bottom_18yard_point[0], right_bottom_18yard_point[1] + 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            output_frames.append(frame)
        
        return output_frames
    
    # Can be used for drawing the lines and contours for debugging
    def draw_frame_lines_and_contours(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            key_points = tracks["Key Points"][frame_num]

            # Draw edges
            #edges = key_points["edges"]
            #contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #cv2.drawContours(frame, contours, -1, (255, 255, 255), 1)

            # Draw lines
            lines = key_points["lines"]
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw contour or line of best fit
            boundary_points = key_points["contour"]
            self.draw_field_boundary(frame, boundary_points[0])
            boundary_points = key_points["line_of_best_fit"]
            self.draw_line_of_best_fit(frame, boundary_points[0], boundary_points[1])
            '''
            if boundary_points is not None:
                if boundary_points[-1] == 'contour':
                    self.draw_field_boundary(frame, boundary_points[0])
                    print("hello contour")
                elif boundary_points[-1] == 'line_of_best_fit':
                    self.draw_line_of_best_fit(frame, boundary_points[0], boundary_points[1])
                    print("hello line of best fit")
            '''
            output_frames.append(frame)
        
        return output_frames
    
    # Calculate distances between intersection points 
    def calculate_distances(self):
        distances = []
        for i in range(1, len(self.intersections)):
            x1, y1 = self.intersections[i-1]
            x2, y2 = self.intersections[i]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distances.append(distance)
        return distances

# Example usage with a single image

def main():
    image_path = 'SoccerTestImage4.png'
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return

    detector = WhiteLineDetector()
    processed_image = detector.process_image(image)
    
    cv2.imshow('Processed Image', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
