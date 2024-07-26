import cv2
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.linear_model import LinearRegression
import pickle
import os

class WhiteLineDetector:
    def __init__(self):
        self.intersections = []

    def detect_edges(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Apply morphological operations to clean up the edges
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges

    def detect_lines(self, edges):
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=20)
        return lines

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

    def is_close_to_edge(self, image, x1, y1, x2, y2, edge_threshold=5):
        height, width = image.shape[:2]
        return (x1 < edge_threshold or x2 < edge_threshold or
                y1 < edge_threshold or y2 < edge_threshold or
                x1 > width - edge_threshold or x2 > width - edge_threshold or
                y1 > height - edge_threshold or y2 > height - edge_threshold)

    def filter_lines(self, lines, image, green_colors):
        filtered_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if (self.is_near_green(image, x1, y1, x2, y2, green_colors) and
                    not self.is_close_to_edge(image, x1, y1, x2, y2)):
                    filtered_lines.append(line)
        return filtered_lines

    def angle_between_lines(self, line1, line2):
        # Calculate the angle between two lines
        x1, y1, x2, y2 = line1[0]
        x3, y3, x4, y4 = line2[0]
        
        # Direction vectors of the lines
        vec1 = np.array([x2 - x1, y2 - y1])
        vec2 = np.array([x4 - x3, y4 - y3])
        
        # Normalize the direction vectors
        unit_vec1 = vec1 / np.linalg.norm(vec1)
        unit_vec2 = vec2 / np.linalg.norm(vec2)
        
        # Calculate the dot product and the angle
        dot_product = np.dot(unit_vec1, unit_vec2)
        angle = np.arccos(dot_product) * (180 / np.pi)
        
        return angle

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

    def draw_intersections(self, image, intersections):
        for point in intersections:
            cv2.circle(image, point, 5, (0, 0, 255), -1)

    def find_field_boundary(self, image, green_colors):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        masks = [cv2.inRange(hsv, color - np.array([10, 50, 50]), color + np.array([10, 255, 255])) for color in green_colors]
        if len(masks) == 0:
            return None
        elif len(masks) == 1:
            mask = masks[0]
        else:
            mask = cv2.bitwise_or(masks[0], masks[1])

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # Apply a large dilation to create a thick boundary
        kernel = np.ones((30, 30), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        # Use Canny edge detector to find edges
        edges = cv2.Canny(dilated_mask, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Return the longest contour which should be the field boundary
            return max(contours, key=lambda c: cv2.arcLength(c, True))
        return None

    def get_line_of_best_fit(self, contour, image_shape):
        x = contour[:, 0, 0].reshape(-1, 1)
        y = contour[:, 0, 1].reshape(-1, 1)

        model = LinearRegression()
        model.fit(x, y)

        # Find the highest point
        highest_point = tuple(contour[contour[:, :, 1].argmin()][0])

        # Determine the x-coordinate of the highest point
        highest_x = highest_point[0]

        # Predict y-values at the left and right edges of the image
        y_pred_left = model.predict(np.array([[0]]))[0][0]
        y_pred_right = model.predict(np.array([[image_shape[1]]]))[0][0]

        # Determine the y-value at the highest point using the linear model
        y_pred_highest = model.predict(np.array([[highest_x]]))[0][0]

        # Calculate the offset between the actual highest point y-value and the predicted y-value
        offset = highest_point[1] - y_pred_highest

        # Adjust the predicted y-values using the offset
        y_pred_left += offset
        y_pred_right += offset

        start_point = (0, int(y_pred_left))
        end_point = (image_shape[1], int(y_pred_right))

        return start_point, end_point, lambda x, y: y >= model.predict([[x]])[0][0] + offset
    

    def is_contour_close_to_edges(self, image, contour, edge_threshold=5):
        height, width = image.shape[:2]

        # Find the leftmost and rightmost points
        leftmost_point = tuple(contour[contour[:, :, 0].argmin()][0])
        rightmost_point = tuple(contour[contour[:, :, 0].argmax()][0])
        
        return ((leftmost_point[0] < edge_threshold or leftmost_point[0] > width - edge_threshold or
                leftmost_point[1] < edge_threshold or leftmost_point[1] > height - edge_threshold) and
                (rightmost_point[0] < edge_threshold or rightmost_point[0] > width - edge_threshold or
                rightmost_point[1] < edge_threshold or rightmost_point[1] > height - edge_threshold))
    def draw_field_boundary(self, image, field_contour):
        cv2.drawContours(image, [field_contour], -1, (255, 0, 255), 3)  # Draw boundary in purple (BGR: 255, 0, 255)

    def draw_line_of_best_fit(self, image, start_point, end_point):
        cv2.line(image, start_point, end_point, (0, 255, 255), 3)  # Draw line in yellow (BGR: 0, 255, 255)
    def is_point_in_bbox(self, point, bbox, margin=5):
        x, y = point
        x1, y1, x2, y2 = map(int, bbox)
        return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


    def process_image(self, image):
        # Detect the field boundary first
        green_colors = self.get_dominant_green_colors(image)
        field_contour = self.find_field_boundary(image, green_colors)

        boundary_func = lambda x, y: True  # Default to consider all lines
        boundary_points = None

        if field_contour is not None:
            boundary_y = np.min(field_contour[:, 0, 1])
            boundary_func = lambda x, y: y >= boundary_y
            boundary_points = (field_contour, 'contour')
            if self.is_contour_close_to_edges(image, field_contour):
            #Find the boundary function for filtering
                boundary_y = np.min(field_contour[:, 0, 1])
                boundary_func = lambda x, y: y >= boundary_y
                boundary_points = (field_contour, 'contour')
                #cv2.line(image, (0, boundary_y), (image.shape[1], boundary_y), (255, 0, 0), 2)
            else:
                start_point, end_point, boundary_func = self.get_line_of_best_fit(field_contour, image.shape)
                boundary_points = (start_point, end_point, 'line_of_best_fit')
        # Process the image to detect edges and lines
        edges = self.detect_edges(image)
        lines = self.detect_lines(edges)
        filtered_lines = self.filter_lines(lines, image, green_colors, boundary_func)

        # Draw the filtered lines
        if filtered_lines is not None:
            for line in filtered_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Find and draw intersections
        combined_intersections = self.find_intersections(filtered_lines)
        combined_intersections = self.combine_close_intersections(combined_intersections)
        self.draw_intersections(image, combined_intersections)
        
        # Draw the field boundary last
        if boundary_points is not None:
            if boundary_points[-1] == 'contour':
                self.draw_field_boundary(image, boundary_points[0])
            elif boundary_points[-1] == 'line_of_best_fit':
                self.draw_line_of_best_fit(image, boundary_points[0], boundary_points[1])
                print()

        
        return image, combined_intersections
    def filter_intersections(self, intersections, player_dict, referee_dict):
        filtered_intersections = []
        for point in intersections:
            inside_bbox = False
            for bbox_dict in [player_dict, referee_dict]:
                for obj in bbox_dict.values():
                    bbox = obj["bbox"]
                    if self.is_point_in_bbox(point, bbox):
                        inside_bbox = True
                        break
                if inside_bbox:
                    break
            if not inside_bbox:
                filtered_intersections.append(point)
        return filtered_intersections

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
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            filtered_intersections = self.filter_intersections(combined_intersections, player_dict, referee_dict)
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
    
    def draw_frame_intersections(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            intersection_points = tracks["Key Points"][frame_num]["points"]
            print(intersection_points)

            for point in intersection_points:
                cv2.circle(frame, tuple(point), 20, (255, 0, 255), -1)
            
            output_frames.append(frame)
        
        return output_frames
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
    image_path = ''
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return

    detector = WhiteLineDetector()
    processed_image, intersections = detector.process_image(image)
    
    cv2.imshow('Processed Image', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()